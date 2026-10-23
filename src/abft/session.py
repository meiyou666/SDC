"""Caller-owned method instances, physical call counters, and checkpoint state.

History is isolated by configuration, layer/role, device, precision, shape and
configured scale profile. Dynamic amax values stay on device in each call's
operands; grouping never reads a tensor to the host. Stateful methods own their
RNG through state_dict/load_state_dict and must initialize device state inside
prepare, on the session's stream. Per-call evidence/scratch belongs to execute.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from threading import Lock

import torch

from .config import parse_abft_config, parse_fault_config
from .methods import get_method


SCHEMA_VERSION = 1
CONFIG_VERSION = 1
_SCOPE_FIELDS = {"layer", "role", "precision", "device", "m", "n", "k", "scale_profile"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clone_state(value, parents=None):
    """Freeze plain weights-only-compatible data without retaining autograd graphs."""
    if isinstance(value, torch.Tensor):
        copied = value.detach().clone()
        if type(copied) is not torch.Tensor:
            raise ValueError("ABFT state must use ordinary tensors, not tensor subclasses")
        return copied
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) not in (dict, list, tuple):
        raise ValueError("ABFT state must contain only dictionaries, lists, tuples, scalars and tensors; "
                         "save generator.get_state(), not a generator object")
    parents = set() if parents is None else parents
    if id(value) in parents:
        raise ValueError("ABFT state must not contain cycles")
    parents.add(id(value))
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise ValueError("ABFT state dictionary keys must be strings")
            return {key: _clone_state(item, parents) for key, item in value.items()}
        items = [_clone_state(item, parents) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    finally:
        parents.remove(id(value))


def _method_metadata(method, name):
    if method is None:
        return {"name": name, "class": None, "stateful": False, "state_schema_version": 1}
    stateful = getattr(method, "stateful", False)
    version = getattr(method, "state_schema_version", 1)
    if type(stateful) is not bool or type(version) is not int or version < 1:
        raise ValueError("ABFT method stateful/state_schema_version metadata is invalid")
    return {"name": name, "class": f"{type(method).__module__}.{type(method).__qualname__}",
            "stateful": stateful, "state_schema_version": version}


def _scope_key(scope):
    if type(scope) is not dict or set(scope) != _SCOPE_FIELDS:
        raise ValueError("ABFT method scope metadata is invalid")
    if not isinstance(scope["layer"], str) or scope["role"] not in {"forward", "dgrad", "wgrad"}:
        raise ValueError("ABFT method layer/role scope is invalid")
    if scope["precision"] not in {"fp32", "bf16", "fp8", "nvfp4"}:
        raise ValueError("ABFT method precision scope is invalid")
    if any(type(scope[key]) is not int or scope[key] <= 0 for key in ("m", "n", "k")):
        raise ValueError("ABFT method shape scope is invalid")
    if not isinstance(scope["device"], str) or str(torch.device(scope["device"])) != scope["device"]:
        raise ValueError("ABFT method device scope is invalid")
    profile = scope["scale_profile"]
    if type(profile) is not dict or set(profile) != {"scaling", "global_scale", "input_dtypes"}:
        raise ValueError("ABFT scale profile metadata is invalid")
    expected = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}
    if profile["scaling"] != expected[scope["precision"]]:
        raise ValueError("ABFT precision/scale profile mismatch")
    scale = profile["global_scale"]
    if scale != "amax" and (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("ABFT global scale profile is invalid")
    dtypes = profile["input_dtypes"]
    if not isinstance(dtypes, list) or len(dtypes) != 2 or not all(isinstance(x, str) for x in dtypes):
        raise ValueError("ABFT input dtype profile is invalid")
    return (scope["layer"], scope["role"], scope["precision"], scope["device"],
            scope["m"], scope["n"], scope["k"], profile["scaling"], scale, *dtypes)


class MethodSession:
    """Retain methods; isolate stateful profiles and reuse one stateless instance.

    Every product request, including activation-checkpoint replay, reserves a
    new call_index. Internal recovery uses the same lease/index; execute owns
    its frozen evidence and once-only final commit. No logical replay ID is
    inferred. Stateful calls reject concurrent host entry and stream changes;
    stateless Classic/Sum have neither stream queries nor synchronization.

    Callers may explicitly snapshot with state_dict and restore with
    load_state_dict for checkpoint resume or paired experiments. A snapshot
    freezes method history/RNG and the physical request counter; it does not
    freeze input tensors or reset an experiment protocol implicitly.
    """

    def __init__(self, config=None, fault=None):
        self.config = parse_abft_config(json.loads(_json(asdict(parse_abft_config(config)))))
        self.fault = parse_fault_config(json.loads(_json(asdict(parse_fault_config(fault)))))
        self.config_signature = hashlib.sha256(_json({
            "version": CONFIG_VERSION, "abft": asdict(self.config), "fault": asdict(self.fault),
        }).encode()).hexdigest()
        self._pending = get_method(self.config.method) if self.config.method != "none" else None
        self.metadata = _method_metadata(self._pending, self.config.method)
        self.stateful = self.metadata["stateful"]
        self.call_index = 0
        self._entries = {}
        self._streams = {}
        self._lock = Lock() if self.stateful else None

    @contextmanager
    def _exclusive(self):
        if self._lock is None:
            yield
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Stateful ABFT session does not support concurrent calls or snapshots")
        try:
            yield
        finally:
            self._lock.release()

    def _bind_stream(self, device):
        if device.type == "cpu":
            return
        if device.type != "cuda":
            raise RuntimeError("Stateful ABFT sessions currently support CPU or one CUDA stream per device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Stateful ABFT history cannot be advanced by CUDA Graph replay")
        stream = torch.cuda.current_stream(device)
        previous = self._streams.get(str(device))
        if previous is not None and previous.cuda_stream != stream.cuda_stream:
            raise RuntimeError("Stateful ABFT history cannot be shared across CUDA streams")
        self._streams[str(device)] = stream

    @contextmanager
    def call(self, a, b, precision, scaling, global_scale, layer, role):
        if not self.stateful:
            # No history can leak across profiles. Keep the default training
            # path free of scope validation, registry lookup, locks and CUDA
            # queries; execute validates the operands before dispatch.
            self.call_index += 1
            yield self._pending, {
                "layer": layer, "role": role, "call_index": self.call_index,
                "execution_kind": "forward_or_recompute" if role == "forward" else "backward",
                "retry": 0, "scale_profile": {"scaling": scaling, "global_scale": global_scale,
                                               "input_dtypes": [str(a.dtype), str(b.dtype)]},
            }
            return
        with self._exclusive():
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.device != b.device:
                raise ValueError("ABFT session requires compatible matrices on one device")
            profile = {"scaling": scaling, "global_scale": global_scale,
                       "input_dtypes": [str(a.dtype), str(b.dtype)]}
            scope = {"layer": layer, "role": role, "precision": precision, "device": str(a.device),
                     "m": int(a.shape[0]), "n": int(b.shape[1]), "k": int(a.shape[1]),
                     "scale_profile": profile}
            key = _scope_key(scope)
            if self.stateful:
                self._bind_stream(a.device)
            method = None
            if self.config.method != "none":
                if key not in self._entries:
                    method = self._pending if self._pending is not None else get_method(self.config.method)
                    if _method_metadata(method, self.config.method) != self.metadata:
                        raise ValueError("ABFT method factory changed its state metadata")
                    self._entries[key] = (scope, method)
                    self._pending = None
                method = self._entries[key][1]
            self.call_index += 1
            yield method, {"layer": layer, "role": role, "call_index": self.call_index,
                           "execution_kind": "forward_or_recompute" if role == "forward" else "backward",
                           "retry": 0, "scale_profile": {**profile, "input_dtypes": list(profile["input_dtypes"])}}

    def state_dict(self):
        with self._exclusive():
            instances = []
            if self.stateful:
                for scope, method in self._entries.values():
                    stream = self._streams.get(scope["device"])
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        state = method.state_dict()
                        if type(state) is not dict:
                            raise ValueError("ABFT method state_dict must return a dictionary")
                        instances.append({"scope": json.loads(_json(scope)), "state": _clone_state(state)})
                # GPU clones must finish before torch.save transfers their
                # storage. This is an explicit STATEFUL checkpoint boundary.
                for stream in self._streams.values():
                    stream.synchronize()
            pending = None
            if self.stateful and self._pending is not None:
                pending = self._pending.state_dict()
                if type(pending) is not dict:
                    raise ValueError("ABFT method state_dict must return a dictionary")
                pending = _clone_state(pending)
            return {"schema_version": SCHEMA_VERSION, "config_version": CONFIG_VERSION,
                    "config_signature": self.config_signature, "method_metadata": dict(self.metadata),
                    "call_index": self.call_index, "instances": instances, "pending_state": pending}

    def load_state_dict(self, payload, *, device=None):
        with self._exclusive():
            fields = {"schema_version", "config_version", "config_signature", "method_metadata",
                      "call_index", "instances", "pending_state"}
            if type(payload) is not dict or set(payload) != fields:
                raise ValueError("ABFT session checkpoint schema is invalid")
            metadata = payload["method_metadata"]
            if (type(metadata) is not dict or set(metadata) != set(self.metadata) or
                    type(metadata["stateful"]) is not bool or type(metadata["state_schema_version"]) is not int):
                raise ValueError("ABFT method/state schema does not match checkpoint")
            if (type(payload["schema_version"]) is not int or type(payload["config_version"]) is not int or
                    payload["schema_version"] != SCHEMA_VERSION or payload["config_version"] != CONFIG_VERSION or
                    payload["config_signature"] != self.config_signature or payload["method_metadata"] != self.metadata):
                raise ValueError("ABFT session configuration/method/state schema does not match checkpoint")
            if type(payload["call_index"]) is not int or payload["call_index"] < 0 or type(payload["instances"]) is not list:
                raise ValueError("ABFT session call counter/instances are invalid")
            if not self.stateful and (payload["instances"] or payload["pending_state"] is not None):
                raise ValueError("Stateless ABFT methods must not restore adaptive history")
            expected_device = None if device is None else str(torch.device(device))
            entries = {}
            for item in payload["instances"]:
                if type(item) is not dict or set(item) != {"scope", "state"} or type(item["state"]) is not dict:
                    raise ValueError("ABFT method checkpoint entry is invalid")
                scope = json.loads(_json(item["scope"]))
                key = _scope_key(scope)
                if key in entries or (expected_device is not None and scope["device"] != expected_device):
                    raise ValueError("ABFT checkpoint has duplicate or mismatched device scopes")
                method = get_method(self.config.method)
                if _method_metadata(method, self.config.method) != self.metadata:
                    raise ValueError("ABFT method factory changed its state metadata")
                method.load_state_dict(_clone_state(item["state"]))
                entries[key] = (scope, method)
            pending = self._pending if not self.stateful else None
            if self.stateful and not entries:
                if type(payload["pending_state"]) is not dict:
                    raise ValueError("Unused stateful ABFT method is missing its initial RNG/state")
                pending = get_method(self.config.method)
                if _method_metadata(pending, self.config.method) != self.metadata:
                    raise ValueError("ABFT method factory changed its state metadata")
                pending.load_state_dict(_clone_state(payload["pending_state"]))
            elif payload["pending_state"] is not None:
                raise ValueError("ABFT checkpoint contains incompatible pending method state")
            if self.stateful:
                for stream in self._streams.values():
                    stream.synchronize()
                for name in {scope["device"] for scope, _ in entries.values()}:
                    if torch.device(name).type == "cuda":
                        torch.cuda.current_stream(torch.device(name)).synchronize()
            self._entries, self._pending = entries, pending
            self._streams = {}
            self.call_index = payload["call_index"]
