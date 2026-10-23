"""CPU contracts for training-owned ABFT history; no kernel/performance claims."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from abft import methods
from abft.methods import AbftMethod
from abft.session import MethodSession


class SampleHistory(AbftMethod):
    """Artificial fixture with private RNG and a small rolling observation buffer."""

    stateful = True
    state_schema_version = 1

    def __init__(self):
        self.generator = torch.Generator().manual_seed(31)
        self.count = 0
        self.history = torch.zeros(8)

    def advance(self):
        sample = torch.rand((), generator=self.generator)
        self.history[self.count % len(self.history)] = sample
        self.count += 1
        return sample

    def prepare(self, operands, config, mm=None):
        raise NotImplementedError("Session tests do not execute GEMMs")

    def propagate(self, partials, state):
        raise NotImplementedError("Session tests do not execute GEMMs")

    def verify(self, observed, output, state, config):
        raise NotImplementedError("Session tests do not execute GEMMs")

    def state_dict(self):
        return {"count": self.count, "history": self.history,
                "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.count = state["count"]
        self.history = state["history"]
        self.generator.set_state(state["rng"])


@pytest.fixture
def history_method(monkeypatch):
    monkeypatch.setattr(methods, "_METHODS", dict(methods._METHODS))
    methods.register_method("session_test", SampleHistory)
    return "session_test"


def call(session, *, shape=(3, 2, 4), precision="fp32", scaling="none",
         global_scale="amax", layer="model.layers.0.mlp", role="forward",
         dtype=torch.float32, device="cpu"):
    # Session only consumes metadata. Fake CUDA operands test stream policy
    # without initializing CUDA or requiring a GPU in these CPU contract tests.
    m, n, k = shape
    a = SimpleNamespace(ndim=2, shape=(m, k), device=torch.device(device), dtype=dtype)
    b = SimpleNamespace(ndim=2, shape=(k, n), device=torch.device(device), dtype=dtype)
    return session.call(a, b, precision, scaling, global_scale, layer, role)


def populate(session, **kwargs):
    with call(session, **kwargs) as (method, context):
        sample = method.advance()
    return method, context, sample


def test_same_profile_retains_instance_and_counts_physical_invocations(history_method):
    session = MethodSession(history_method)
    with call(session) as (first, context):
        assert context["call_index"] == 1
        assert context["execution_kind"] == "forward_or_recompute"
        assert context["retry"] == 0
    with call(session) as (second, context):
        assert second is first
        assert context["call_index"] == 2
        assert context["execution_kind"] == "forward_or_recompute"
        # Reserving an invocation must not commit/advance method history.
        assert second.count == 0
    assert session.call_index == 2


@pytest.mark.parametrize("change", [
    {"layer": "model.layers.1.mlp"},
    {"role": "dgrad"},
    {"role": "wgrad"},
    {"shape": (5, 2, 4)},
    {"shape": (3, 5, 4)},
    {"shape": (3, 2, 7)},
    {"precision": "bf16"},
    {"precision": "fp8", "scaling": "tensorwise"},
    {"precision": "nvfp4", "scaling": "block16"},
    {"global_scale": 2.0},
    {"dtype": torch.bfloat16},
])
def test_history_isolated_by_execution_profile(history_method, change):
    session = MethodSession(history_method)
    first, _, _ = populate(session)
    with call(session, **change) as (other, context):
        assert other is not first
        assert other.count == 0
        assert context["execution_kind"] == (
            "backward" if change.get("role") in {"dgrad", "wgrad"}
            else "forward_or_recompute"
        )
    with call(session) as (again, _):
        assert again is first
        assert again.count == 1


def test_device_history_isolated_without_cuda_execution(history_method, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=7))
    session = MethodSession(history_method)
    first, _, _ = populate(session)
    with call(session, device="cuda:0") as (gpu0, _):
        assert gpu0 is not first
    with call(session, device="cuda:1") as (gpu1, _):
        assert gpu1 is not gpu0
    with call(session, device="cuda:0") as (again, _):
        assert again is gpu0


def test_context_mutation_cannot_change_saved_scope(history_method):
    session = MethodSession(history_method)
    with call(session) as (first, context):
        context["scale_profile"]["input_dtypes"][0] = "changed_by_consumer"
        context["scale_profile"]["global_scale"] = 99
    with call(session) as (again, _):
        assert again is first
    scope = session.state_dict()["instances"][0]["scope"]
    assert scope["scale_profile"] == {
        "scaling": "none", "global_scale": "amax",
        "input_dtypes": ["torch.float32", "torch.float32"],
    }


def test_weights_only_checkpoint_restores_history_rng_and_counter(history_method, tmp_path):
    session = MethodSession(history_method)
    original, _, _ = populate(session)
    populate(session)
    populate(session, role="dgrad")
    path = tmp_path / "abft_state.pt"
    torch.save(session.state_dict(), path)
    checkpoint = torch.load(path, weights_only=True)

    resumed = MethodSession(history_method)
    resumed.load_state_dict(checkpoint, device="cpu")
    assert resumed.call_index == 3
    with call(resumed) as (restored, context):
        assert restored is not original
        assert restored.count == 2
        assert context["call_index"] == 4
        torch.testing.assert_close(restored.history, original.history, rtol=0, atol=0)
        torch.testing.assert_close(restored.advance(), original.advance(), rtol=0, atol=0)
    with call(resumed, role="dgrad") as (backward, context):
        assert backward.count == 1
        assert context["call_index"] == 5
    assert checkpoint["instances"][0]["state"]["rng"].device.type == "cpu"


def test_snapshots_and_restored_state_do_not_alias_live_tensors(history_method):
    session = MethodSession(history_method)
    original, _, _ = populate(session)
    frozen = session.state_dict()
    saved_history = frozen["instances"][0]["state"]["history"].clone()
    original.history.add_(10)
    torch.testing.assert_close(frozen["instances"][0]["state"]["history"], saved_history)

    resumed = MethodSession(history_method)
    resumed.load_state_dict(frozen, device="cpu")
    frozen["instances"][0]["state"]["history"].fill_(-1)
    with call(resumed) as (restored, _):
        torch.testing.assert_close(restored.history, saved_history)


def test_unused_stateful_session_restores_pending_rng(history_method):
    payload = MethodSession(history_method).state_dict()
    assert payload["instances"] == []
    generator = torch.Generator().manual_seed(82)
    payload["pending_state"]["rng"] = generator.get_state()
    resumed = MethodSession(history_method)
    resumed.load_state_dict(payload, device="cpu")
    with call(resumed) as (method, context):
        assert context["call_index"] == 1
        torch.testing.assert_close(method.advance(), torch.rand((), generator=generator), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["none", "classic", "sum"])
def test_stateless_calls_and_checkpoint_do_not_query_cuda_or_method_state(name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stateless session introduced a CUDA/history operation")
    for attr in ("current_stream", "is_current_stream_capturing", "synchronize", "stream"):
        monkeypatch.setattr(torch.cuda, attr, forbidden)
    monkeypatch.setattr(AbftMethod, "state_dict", forbidden)
    monkeypatch.setattr(AbftMethod, "load_state_dict", forbidden)
    session = MethodSession(name)
    with call(session, device="cuda:0") as (method, _):
        assert (method is None) == (name == "none")
    payload = session.state_dict()
    assert payload["instances"] == [] and payload["pending_state"] is None
    restored = MethodSession(name)
    restored.load_state_dict(payload, device="cuda:0")
    with call(restored, device="cuda:0") as (_, context):
        assert context["call_index"] == 2


@pytest.mark.parametrize("name", ["classic", "sum"])
def test_stateless_method_reused_across_profiles_without_scope_registry(name, monkeypatch):
    import abft.session as session_module

    session = MethodSession(name)
    with call(session) as (first, _):
        assert first is not None

    def forbidden(*args, **kwargs):
        raise AssertionError("Stateless hot path performed a factory or scope lookup")

    monkeypatch.setattr(session_module, "get_method", forbidden)
    monkeypatch.setattr(session_module, "_scope_key", forbidden)
    for profile in ({"shape": (5, 7, 9)}, {"role": "dgrad"}, {"layer": "other"},
                    {"precision": "bf16"}, {"device": "cuda:0"}):
        with call(session, **profile) as (same, context):
            assert same is first
            assert "scale_profile" in context
    assert session._entries == {}
    snapshot = session.state_dict()
    session.load_state_dict(snapshot)
    with call(session) as (same, context):
        assert same is first
        assert context["call_index"] == snapshot["call_index"] + 1


def test_stateful_session_rejects_reentry_and_concurrent_snapshot(history_method):
    session = MethodSession(history_method)
    with call(session):
        with pytest.raises(RuntimeError, match="concurrent"):
            with call(session):
                pass
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(RuntimeError, match="concurrent"):
                executor.submit(session.state_dict).result(timeout=5)
    # Failed reservations neither consume an index nor leave the lease locked.
    with call(session) as (_, context):
        assert context["call_index"] == 2


def test_stateful_cuda_stream_changes_and_capture_rejected(history_method, monkeypatch):
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    session = MethodSession(history_method)
    with call(session, device="cuda:0"):
        pass
    stream = SimpleNamespace(cuda_stream=9)
    with pytest.raises(RuntimeError, match="CUDA streams"):
        with call(session, device="cuda:0"):
            pass
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Graph"):
        with call(session, device="cuda:0"):
            pass
    assert session.call_index == 1


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True),
    ("config_version", 2), ("config_version", True),
    ("config_signature", "different-config"),
    ("call_index", -1), ("call_index", True),
    ("instances", {}),
])
def test_malformed_checkpoint_rejected_without_changing_live_state(history_method, field, value):
    session = MethodSession(history_method)
    original, _, _ = populate(session)
    payload = session.state_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        session.load_state_dict(payload, device="cpu")
    assert session.call_index == 1
    with call(session) as (same, _):
        assert same is original and same.count == 1


@pytest.mark.parametrize("field,value", [
    ("name", "different-method"), ("class", "different.Class"),
    ("stateful", False), ("state_schema_version", 2),
])
def test_method_identity_and_state_version_mismatch_rejected(history_method, field, value):
    session = MethodSession(history_method)
    payload = session.state_dict()
    payload["method_metadata"][field] = value
    with pytest.raises(ValueError, match="does not match"):
        session.load_state_dict(payload)


def test_config_and_fault_changes_rejected_on_restore(history_method):
    original = MethodSession({"method": history_method, "parameters": {"window": 8}})
    checkpoint = original.state_dict()
    for other in (
        MethodSession({"method": history_method, "parameters": {"window": 16}}),
        MethodSession({"method": history_method, "parameters": {"window": 8}, "rtol": 0.1}),
        MethodSession({"method": history_method, "parameters": {"window": 8}},
                      {"location": "accumulator", "call_index": 3}),
    ):
        with pytest.raises(ValueError, match="does not match"):
            other.load_state_dict(checkpoint)


def test_duplicate_scope_and_device_mismatch_rejected(history_method):
    session = MethodSession(history_method)
    populate(session)
    payload = session.state_dict()
    payload["instances"].append(deepcopy(payload["instances"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        MethodSession(history_method).load_state_dict(payload, device="cpu")
    payload["instances"].pop()
    with pytest.raises(ValueError, match="device"):
        MethodSession(history_method).load_state_dict(payload, device="cuda:0")


@pytest.mark.parametrize("field,value", [
    ("role", "replay"), ("precision", "fp16"), ("m", 0), ("n", True),
    ("k", -1), ("device", "cuda:00"),
    ("scale_profile", {"scaling": "tensorwise", "global_scale": "amax",
                       "input_dtypes": ["torch.float32", "torch.float32"]}),
])
def test_malformed_scope_rejected(history_method, field, value):
    session = MethodSession(history_method)
    populate(session)
    payload = session.state_dict()
    payload["instances"][0]["scope"][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        MethodSession(history_method).load_state_dict(payload)


@pytest.mark.parametrize("invalid", [object(), torch.Generator(), {1: "non-string-key"}])
def test_custom_objects_and_generators_must_not_enter_checkpoint(history_method, monkeypatch, invalid):
    session = MethodSession(history_method)
    method, _, _ = populate(session)
    monkeypatch.setattr(method, "state_dict", lambda: {"bad": invalid})
    with pytest.raises(ValueError, match="ABFT state"):
        session.state_dict()


@pytest.mark.parametrize("used", [False, True])
def test_method_state_dict_must_be_dictionary(history_method, monkeypatch, used):
    session = MethodSession(history_method)
    if used:
        populate(session)
    monkeypatch.setattr(SampleHistory, "state_dict", lambda self: [1, 2])
    with pytest.raises(ValueError, match="dictionary"):
        session.state_dict()


def test_cyclic_state_rejected(history_method, monkeypatch):
    session = MethodSession(history_method)
    method, _, _ = populate(session)
    cycle = []
    cycle.append(cycle)
    monkeypatch.setattr(method, "state_dict", lambda: {"cycle": cycle})
    with pytest.raises(ValueError, match="cycles"):
        session.state_dict()
