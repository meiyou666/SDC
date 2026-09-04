from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import torch
from torch import nn

from .core import (
    CANDIDATE_PROJECTIONS,
    CRITICAL_PROJECTIONS,
    InjectionRecord,
    ProjectionKey,
    ProtectionMode,
    RunState,
    apply_protection,
    flip_fp16_bit,
    fp16_bits_at,
)


@dataclass(frozen=True)
class ProjectionRef:
    layer_index: int
    projection: str
    module: nn.Module


class HookSession:
    def __init__(self, handles: Iterable[Any]) -> None:
        self._handles = list(handles)
        self._removed = False

    def remove(self) -> None:
        if self._removed:
            return
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._removed = True

    def __enter__(self) -> "HookSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.remove()


def _resolve_layers(model: nn.Module) -> nn.ModuleList:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError(
            "Expected Qwen2 structure model.model.layers or model.layers"
        )
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Qwen2 layers must be an nn.ModuleList")
    if len(layers) == 0:
        raise ValueError("The model contains no decoder layers")
    return layers


def iter_qwen2_projections(model: nn.Module) -> Tuple[ProjectionRef, ...]:
    references: List[ProjectionRef] = []

    for layer_index, layer in enumerate(_resolve_layers(model)):
        self_attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if self_attn is None or mlp is None:
            raise TypeError(f"Layer {layer_index} lacks self_attn or mlp")

        owners = {
            "q_proj": self_attn,
            "k_proj": self_attn,
            "v_proj": self_attn,
            "o_proj": self_attn,
            "gate_proj": mlp,
            "up_proj": mlp,
            "down_proj": mlp,
        }

        for projection in CANDIDATE_PROJECTIONS:
            module = getattr(owners[projection], projection, None)
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"Layer {layer_index} has no module {projection}"
                )
            references.append(
                ProjectionRef(layer_index, projection, module)
            )

    return tuple(references)


def _make_composite_hook(
    state: RunState,
    layer_index: int,
    projection: str,
):
    key: ProjectionKey = (layer_index, projection)

    def hook(
        module: nn.Module,
        inputs: Tuple[object, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Projection {key} returned {type(output)!r}, not Tensor"
            )
        if state.current_step < 0:
            raise RuntimeError(
                "Projection hook ran before the root forward-pre-hook"
            )

        result = output
        if state.current_step == 0:
            if state.mode is ProtectionMode.PAPER_CLAMP:
                result = torch.where(
                    torch.isnan(result),
                    torch.zeros_like(result),
                    result,
                )
            if projection in CRITICAL_PROJECTIONS:
                state.calibrate(key, result)
            return result

        fault = state.fault_for_current_event(
            layer_index,
            projection,
        )
        if fault is not None:
            if result.ndim != 3 or result.shape[0] != 1 or result.shape[1] != 1:
                raise RuntimeError(
                    "Fault injection requires one generated-token activation "
                    f"with shape [1, 1, width], got {tuple(result.shape)}"
                )
            out_features = getattr(module, "out_features", None)
            if out_features is not None and result.shape[-1] != int(out_features):
                raise RuntimeError(
                    f"Projection width {result.shape[-1]} does not match "
                    f"module.out_features={out_features}"
                )
            if fault.flat_index >= result.shape[-1]:
                raise IndexError(
                    f"Fault feature {fault.flat_index} is outside projection "
                    f"width {result.shape[-1]}"
                )
            before_bits = fp16_bits_at(result, fault.flat_index)
            result = flip_fp16_bit(
                result,
                flat_index=fault.flat_index,
                bit_index=fault.bit_index,
            )
            after_bits = fp16_bits_at(result, fault.flat_index)

        if projection in CRITICAL_PROJECTIONS:
            result = apply_protection(
                result,
                state.mode,
                state.bounds_for(key),
            )

        if fault is not None:
            record = InjectionRecord(
                spec_id=fault.stable_hash,
                observed_step=state.current_step,
                layer_index=layer_index,
                projection=projection,
                flat_index=fault.flat_index,
                bit_index=fault.bit_index,
                output_shape=tuple(int(size) for size in output.shape),
                before_bits=before_bits,
                after_bits=after_bits,
                protected_bits=fp16_bits_at(result, fault.flat_index),
            )
            state.record_injection(fault, record)

        return result

    return hook


def install_qwen2_hooks(
    model: nn.Module,
    state: RunState,
) -> HookSession:
    if model.training:
        raise RuntimeError("Call model.eval() before installing FT2 hooks")

    references = iter_qwen2_projections(model)
    available = {
        (ref.layer_index, ref.projection)
        for ref in references
    }
    expected_critical = {
        key
        for key in available
        if key[1] in CRITICAL_PROJECTIONS
    }

    for fault in state.faults:
        target = (fault.layer_index, fault.projection)
        if target not in available:
            raise ValueError(
                f"Fault targets a projection absent from the model: {target}"
            )

    state.bind(expected_critical)

    handles: List[Any] = []

    try:
        def root_forward_pre_hook(
            module: nn.Module,
            inputs: Tuple[object, ...],
        ) -> None:
            state.begin_forward()

        handles.append(
            model.register_forward_pre_hook(root_forward_pre_hook)
        )

        for ref in references:
            handles.append(
                ref.module.register_forward_hook(
                    _make_composite_hook(
                        state,
                        ref.layer_index,
                        ref.projection,
                    )
                )
            )
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        raise

    return HookSession(handles)
