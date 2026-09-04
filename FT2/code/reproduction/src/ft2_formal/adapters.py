from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ProjectionSite:
    key: str
    layer_index: int
    projection: str
    module_path: str
    module: nn.Module
    sampling_weight: int
    critical: bool
    out_features: int


@dataclass(frozen=True)
class ModelAdapter:
    model_key: str
    sites: Tuple[ProjectionSite, ...]
    num_layers: int
    attention_implementation: str

    @property
    def critical_sites(self) -> Tuple[ProjectionSite, ...]:
        return tuple(site for site in self.sites if site.critical)

    @property
    def sites_by_key(self) -> Mapping[str, ProjectionSite]:
        return {site.key: site for site in self.sites}

    @property
    def projection_weights(self) -> Mapping[str, int]:
        result: dict[str, int] = {}
        for site in self.sites:
            previous = result.setdefault(site.projection, site.sampling_weight)
            if previous != site.sampling_weight:
                raise RuntimeError("Projection weight changed across layers")
        return result


def _linear_site(
    *,
    layer_index: int,
    projection: str,
    path: str,
    module: nn.Module,
    weight: int,
    critical: bool,
    require_fp16: bool,
) -> ProjectionSite:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"{path} is not nn.Linear: {type(module)!r}")
    if require_fp16 and module.weight.dtype != torch.float16:
        raise TypeError(f"{path} weight is {module.weight.dtype}, expected float16")
    return ProjectionSite(
        key=f"layer.{layer_index}.{projection}",
        layer_index=layer_index,
        projection=projection,
        module_path=path,
        module=module,
        sampling_weight=weight,
        critical=critical,
        out_features=int(module.out_features),
    )


def _discover_opt(model: nn.Module, require_fp16: bool) -> ModelAdapter:
    try:
        layers = model.model.decoder.layers
    except AttributeError as exc:
        raise TypeError("Expected OPTForCausalLM model.model.decoder.layers") from exc
    if len(layers) != 32:
        raise ValueError(f"OPT-2.7B must have 32 layers, got {len(layers)}")
    definitions = (
        ("v_proj", "self_attn.v_proj", 1, True),
        ("k_proj", "self_attn.k_proj", 1, False),
        ("q_proj", "self_attn.q_proj", 1, False),
        ("out_proj", "self_attn.out_proj", 1, True),
        ("fc1", "fc1", 4, False),
        ("fc2", "fc2", 4, True),
    )
    sites = []
    for layer_index, layer in enumerate(layers):
        for projection, relative, weight, critical in definitions:
            owner = layer
            for component in relative.split("."):
                owner = getattr(owner, component)
            sites.append(
                _linear_site(
                    layer_index=layer_index,
                    projection=projection,
                    path=f"model.decoder.layers.{layer_index}.{relative}",
                    module=owner,
                    weight=weight,
                    critical=critical,
                    require_fp16=require_fp16,
                )
            )
    adapter = ModelAdapter(
        model_key="opt_2_7b",
        sites=tuple(sites),
        num_layers=32,
        attention_implementation="eager",
    )
    if len(adapter.sites) != 192 or len(adapter.critical_sites) != 96:
        raise AssertionError("Unexpected OPT site cardinality")
    return adapter


def _discover_qwen(model: nn.Module, require_fp16: bool) -> ModelAdapter:
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise TypeError("Expected Qwen2ForCausalLM model.model.layers") from exc
    if len(layers) != 28:
        raise ValueError(f"Qwen2-Math-7B must have 28 layers, got {len(layers)}")
    definitions = (
        ("v_proj", "self_attn.v_proj", 1, True),
        ("k_proj", "self_attn.k_proj", 1, False),
        ("q_proj", "self_attn.q_proj", 7, False),
        ("o_proj", "self_attn.o_proj", 7, True),
        ("up_proj", "mlp.up_proj", 37, True),
        ("gate_proj", "mlp.gate_proj", 37, False),
        ("down_proj", "mlp.down_proj", 37, True),
    )
    sites = []
    for layer_index, layer in enumerate(layers):
        for projection, relative, weight, critical in definitions:
            owner = layer
            for component in relative.split("."):
                owner = getattr(owner, component)
            sites.append(
                _linear_site(
                    layer_index=layer_index,
                    projection=projection,
                    path=f"model.layers.{layer_index}.{relative}",
                    module=owner,
                    weight=weight,
                    critical=critical,
                    require_fp16=require_fp16,
                )
            )
    adapter = ModelAdapter(
        model_key="qwen2_math_7b",
        sites=tuple(sites),
        num_layers=28,
        attention_implementation="sdpa",
    )
    if len(adapter.sites) != 196 or len(adapter.critical_sites) != 112:
        raise AssertionError("Unexpected Qwen2 site cardinality")
    return adapter


def discover_adapter(
    model: nn.Module,
    model_key: str,
    *,
    require_fp16: bool = True,
) -> ModelAdapter:
    if model_key == "opt_2_7b":
        adapter = _discover_opt(model, require_fp16)
    elif model_key == "qwen2_math_7b":
        adapter = _discover_qwen(model, require_fp16)
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")
    configured = str(getattr(model.config, "_attn_implementation", ""))
    if configured and configured != adapter.attention_implementation:
        raise ValueError(
            f"{model_key} attention backend is {configured!r}, "
            f"expected {adapter.attention_implementation!r}"
        )
    module_ids = [id(site.module) for site in adapter.sites]
    if len(module_ids) != len(set(module_ids)):
        raise AssertionError("The same Linear module was discovered more than once")
    return adapter
