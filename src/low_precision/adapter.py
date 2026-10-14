"""Bridge ordinary linear layers to Transformer Engine training recipes."""

from importlib.metadata import version
import math

import torch
from torch import nn


TE_VERSION = "2.21.0.dev0"
TE_SOURCE_COMMIT = "fb98b935d6054a2fc68137951051695de3acaee3"


def _validate_precision(precision):
    if precision not in {"fp8", "fp4"}:
        raise ValueError("precision 必须为 fp8 或 fp4。")


def metadata(precision):
    """Describe the fixed recipes without importing the CUDA extension."""
    _validate_precision(precision)
    return {
        "precision": precision,
        "backend": "transformer_engine",
        "transformer_engine_version": TE_VERSION,
        "transformer_engine_source_commit": TE_SOURCE_COMMIT,
        "target_compute_capability": [12, 0],
        "recipe": "Float8CurrentScaling(HYBRID)" if precision == "fp8" else "NVFP4BlockScaling",
        "forward_operands": "E4M3/E4M3" if precision == "fp8" else "E2M1/E2M1",
        "dgrad_operands": "E5M2/E4M3" if precision == "fp8" else "E2M1/E2M1",
        "wgrad_operands": "E5M2/E4M3" if precision == "fp8" else "E2M1/E2M1",
        "accumulator": "float32",
        "master_parameters": "float32",
        "activation_dtype": "bfloat16",
        "gemm_outputs": {"forward": "bfloat16", "dgrad": "bfloat16", "wgrad": "bfloat16"},
        "fp8_split_accumulator": (
            {"forward": False, "dgrad": True, "wgrad": True} if precision == "fp8" else None
        ),
        "scaling": "dynamic_tensorwise" if precision == "fp8" else "nvfp4_block16",
        "scale_dtype": "float32" if precision == "fp8" else "E4M3_blocks_and_float32_global",
        "nvfp4_rht": precision == "fp4",
        "nvfp4_stochastic_rounding": precision == "fp4",
        "nvfp4_weight_2d_quantization": precision == "fp4",
    }


def _load_engine(precision):
    """Import only when installing CUDA layers; CPU tooling needs no TE."""
    if not torch.cuda.is_available():
        raise RuntimeError("Transformer Engine 低精度训练需要 CUDA GPU。")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(f"当前低精度镜像为 SM120（RTX 5090）构建，当前计算能力为 {capability}。")
    installed = version("transformer-engine")
    if installed != TE_VERSION:
        raise RuntimeError(f"需要 transformer-engine=={TE_VERSION}，当前为 {installed}。")
    from transformer_engine.common.recipe import Float8CurrentScaling, Format, NVFP4BlockScaling
    import transformer_engine.pytorch as te

    if precision == "fp8":
        recipe = Float8CurrentScaling(fp8_format=Format.HYBRID, backward_override=None)
        if recipe.fp8_quant_fwd_inp.power_2_scale:
            raise ValueError("本配置要求 Float8CurrentScaling 默认非二次幂缩放。")
    else:
        recipe = NVFP4BlockScaling(
            disable_rht=False, disable_stochastic_rounding=False, disable_2d_quantization=False,
            row_scaled_activation=False, nvfp4_4over6="none", backward_override=None,
        )
    return te, recipe


def install(model, precision):
    """Replace linear modules, preserving parameters and module aliases."""
    _validate_precision(precision)
    layers = [(name, layer) for name, layer in model.named_modules(remove_duplicate=False)
              if isinstance(layer, nn.Linear)]
    if not layers:
        raise ValueError("模型中没有可替换的 nn.Linear。")
    output_alignment = 8 if precision == "fp8" else 32
    input_alignment = 16 if precision == "fp8" else 32
    for name, layer in layers:
        if type(layer) is not nn.Linear:
            raise ValueError(f"{name or '<root>'}: 仅支持原生 nn.Linear，不能覆盖已有自定义算子。")
        if layer.in_features % input_alignment or layer.out_features % output_alignment:
            raise ValueError(f"{name or '<root>'}: {precision} 不支持线性层形状 "
                             f"({layer.out_features}, {layer.in_features})。")
        if layer.weight.dtype != torch.float32 or (layer.bias is not None and layer.bias.dtype != torch.float32):
            raise ValueError("低精度训练组件要求 FP32 master parameters。")
    te, recipe = _load_engine(precision)

    class RecipeLinear(te.Linear):
        def forward(self, inp):
            tokens = math.prod(inp.shape[:-1])
            if inp.ndim < 2 or inp.shape[-1] != self.in_features or tokens % output_alignment:
                raise ValueError(f"{precision} 不支持输入形状 {tuple(inp.shape)}；"
                                 f"展开后的 token 数须为 {output_alignment} 的倍数。")
            # TE quantizes FP4 inputs before AMP casts; its RHT requires BF16.
            # This differentiable cast keeps the original FP32 input gradient.
            if precision == "fp4":
                inp = inp.to(torch.bfloat16)
            with te.autocast(enabled=True, recipe=recipe):
                return super().forward(inp)

    replacements = {}
    for _, layer in layers:
        if id(layer) in replacements:
            continue
        replacement = RecipeLinear(
            layer.in_features, layer.out_features, bias=layer.bias is not None,
            params_dtype=layer.weight.dtype, device="meta", fuse_wgrad_accumulation=False,
        )
        replacement.weight = layer.weight
        replacement.bias = layer.bias
        replacement.train(layer.training)
        replacements[id(layer)] = replacement
    if isinstance(model, nn.Linear):
        return replacements[id(model)]
    for name, layer in layers:
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute, replacements[id(layer)])
    return model
