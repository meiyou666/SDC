"""Connect ordinary nn.Linear forward/backward to official CUTLASS GEMM."""

from collections import deque
import json
from pathlib import Path

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from abft.config import parse_abft_config, parse_fault_config

from . import backend


class _Protection:
    """Shared bounded reports and a model-wide physical GEMM call counter.

    Forward includes checkpoint replay: custom autograd disables grad mode even
    on the original forward, so grad mode cannot identify recomputation.
    JSONL recording is opt-in and explicitly synchronizes report tensors.
    """

    def __init__(self, abft, fault, report_max_records=0, report_path=None):
        self.abft = parse_abft_config(abft)
        self.fault = parse_fault_config(fault)
        self.reports = deque(maxlen=report_max_records)
        self.report_path = Path(report_path) if report_path else None
        self.call_index = 0

    def record(self, report):
        if report is None:
            return
        if self.reports.maxlen:
            self.reports.append(report)
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with self.report_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report.to_record()) + "\n")

    def product(self, a, b, precision, scaling, global_scale, layer, role):
        import abft

        self.call_index += 1
        context = {"layer": layer, "role": role, "call_index": self.call_index,
                   "execution_kind": "forward_or_recompute" if role == "forward" else "backward",
                   "retry": 0}
        try:
            output, report = abft.execute(a, b, precision, scaling, global_scale,
                                          self.abft, self.fault, context)
        except Exception as exc:
            self.record(getattr(exc, "report", None))
            raise
        self.record(report)
        return output


def packed(matrix):
    # Preserve a packed matrix or its simple transpose. Materialize only other
    # strides, such as the expanded gradient produced by output.sum().backward().
    if matrix.is_contiguous() or matrix.T.is_contiguous():
        return matrix
    return matrix.contiguous()


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, dtype, precision, scaling, global_scale, protection=None, layer=""):
        with torch.autocast(device_type=x.device.type, enabled=False):
            low_precision = precision in {"fp8", "nvfp4"}
            a = packed(x.reshape(-1, x.shape[-1]) if low_precision else x.reshape(-1, x.shape[-1]).to(dtype))
            w = packed(weight if low_precision else weight.to(dtype))
            ctx.save_for_backward(a, w)
            ctx.input_shape = x.shape
            ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
            ctx.bias_dtype = bias.dtype if bias is not None else None
            ctx.precision, ctx.scaling, ctx.global_scale = precision, scaling, global_scale
            ctx.protection, ctx.layer = protection, layer
            out = (backend.gemm(a, w.T, precision, scaling, global_scale) if protection is None else
                   protection.product(a, w.T, precision, scaling, global_scale, layer, "forward")).to(dtype)
            if bias is not None:
                out = out + bias.to(dtype)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        a, w = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            dy = packed(grad_output.reshape(-1, grad_output.shape[-1]).to(a.dtype))
            def product(left, right, role):
                if ctx.protection is not None:
                    return ctx.protection.product(left, right, ctx.precision, ctx.scaling,
                                                  ctx.global_scale, ctx.layer, role)
                return backend.gemm(left, right, ctx.precision, ctx.scaling, ctx.global_scale)

            dx = dw = db = None
            if ctx.needs_input_grad[0]:
                dx = product(dy, w, "dgrad").reshape(ctx.input_shape).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                dw = product(dy.T, a, "wgrad").to(ctx.weight_dtype)
            if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
                db = dy.sum(0).to(ctx.bias_dtype)
        return (dx, dw, db, None, None, None, None, None, None)[:len(ctx.needs_input_grad)]


class CutlassLinear(nn.Linear):
    def __init__(self, original, precision=None, scaling="none", global_scale="amax",
                 protection=None, layer=""):
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.precision, self.scaling, self.global_scale = precision, scaling, global_scale
        self.protection, self.layer_name = protection, layer
        self.train(original.training)

    def forward(self, x):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        if self.precision == "fp32":
            dtype = torch.float32
        elif self.precision in {"bf16", "fp8", "nvfp4"}:
            dtype = torch.bfloat16
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("当前基础训练使用 BF16 或 FP32。")
        precision = self.precision or ("bf16" if dtype == torch.bfloat16 else "fp32")
        args = (x, self.weight, self.bias, dtype, precision, self.scaling, self.global_scale)
        return _Linear.apply(*args) if self.protection is None else _Linear.apply(
            *args, self.protection, self.layer_name)


def install(model, precision=None, scaling="none", global_scale="amax", *,
            abft_config=None, fault_config=None, report_max_records=0, report_path=None):
    """Replace all existing Linear modules, preserving parameter identity."""
    abft_config, fault_config = parse_abft_config(abft_config), parse_fault_config(fault_config)
    protection = None
    if abft_config.method != "none" or fault_config.location != "none":
        protection = _Protection(abft_config, fault_config, report_max_records, report_path)
    if isinstance(model, nn.Linear) and not isinstance(model, CutlassLinear):
        return CutlassLinear(model, precision, scaling, global_scale, protection)
    layers = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
              if isinstance(module, nn.Linear) and not isinstance(module, CutlassLinear)]
    if not layers:
        raise ValueError("模型中没有可替换的 nn.Linear。")
    replacements = {}
    for name, module in layers:
        if id(module) not in replacements:
            replacements[id(module)] = CutlassLinear(module, precision, scaling, global_scale, protection, name)
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute, replacements[id(module)])
    return model
