"""Connect ordinary nn.Linear forward/backward to official CUTLASS GEMM."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from . import backend


def packed(matrix):
    # Preserve a packed matrix or its simple transpose. Materialize only other
    # strides, such as the expanded gradient produced by output.sum().backward().
    if matrix.is_contiguous() or matrix.T.is_contiguous():
        return matrix
    return matrix.contiguous()


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, dtype):
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = packed(x.reshape(-1, x.shape[-1]).to(dtype))
            w = packed(weight.to(dtype))
            ctx.save_for_backward(a, w)
            ctx.input_shape = x.shape
            ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
            ctx.bias_dtype = bias.dtype if bias is not None else None
            out = backend.mm(a, w.T)
            if bias is not None:
                out = out + bias.to(dtype)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        a, w = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            dy = packed(grad_output.reshape(-1, grad_output.shape[-1]).to(a.dtype))
            dx = dw = db = None
            if ctx.needs_input_grad[0]:
                dx = backend.mm(dy, w).reshape(ctx.input_shape).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                dw = backend.mm(dy.T, a).to(ctx.weight_dtype)
            if ctx.bias_dtype is not None and ctx.needs_input_grad[2]:
                db = dy.sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None


class CutlassLinear(nn.Linear):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.train(original.training)

    def forward(self, x):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("当前基础训练使用 BF16 或 FP32。")
        return _Linear.apply(x, self.weight, self.bias, dtype)


def install(model):
    """Replace all existing Linear modules, preserving parameter identity."""
    if isinstance(model, nn.Linear) and not isinstance(model, CutlassLinear):
        return CutlassLinear(model)
    layers = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
              if isinstance(module, nn.Linear) and not isinstance(module, CutlassLinear)]
    if not layers:
        raise ValueError("模型中没有可替换的 nn.Linear。")
    replacements = {}
    for name, module in layers:
        if id(module) not in replacements:
            replacements[id(module)] = CutlassLinear(module)
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute, replacements[id(module)])
    return model
