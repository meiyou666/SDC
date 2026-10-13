"""Autograd adapter that reuses the model's original Parameter objects."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from algorithms import Algorithm, detector_for

from .api import RecoveryError, gemm
from .types import ExecutionContext


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, layer, token, output_dtype):
        flat = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(x, weight)
        ctx.layer, ctx.token, ctx.has_bias = layer, token, bias is not None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        out = layer.multiply(flat, weight.T, "forward", token)
        if bias is not None:
            out = out + bias.float()
        return out.reshape(*x.shape[:-1], weight.shape[0]).to(output_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        flat = x.reshape(-1, x.shape[-1])
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = ctx.layer.multiply(dy, weight, "dgrad", ctx.token).reshape_as(x).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = ctx.layer.multiply(dy.T, flat, "wgrad", ctx.token).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = dy.float().sum(0).to(ctx.bias_dtype)
        return dx, dw, db, None, None, None


class ProtectedLinear(nn.Module):
    def __init__(self, original, *, name, profile="bf16", algorithm="classic", backend="nvidia",
                 atol=1e-3, rtol=1e-3, recovery="recompute", seed=2026, capture_dir=None):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.name, self.profile, self.backend = name, profile, backend
        self.method = Algorithm(algorithm, atol, rtol)
        self.detector = detector_for(self.method, backend)
        self.recovery = recovery if algorithm != "none" else "none"
        self.seed, self.capture_dir = seed, Path(capture_dir) if capture_dir else None
        self._captured = set()
        self.train(original.training)

    def multiply(self, a, b, phase, token):
        context = ExecutionContext(self.seed, self.name, token)
        if self.capture_dir and phase not in self._captured:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"a": a.detach().float().cpu(), "b": b.detach().float().cpu(),
                        "phase": phase, "context": asdict(context), "profile": self.profile},
                       self.capture_dir / f"{self.name}.{phase}.pt")
            self._captured.add(phase)
        result = gemm(a, b, profile=self.profile, algorithm=self.method, backend=self.backend,
                      phase=phase, context=context, recovery=self.recovery, detector=self.detector)
        if bool(result.unrecovered) or not bool(torch.isfinite(result.output).all()):
            raise RecoveryError(f"{self.name} {phase} 的结果未恢复，停止参数更新。")
        return result.output

    def forward(self, x):
        # Checkpoint recomputation restores Torch's CPU RNG. Saving it in the
        # Trainer checkpoint also preserves quantization seeds on resume.
        token = int(torch.randint(0, 2**31, ()))
        output_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return _Linear.apply(x, self.weight, self.bias, self, token, output_dtype)


def install(model, *, targets=("gate_proj", "up_proj", "down_proj"), **settings):
    """Replace selected linear modules; attention and optimizer stay unchanged."""
    matches = [(name, child) for name, child in model.named_modules()
               if isinstance(child, nn.Linear) and name.rsplit(".", 1)[-1] in targets]
    if not matches:
        raise ValueError("没有找到指定的线性层。")
    for name, child in matches:
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attribute,
                ProtectedLinear(child, name=name, **settings))
    return model
