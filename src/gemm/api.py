"""One contract for standalone experiments and trainable linear layers."""

import copy

import torch

from algorithms import detector_for, resolve_algorithm

from .quantization import hadamard_pair
from .types import PHASES, PROFILES, ExecutionContext


class RecoveryError(RuntimeError):
    pass


@torch.no_grad()
def gemm(a, b, *, profile="bf16", algorithm="classic", backend="nvidia", phase="forward",
         context=None, fault=None, recovery="none", detector=None):
    """Compute FP32 C=A@B. Use ProtectedLinear for autograd.

    ``recovery`` is none, native (syndrome correction), or recompute.
    Recompute synchronizes to decide whether a second protected call is needed.
    """
    if profile not in PROFILES or phase not in PHASES:
        raise ValueError("未知精度配置或 GEMM 阶段。")
    if recovery not in {"none", "native", "recompute"}:
        raise ValueError("recovery 必须为 none、native 或 recompute。")
    if not all(isinstance(x, torch.Tensor) and x.ndim == 2 and x.is_floating_point() for x in (a, b)):
        raise ValueError("GEMM 输入必须为二维浮点 Tensor。")
    if a.device != b.device or a.shape[1] != b.shape[0] or min(*a.shape, *b.shape) == 0:
        raise ValueError("GEMM 形状或设备不匹配。")
    method = resolve_algorithm(algorithm)
    detector = detector if detector is not None else detector_for(method, backend)
    if method.name == "none" and recovery != "none":
        raise ValueError("无保护 GEMM 不设置恢复策略。")
    context = context or ExecutionContext()
    if backend == "nvidia":
        from backends.nvidia.runtime import execute
    elif backend == "reference":
        from backends.reference import execute
    elif backend == "ascend":
        raise NotImplementedError("昇腾后端保留同一接口；本分支提供 NVIDIA CUTLASS 实现。")
    else:
        raise ValueError(f"未知后端：{backend}")
    # Disable outer Trainer autocast; this contract selects every GEMM precision.
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b = a.float(), b.float()
        if profile == "nvfp4" and phase == "wgrad":
            a, b = hadamard_pair(a, b, context.quant_seed(phase) ^ 0x48A2)
        if fault:
            fault.validate(a.shape[0], b.shape[1], a.shape[1], PROFILES[profile])
        params = dict(profile=profile, method=method, phase=phase, context=context, detector=detector)
        state = copy.deepcopy(detector.state_dict()) if detector is not None and recovery == "recompute" else None
        result = execute(a, b, **params, fault=fault, correct=recovery == "native")
        if recovery == "recompute" and bool(result.detected):
            if state is not None:
                detector.load_state_dict(state)
            retry = execute(a, b, **params, fault=None, correct=False)
            if bool(retry.detected) or not bool(torch.isfinite(retry.output).all()):
                raise RecoveryError("同源重算后校验仍失败，已停止本次训练。")
            result.output = retry.output
            result.status[5] = 0
            result.recomputed = True
    return result
