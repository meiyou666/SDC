"""Ascend registration point; CUDA kernels are never dispatched to an NPU."""


def execute(*args, **kwargs):
    raise NotImplementedError("昇腾 GEMM 后端尚未实现。请实现同一接口的 Ascend C 内核。")
