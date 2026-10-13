"""Shared GEMM entry point for training and paired ABFT experiments."""

from .api import gemm
from .types import PROFILES, ExecutionContext, Fault, GemmResult


__all__ = ["gemm", "ExecutionContext", "Fault", "GemmResult", "PROFILES"]
