"""Algorithm registration and shared settings."""

from dataclasses import dataclass
import math


_FACTORIES = {}


def register(name, backend, factory):
    if name in {"none", "classic"} or (name, backend) in _FACTORIES:
        raise ValueError(f"算法已注册：{name}/{backend}")
    _FACTORIES[name, backend] = factory


def detector_for(method, backend):
    if method.name in {"none", "classic"}:
        return None
    try:
        return _FACTORIES[method.name, backend](method)
    except KeyError as exc:
        raise NotImplementedError(f"{method.name} 缺少 {backend} 实现。") from exc


@dataclass(frozen=True)
class Algorithm:
    name: str = "classic"
    atol: float = 1e-3
    rtol: float = 1e-3

    def __post_init__(self):
        if self.name not in {"none", "classic"} and not any(name == self.name for name, _ in _FACTORIES):
            raise NotImplementedError(f"{self.name} 尚未注册实现，请按 algorithms/README.md 接入。")
        if any(not math.isfinite(v) or v < 0 for v in (self.atol, self.rtol)):
            raise ValueError("atol 和 rtol 必须是有限非负数。")


def resolve_algorithm(value: str | Algorithm) -> Algorithm:
    return value if isinstance(value, Algorithm) else Algorithm(name=value)
