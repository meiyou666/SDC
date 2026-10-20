"""Lightweight experiment configuration; importing this module never loads Torch."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import math


@dataclass(frozen=True)
class AbftConfig:
    method: str = "none"
    atol: float = 1e-4
    rtol: float = 1e-4
    recovery: str = "none"
    on_error: str = "raise"
    reference_block_k: int = 256
    diagnostics: bool = False
    timing: bool = False
    reference_mode: str = "separate"
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        def validate_parameters(value):
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise ValueError("ABFT parameters require string dictionary keys")
                for child in value.values():
                    validate_parameters(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    validate_parameters(child)
        if not isinstance(self.parameters, dict):
            raise ValueError("ABFT parameters must be a string-key dictionary")
        try:
            json.dumps(self.parameters, allow_nan=False)
            validate_parameters(self.parameters)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("ABFT parameters must contain finite JSON-serializable values and string keys") from error
        if self.reference_mode not in {"separate", "input_fused"}:
            raise ValueError("reference_mode must be separate or input_fused")
        if self.reference_mode == "input_fused" and self.method == "none":
            raise ValueError("input_fused requires an ABFT method")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("ABFT method must be a nonempty registered name")
        for value in (self.atol, self.rtol):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("ABFT tolerances must be finite and nonnegative")
        if self.recovery not in {"none", "recompute"} or self.on_error not in {"raise", "report"}:
            raise ValueError("Invalid ABFT recovery/on_error policy")
        if type(self.reference_block_k) is not int or self.reference_block_k <= 0 or self.reference_block_k % 16:
            raise ValueError("reference_block_k must be a positive multiple of 16")
        if type(self.diagnostics) is not bool:
            raise ValueError("diagnostics must be boolean")
        if type(self.timing) is not bool:
            raise ValueError("timing must be boolean")


@dataclass(frozen=True)
class FaultConfig:
    location: str = "none"
    row: int = 0
    col: int = 0
    bit: int = 0
    call_index: int = 1
    layer: str | None = None
    role: str | None = None

    def __post_init__(self):
        if self.location not in {"none", "a", "b", "accumulator"}:
            raise ValueError("Invalid fault location")
        for name in ("row", "col", "bit", "call_index"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (1 if name == "call_index" else 0):
                raise ValueError(f"Invalid fault {name}")
        for name in ("layer", "role"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"fault {name} must be a string or None")
        if self.role is not None and self.role not in {"forward", "dgrad", "wgrad"}:
            raise ValueError("fault role must be forward, dgrad, wgrad, or None")


def _parse(value, cls):
    if isinstance(value, cls):
        return value
    if value is None or value == "none":
        return cls()
    if isinstance(value, str) and cls is AbftConfig:
        return cls(method=value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**value)


def parse_abft_config(value=None):
    return _parse(value, AbftConfig)


def parse_fault_config(value=None):
    return _parse(value, FaultConfig)
