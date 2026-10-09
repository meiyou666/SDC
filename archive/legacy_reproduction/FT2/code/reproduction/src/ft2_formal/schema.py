from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

import torch


class FaultType(str, Enum):
    FP16_1BIT = "fp16_1bit"
    FP16_2BIT = "fp16_2bit"
    FP16_EXPONENT_BIT = "fp16_exponent_bit"


class BoundsSource(str, Enum):
    NONE = "none"
    FIRST_TOKEN = "first_token"
    OFFLINE = "offline"


class Correction(str, Enum):
    NONE = "none"
    PAPER_CLAMP = "paper_clamp"
    REPOSITORY_ZERO = "repository_zero"


@dataclass(frozen=True)
class Bounds:
    raw_min: float
    raw_max: float
    scaling_factor: float

    def __post_init__(self) -> None:
        values = (self.raw_min, self.raw_max, self.scaling_factor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Bounds values must be finite, got {values}")
        if self.raw_min > self.raw_max:
            raise ValueError("raw_min must not exceed raw_max")
        if self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        if self.lower > self.upper:
            raise ValueError("scaled lower bound must not exceed upper bound")

    @property
    def lower(self) -> float:
        return self.raw_min * self.scaling_factor

    @property
    def upper(self) -> float:
        return self.raw_max * self.scaling_factor

    def to_dict(self) -> dict[str, float]:
        return {
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "scaling_factor": self.scaling_factor,
            "lower": self.lower,
            "upper": self.upper,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Bounds":
        return cls(
            raw_min=float(value["raw_min"]),
            raw_max=float(value["raw_max"]),
            scaling_factor=float(value["scaling_factor"]),
        )


@dataclass(frozen=True)
class ProtectionSpec:
    bounds_source: BoundsSource
    correction: Correction
    scaling_factor: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds_source", BoundsSource(self.bounds_source))
        object.__setattr__(self, "correction", Correction(self.correction))
        if not math.isfinite(self.scaling_factor) or self.scaling_factor <= 0:
            raise ValueError("scaling_factor must be finite and positive")
        if self.bounds_source is BoundsSource.NONE:
            if self.correction is not Correction.NONE:
                raise ValueError("No-protection mode must use correction=none")
        elif self.correction is Correction.NONE:
            raise ValueError("A bounds source requires a correction operator")

    @property
    def mode_id(self) -> str:
        if self.bounds_source is BoundsSource.NONE:
            return "no_protection"
        return f"{self.correction.value}_{self.bounds_source.value}_bounds"


@dataclass(frozen=True)
class FaultSpec:
    model_key: str
    dataset_key: str
    dataset_index: int
    sample_position: int
    trial_index: int
    fault_type: FaultType
    target_step: int
    layer_index: int
    projection: str
    site_key: str
    sequence_index: int
    feature_index: int
    expected_sequence_length: int
    expected_out_features: int
    bit_positions: Tuple[int, ...]
    campaign_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "fault_type", FaultType(self.fault_type))
        object.__setattr__(
            self, "bit_positions", tuple(int(value) for value in self.bit_positions)
        )
        integers = (
            self.dataset_index,
            self.sample_position,
            self.trial_index,
            self.target_step,
            self.layer_index,
            self.sequence_index,
            self.feature_index,
            self.expected_sequence_length,
            self.expected_out_features,
            self.campaign_seed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("FaultSpec integer coordinates must be int values")
        if min(
            self.dataset_index,
            self.sample_position,
            self.trial_index,
            self.target_step,
            self.layer_index,
            self.sequence_index,
            self.feature_index,
        ) < 0:
            raise ValueError("Fault coordinates must be non-negative")
        if self.expected_sequence_length <= 0 or self.expected_out_features <= 0:
            raise ValueError("Expected activation dimensions must be positive")
        if self.sequence_index >= self.expected_sequence_length:
            raise ValueError("sequence_index is outside expected activation")
        if self.feature_index >= self.expected_out_features:
            raise ValueError("feature_index is outside expected activation")
        if not all((self.model_key, self.dataset_key, self.projection, self.site_key)):
            raise ValueError("Fault identity strings must be non-empty")
        if len(set(self.bit_positions)) != len(self.bit_positions):
            raise ValueError("bit_positions must be distinct")
        if any(bit < 0 or bit > 15 for bit in self.bit_positions):
            raise ValueError("FP16 bit positions must be in [0, 15]")
        expected_count = 2 if self.fault_type is FaultType.FP16_2BIT else 1
        if len(self.bit_positions) != expected_count:
            raise ValueError(
                f"{self.fault_type.value} requires {expected_count} bit position(s)"
            )
        if (
            self.fault_type is FaultType.FP16_EXPONENT_BIT
            and not set(self.bit_positions).issubset({10, 11, 12, 13, 14})
        ):
            raise ValueError("Exponent faults may only use FP16 bits 10..14")

    @property
    def flat_index(self) -> int:
        return self.sequence_index * self.expected_out_features + self.feature_index

    @property
    def spec_id(self) -> str:
        payload = {"schema": "ft2-fault-v2", **asdict(self)}
        payload["fault_type"] = self.fault_type.value
        payload["bit_positions"] = list(self.bit_positions)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fault_type"] = self.fault_type.value
        result["bit_positions"] = list(self.bit_positions)
        result["spec_id"] = self.spec_id
        result["flat_index"] = self.flat_index
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FaultSpec":
        fields = {
            key: item
            for key, item in value.items()
            if key not in {"spec_id", "flat_index"}
        }
        fields["bit_positions"] = tuple(fields["bit_positions"])
        spec = cls(**fields)
        supplied = value.get("spec_id")
        if supplied is not None and supplied != spec.spec_id:
            raise ValueError("Fault spec_id does not match its content")
        return spec


def fp16_bits_at(value: torch.Tensor, flat_index: int) -> int:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise TypeError("Expected a strided torch.Tensor")
    if value.dtype != torch.float16:
        raise TypeError(f"Expected torch.float16, got {value.dtype}")
    if not 0 <= flat_index < value.numel():
        raise IndexError("flat_index is outside the tensor")
    signed = int(
        value.detach().contiguous().view(torch.int16).reshape(-1)[flat_index].item()
    )
    return signed & 0xFFFF


def flip_fp16_bits(
    value: torch.Tensor,
    flat_index: int,
    bit_positions: Sequence[int],
) -> torch.Tensor:
    positions = tuple(int(bit) for bit in bit_positions)
    if value.dtype != torch.float16:
        raise TypeError(f"Fault injection requires float16, got {value.dtype}")
    if value.layout != torch.strided:
        raise TypeError("Fault injection requires a strided tensor")
    if not 0 <= flat_index < value.numel():
        raise IndexError("flat_index is outside the tensor")
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("bit_positions must be non-empty and distinct")
    if any(bit < 0 or bit > 15 for bit in positions):
        raise ValueError("FP16 bit positions must be in [0, 15]")
    if torch.is_grad_enabled() and value.requires_grad:
        raise RuntimeError("Fault injection is inference-only")

    unsigned_mask = 0
    for bit in positions:
        unsigned_mask |= 1 << bit
    signed_mask = unsigned_mask if unsigned_mask < 1 << 15 else unsigned_mask - (1 << 16)
    result = value.detach().contiguous().clone()
    bits = result.view(torch.int16).reshape(-1)
    mask = torch.tensor(signed_mask, dtype=torch.int16, device=result.device)
    bits[flat_index] = torch.bitwise_xor(bits[flat_index], mask)
    return result


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value


@dataclass(frozen=True)
class InjectionTrace:
    spec_id: str
    observed_step: int
    site_key: str
    output_shape: Tuple[int, ...]
    flat_index: int
    bit_positions: Tuple[int, ...]
    before_bits: int
    after_bits: int
    protected_bits: int
    before_value: float
    after_value: float
    protected_value: float
    bounds: Optional[Bounds]
    correction_action: str

    def __post_init__(self) -> None:
        if len(self.spec_id) != 64:
            raise ValueError("spec_id must be a SHA-256 digest")
        for value in (self.before_bits, self.after_bits, self.protected_bits):
            if not 0 <= value <= 0xFFFF:
                raise ValueError("Raw FP16 bits must be uint16 values")

    @property
    def xor_mask(self) -> int:
        mask = 0
        for bit in self.bit_positions:
            mask |= 1 << bit
        return mask

    @property
    def bit_flip_verified(self) -> bool:
        return self.after_bits == (self.before_bits ^ self.xor_mask)

    @property
    def hamming_distance(self) -> int:
        return (self.before_bits ^ self.after_bits).bit_count()

    @property
    def detected(self) -> bool:
        return self.correction_action != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "observed_step": self.observed_step,
            "site_key": self.site_key,
            "output_shape": list(self.output_shape),
            "flat_index": self.flat_index,
            "bit_positions": list(self.bit_positions),
            "before_bits_hex": f"0x{self.before_bits:04x}",
            "after_bits_hex": f"0x{self.after_bits:04x}",
            "protected_bits_hex": f"0x{self.protected_bits:04x}",
            "before_value": _json_number(self.before_value),
            "after_value": _json_number(self.after_value),
            "protected_value": _json_number(self.protected_value),
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "correction_action": self.correction_action,
            "detected": self.detected,
            "bit_flip_verified": self.bit_flip_verified,
            "hamming_distance": self.hamming_distance,
        }
