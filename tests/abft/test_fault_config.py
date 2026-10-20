"""Fault selector validation for direct and mapping-based configuration."""
import pytest

from abft.config import FaultConfig, parse_fault_config


@pytest.mark.parametrize("role", [None, "forward", "dgrad", "wgrad"])
def test_supported_fault_roles(role):
    assert FaultConfig(role=role).role == role
    assert parse_fault_config({"role": role}).role == role


@pytest.mark.parametrize("role", ["wgard", "", "FORWARD", "none", 1, [], {}])
def test_invalid_fault_roles_rejected(role):
    with pytest.raises(ValueError, match="fault role"):
        FaultConfig(role=role)
    with pytest.raises(ValueError, match="fault role"):
        parse_fault_config({"role": role})
