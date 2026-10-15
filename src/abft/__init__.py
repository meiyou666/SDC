"""ABFT experiment interfaces with lazy loading of the GPU implementation."""
from .config import AbftConfig, FaultConfig, parse_abft_config, parse_fault_config


def execute(*args, **kwargs):
    from .engine import execute as implementation
    return implementation(*args, **kwargs)


def execute_prepared(*args, **kwargs):
    from .engine import execute_prepared as implementation
    return implementation(*args, **kwargs)


__all__ = ["AbftConfig", "FaultConfig", "parse_abft_config", "parse_fault_config", "execute", "execute_prepared"]
