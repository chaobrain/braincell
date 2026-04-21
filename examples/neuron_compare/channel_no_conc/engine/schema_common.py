"""Shared schema helpers for NEURON vs braincell comparison templates."""



from typing import Any, Mapping


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__!s}.")
    return value


def require_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
    return value


def require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}.")
    return float(value)


def require_literal(value: Any, *, name: str, allowed: tuple[str, ...]) -> str:
    text = require_str(value, name=name)
    if text not in allowed:
        raise ValueError(f"{name} must be one of {allowed!r}, got {text!r}.")
    return text


def require_submapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in payload:
        raise KeyError(f"Missing required section {key!r}.")
    return require_mapping(payload[key], name=key)

