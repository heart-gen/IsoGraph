"""Hydra helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


def _convert_value(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    if origin in (UnionType, None) and getattr(target_type, "__args__", None):
        non_none = [arg for arg in get_args(target_type) if arg is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _convert_value(value, non_none[0])
    if target_type is float and value is not None:
        return float(value)
    if target_type is int and value is not None:
        return int(value)
    if target_type is bool and value is not None:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Cannot coerce {value!r} to bool")
    if target_type is str and value is not None:
        return str(value)
    if target_type is Path and value is not None:
        return Path(value)
    if origin is list and value is not None:
        inner = get_args(target_type)[0]
        return [_convert_value(item, inner) for item in value]
    if is_dataclass(target_type) and value is not None:
        return instantiate_dataclass(target_type, value)
    return value


def instantiate_dataclass(config_type: type[T], payload: dict[str, Any]) -> T:
    import dataclasses

    type_hints = get_type_hints(config_type)
    values: dict[str, Any] = {}
    for field in fields(config_type):
        if field.name not in payload or payload[field.name] is None:
            # Prefer the field's own default over a None payload value.
            if field.default is not dataclasses.MISSING:
                values[field.name] = field.default
                continue
            if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                values[field.name] = field.default_factory()  # type: ignore[misc]
                continue
        current = payload.get(field.name)
        values[field.name] = _convert_value(current, type_hints.get(field.name, field.type))
    return config_type(**values)


def load_config(config_name: str, overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = (Path(__file__).resolve().parents[3] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)
