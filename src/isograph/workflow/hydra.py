"""Hydra helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from isograph.workflow.config import BenchmarkCommandConfig, CompareCommandConfig, FitCommandConfig

T = TypeVar("T")


def register_configs() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="benchmark_schema", node=BenchmarkCommandConfig)
    cs.store(name="fit_schema", node=FitCommandConfig)
    cs.store(name="compare_schema", node=CompareCommandConfig)


def _convert_value(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    if origin in (UnionType, None) and getattr(target_type, "__args__", None):
        non_none = [arg for arg in get_args(target_type) if arg is not type(None)]
        if value is None:
            return None
        if len(non_none) == 1:
            return _convert_value(value, non_none[0])
    if target_type is Path and value is not None:
        return Path(value)
    if origin is list and value is not None:
        inner = get_args(target_type)[0]
        return [_convert_value(item, inner) for item in value]
    if is_dataclass(target_type) and value is not None:
        return instantiate_dataclass(target_type, value)
    return value


def instantiate_dataclass(config_type: type[T], payload: dict[str, Any]) -> T:
    type_hints = get_type_hints(config_type)
    values: dict[str, Any] = {}
    for field in fields(config_type):
        current = payload.get(field.name)
        values[field.name] = _convert_value(current, type_hints.get(field.name, field.type))
    return config_type(**values)


def load_config(config_name: str, overrides: list[str]) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    register_configs()
    config_dir = (Path(__file__).resolve().parents[3] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)
