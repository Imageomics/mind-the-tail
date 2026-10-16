"""YAML config loading shared by all command-line entry points."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import types
import typing as tp

import yaml

T = tp.TypeVar("T")


def _expand(value: object) -> object:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _coerce(annotation: object, value: object) -> object:
    """Convert a YAML value to the annotated dataclass field type."""
    if value is None:
        return None
    origin = tp.get_origin(annotation)
    if origin in (tp.Union, types.UnionType):
        options = [arg for arg in tp.get_args(annotation) if arg is not type(None)]
        return _coerce(options[0], value) if len(options) == 1 else value
    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, tp.cast(dict, value))
    if annotation is pathlib.Path:
        return pathlib.Path(str(value)).expanduser()
    if origin is tuple:
        return tuple(value)  # type: ignore[arg-type]
    return value


def from_dict(schema: type[T], data: dict[str, object]) -> T:
    hints = tp.get_type_hints(schema)
    names = {field.name for field in dataclasses.fields(schema)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"Unknown {schema.__name__} options: {unknown}")
    return schema(**{key: _coerce(hints[key], value) for key, value in data.items()})


def load_config(path: pathlib.Path, schema: type[T], overrides: tp.Sequence[str] = ()) -> T:
    """Load YAML into ``schema``; ``overrides`` are ``dotted.key=value`` strings."""
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = raw
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(value)
    return from_dict(schema, tp.cast(dict, _expand(raw)))


def to_jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def dump_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cli_config(schema: type[T], description: str) -> T:
    """Parse ``--config path.yaml [--set key=value ...]`` into ``schema``."""
    import logging

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")
    return load_config(args.config, schema, args.set)
