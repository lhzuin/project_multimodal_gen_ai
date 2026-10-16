"""Safe, explicit YAML configuration IO without Hydra-specific dependencies."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently overriding experiment settings."""


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a .yaml/.yml mapping; unsafe tags and duplicate keys are rejected.

    Paths in the configuration remain relative to the project working directory,
    just as before. Hydra defaults/interpolation/CLI overrides are not applied.
    """
    path = Path(path)
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("configuration files must use .yaml or .yml")
    with path.open(encoding="utf-8") as handle:
        values = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(values, Mapping):
        raise ValueError("configuration must contain a YAML mapping")
    return dict(values)


def dump_config(config: Mapping[str, Any]) -> str:
    """Serialize explicit values (including scientific-notation floats) safely."""
    return yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True)
