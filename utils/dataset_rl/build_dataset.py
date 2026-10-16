"""Build the mixed dataset: python -m utils.dataset_rl.build_dataset --help."""

import argparse
from dataclasses import fields
import logging
from pathlib import Path

from .builder import DatasetBuilder
from .config import BuildConfig
from train.rl.config_io import load_config as load_yaml


def load_config(path: Path | None) -> BuildConfig:
    values = load_yaml(path) if path else {}
    known = {field.name for field in fields(BuildConfig)}
    if set(values) - known:
        raise ValueError(f"unknown build options: {sorted(set(values) - known)}")
    for name in ("shards", "local_parquet", "exclusion_openings"):
        if name in values:
            values[name] = tuple(values[name])
    return BuildConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional BuildConfig YAML")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    DatasetBuilder(load_config(args.config)).build()


if __name__ == "__main__":
    main()
