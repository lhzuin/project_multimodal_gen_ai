# RL experiment configurations

Run from the repository root:

```bash
.venv/bin/python -m train.rloo --config configs/rl/rloo_encoder_lichess_mixed_v2.yaml
```

- `rloo_encoder_lichess_mixed_v1.yaml`: exact settings of the completed original
  experiment, converted from JSON without changing any values.
- `rloo_encoder_lichess_mixed_v2.yaml`: exact settings of the previous v2 JSON;
  JSONL logging, optional W&B (disabled by default), explicit norm limit 1.0,
  checkpointing every 10 batches, and separate v2 output filenames.
- `examples/`: small GRPO/RLOO encoder and decoder integration examples.

Both versions initialize from the original supervised checkpoint. V2 is not a
resume or extra epoch unless you explicitly configure it. Do not rerun v1 casually:
its original output paths still point at the completed experiment's checkpoints.

Launchers and the held-out evaluator require `.yaml` or `.yml` and share safe
loading with duplicate-key checks. This is plain YAML, not Hydra composition:
there is no `defaults` expansion, `${...}` interpolation, or `--config-name`.
Training classes still accept config objects and do not depend on Hydra or YAML.
Relative file paths are resolved from the working directory, not this folder.

Dataset manifests, build provenance, evaluation results and metric journals remain
JSON/JSONL. The six migrated experiment JSON files were removed. A canonical
settings-hash regression test guards the exact v1/v2 migration; the dataset
manifest retains the original v1 JSON hash and references the relocated YAML.
