#!/usr/bin/env bash
# Register this script as a UCI engine; set CHECKPOINT/MODEL_TYPE as needed.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_EXECUTABLE:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python3; fi
MODEL_TYPE="${MODEL_TYPE:-decoder}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/chess_llm_decoder_v12.pt}"
N_LAYERS="${N_LAYERS:-12}"
cd "$PROJECT_ROOT"
exec "$PYTHON" -u -m utils.uci_engine \
  --model_type "$MODEL_TYPE" --ckpt "$CHECKPOINT" --n_layers "$N_LAYERS" \
  --tokenizer_path "$PROJECT_ROOT/tokenizers/chess_uci_vocab.json" "$@"
