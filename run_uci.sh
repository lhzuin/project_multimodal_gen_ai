

# DO --> chmod +x run_uci.sh
# RUN --> ./run_uci.sh


#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Option A: use the project's venv (recommended if they created one)
PYTHON="$PROJECT_ROOT/venv/bin/python"

# Option B: fallback to system python3 if no venv exists
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

CKPT="$PROJECT_ROOT/checkpoints/chess_llm_decoder_v3.pt"
TOKENIZER="$PROJECT_ROOT/tokenizers/chess_uci_vocab.json"

cd "$PROJECT_ROOT"

"$PYTHON" -u -m utils.uci_engine \
  --model_type decoder \
  --ckpt "$CKPT" \
  --tokenizer_path "$TOKENIZER"