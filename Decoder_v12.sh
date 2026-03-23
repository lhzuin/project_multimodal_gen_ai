#!/bin/bash
cd "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai" || exit 1

"/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/.venv/bin/python" -u -m utils.uci_engine \
  --model_type decoder \
  --ckpt "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai/checkpoints/chess_llm_decoder_v12.pt" \
  --tokenizer_path "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai/tokenizers/chess_uci_vocab.json" \
  --temperature 0.5 \
  --greedy \
  --n_layers 12