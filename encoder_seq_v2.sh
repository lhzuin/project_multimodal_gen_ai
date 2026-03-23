#!/bin/bash
cd "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai" || exit 1

"/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/.venv/bin/python" -u -m utils.uci_engine \
  --model_type encoder \
  --ckpt "/Users/luiszuin/Desktop/Polytechnique/3AP2/multimodal_gen_ai/project_multimodal_gen_ai/checkpoints/chess_encoder_seq_v2_ep12.pt" \
  --temperature 0.5 \
  --greedy \
  --n_layers 8 \
  --version 2 \
  --device mps