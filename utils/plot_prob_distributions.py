#!/usr/bin/env python3
"""
Plot teacher vs model policy distributions on random positions from the distillation DB.

For each sampled position:
  1) Teacher distribution: softmax(cp/mate scores with tau) over the stored top-K moves.
  2) Model distribution: softmax over legal moves using encoder's (from,to) logits (+ promo logits factor).

Then we plot both distributions on the union:
  union = teacher_topK ∪ model_top10

Usage example:
  python plot_distill_encoder_distributions.py \
    --cfg distillation_encoder_v3.yaml \
    --ckpt checkpoints/distill_encoder_v3_mixed.pt \
    --n 12 \
    --outdir debug_plots \
    --device cuda

Notes:
- This script imports "private" helpers from distillation_db.py (_MultiShardIndex, _scores_to_probs).
  That’s deliberate to reuse your infrastructure with minimal duplication.
- You may need to adjust imports depending on your repo layout (models.* vs local files).
"""

from omegaconf import OmegaConf
import os
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import chess

from omegaconf import OmegaConf

# ---- repo-local imports (adjust if needed) ----
from dataset.distillation_db import DistillationEncoderDataset, DistillationShardReader, _MultiShardIndex, _scores_to_probs
from models.chess_encoder_player import ChessEncoderPlayer


PROMO_TO_CLASS = {
    None: 0,  # "none"
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}




def register_minimal_hydra_resolver():
    # Only supports ${hydra:runtime.cwd}
    if not OmegaConf.has_resolver("hydra"):
        def _hydra_resolver(key: str):
            if key == "runtime.cwd":
                return os.getcwd()
            raise ValueError(f"Unsupported hydra interpolation: {key}")
        OmegaConf.register_new_resolver("hydra", _hydra_resolver)

def load_model_from_cfg(cfg, ckpt_path: str, device: str):
    """
    Builds ChessEncoderPlayer from cfg.model.instance fields and loads checkpoint.
    Works whether ckpt is a full model state_dict or a dict containing "model".
    """
    mcfg = cfg.model.instance

    model = ChessEncoderPlayer(
        img_size=int(mcfg.img_size),
        encoder_dim=int(mcfg.encoder_dim),
        n_heads=int(mcfg.n_heads),
        n_layers=int(mcfg.n_layers),
        encoder_dropout=float(mcfg.encoder_dropout),
        ep_embed_dim=int(mcfg.ep_embed_dim),
        vit_path=mcfg.get("vit_path", None),
        freeze_vit=bool(mcfg.get("freeze_vit", True)),
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_model] missing keys:", len(missing))
        print("[load_model] unexpected keys:", len(unexpected))
    model.eval()
    return model


@torch.no_grad()
def model_distribution_over_legal_moves(model, *, item, device: str, temperature: float = 1.0):
    """
    Returns dict {uci: prob} over ALL legal moves in the position.
    Uses:
      logit(move) = policy_logits[from,to] + promo_logits[promo_class]
    then softmax over legal moves.
    """
    fen = item["fen"]
    board = chess.Board(fen)

    piece_probs = item["piece_probs"].unsqueeze(0).to(device)  # [1,64,13]
    turn = item["turn"].unsqueeze(0).to(device)                # [1]
    castling = item["castling"].unsqueeze(0).to(device)        # [1,4]
    ep_square = item["ep_square"].unsqueeze(0).to(device)      # [1]

    out = model(piece_probs=piece_probs, turn=turn, castling=castling, ep_square=ep_square)
    policy_logits = out["policy_logits"][0]  # [64,64]
    promo_logits = out["promo_logits"][0]    # [5]

    moves = list(board.legal_moves)
    if len(moves) == 0:
        return {}

    logits = []
    uci_list = []
    for mv in moves:
        base = policy_logits[mv.from_square, mv.to_square]
        promo_class = PROMO_TO_CLASS.get(mv.promotion, 0)
        full = base + promo_logits[promo_class]
        logits.append(full)
        uci_list.append(mv.uci())

    logits = torch.stack(logits, dim=0).float()
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

    return {u: float(p) for u, p in zip(uci_list, probs)}


def teacher_distribution_from_row(row, *, tau: float, mate_cp: int):
    """
    Returns dict {uci: prob} over the stored top-K teacher moves for this position.
    """
    uci_topk = json.loads(row["uci_topk_json"])
    cp_topk = json.loads(row["cp_topk_json"])
    mate_topk = json.loads(row["mate_topk_json"])

    # _scores_to_probs already does cp/mate handling + temperature.
    probs = _scores_to_probs(cp_topk, mate_topk, tau=float(tau), mate_cp=int(mate_cp)).cpu().numpy()
    return {u: float(p) for u, p in zip(uci_topk, probs)}


def choose_union_moves(teacher_probs: dict, model_probs: dict, *, model_topk: int = 10):
    """
    union = teacher moves ∪ top model moves
    Returned as a list of uci strings sorted by max(teacher,model) descending.
    """
    model_sorted = sorted(model_probs.items(), key=lambda kv: kv[1], reverse=True)
    top_model = [u for u, _ in model_sorted[:model_topk]]

    union = set(teacher_probs.keys()) | set(top_model)

    def score(u):
        return max(teacher_probs.get(u, 0.0), model_probs.get(u, 0.0))

    ordered = sorted(list(union), key=score, reverse=True)
    return ordered


def plot_one_position(*, idx: int, fen: str, moves: list, teacher_probs: dict, model_probs: dict, outpath: Path):
    """
    Makes a single figure (bar plot with two distributions).
    """
    t = np.array([teacher_probs.get(u, 0.0) for u in moves], dtype=np.float32)
    m = np.array([model_probs.get(u, 0.0) for u in moves], dtype=np.float32)

    x = np.arange(len(moves))
    w = 0.42

    plt.figure(figsize=(max(10, 0.6 * len(moves)), 5))
    plt.bar(x - w / 2, t, width=w, label="Teacher (tau-softmax on top-K)")
    plt.bar(x + w / 2, m, width=w, label="Model (softmax over legal moves)")

    plt.xticks(x, moves, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Probability")
    plt.title(f"idx={idx} | {fen.split(' ')[0]} | turn={'w' if ' w ' in (' ' + fen + ' ') else 'b'}")
    plt.ylim(0.0, max(0.05, float(max(t.max(initial=0.0), m.max(initial=0.0))) * 1.15))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/distillation_encoder_v3.yaml", help="Path to distillation_encoder_*.yaml")
    ap.add_argument("--ckpt", type=str, default="checkpoints/chess_encoder_player_v3.pt", help="Path to encoder checkpoint (.pt)")
    ap.add_argument("--n", type=int, default=12, help="Number of random positions to plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--outdir", type=str, default="distill_plots")
    ap.add_argument("--model_topk", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0, help="Extra temperature for model softmax (debug knob)")
    args = ap.parse_args()

    register_minimal_hydra_resolver()
    cfg = OmegaConf.load(args.cfg)

    meta_path = str(cfg.distill.meta_path)
    tau = float(cfg.distill.tau)
    mate_cp = int(cfg.distill.mate_cp)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Dataset gives us fen + state tensors (piece_probs/turn/castling/ep_square)
    ds = DistillationEncoderDataset(
        meta_path=meta_path,
        tau=tau,
        mate_cp=mate_cp,
        return_text=False,
        return_state_tensors=True,
    )

    # To get teacher UCI strings + cp/mate, reuse the shard readers directly
    index = _MultiShardIndex(meta_path)
    readers = [DistillationShardReader(p) for (p, _) in index.shards]

    # Load model
    model = load_model_from_cfg(cfg, args.ckpt, args.device)

    rng = np.random.default_rng(args.seed)
    n = min(int(args.n), len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False).tolist()

    print(f"[info] meta_path={meta_path}")
    print(f"[info] tau={tau}, mate_cp={mate_cp}")
    print(f"[info] plotting n={n} positions into {outdir.resolve()}")

    for k, idx in enumerate(idxs):
        item = ds[idx]
        if not item.get("valid", False):
            continue

        # fetch raw DB row for this idx to get uci_topk/cp/mate
        shard_id, local = index.locate(int(idx))
        row = readers[shard_id].get_by_index(int(local))

        teacher_probs = teacher_distribution_from_row(row, tau=tau, mate_cp=mate_cp)
        model_probs = model_distribution_over_legal_moves(
            model, item=item, device=args.device, temperature=args.temperature
        )

        union_moves = choose_union_moves(teacher_probs, model_probs, model_topk=int(args.model_topk))

        fen = item["fen"]
        outpath = outdir / f"pos_{k:03d}_idx_{idx}.png"
        print(row)
        plot_one_position(
            idx=int(idx),
            fen=fen,
            moves=union_moves,
            teacher_probs=teacher_probs,
            model_probs=model_probs,
            outpath=outpath,
        )

    print("[done]")


if __name__ == "__main__":
    main()