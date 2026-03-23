from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .dataset import ChessGameSampleDataset
from .collate import collate_positions_from_games

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SPRITES_DIR = PROJECT_ROOT / "dataset" / "sprites"


def sanity_check_batch(batch, resolution: int):
    assert "images" in batch and batch["images"].ndim == 4, "images must be [B,3,H,W]"
    B, C, H, W = batch["images"].shape
    assert C == 3 and H == resolution and W == resolution, f"bad image shape: {batch['images'].shape}"

    assert batch["labels64"].shape == (B, 64), f"bad labels shape: {batch['labels64'].shape}"
    assert batch["labels64"].dtype in (torch.int64, torch.long), "labels64 must be int64"

    mn = int(batch["labels64"].min().item())
    mx = int(batch["labels64"].max().item())
    assert 0 <= mn and mx <= 12, f"labels64 out of range: min={mn}, max={mx}"

    assert batch["turn"].shape == (B,), f"turn shape: {batch['turn'].shape}"
    assert batch["castling"].shape == (B, 4), f"castling shape: {batch['castling'].shape}"
    assert batch["ep_square"].shape == (B,), f"ep_square shape: {batch['ep_square'].shape}"

    for k in ["move_from", "move_to", "move_promo", "next_move_from", "next_move_to", "next_move_promo"]:
        assert batch[k].shape == (B,), f"{k} shape: {batch[k].shape}"
        assert batch[k].dtype in (torch.int64, torch.long), f"{k} dtype: {batch[k].dtype}"

    for k in ["move_from", "move_to", "next_move_from", "next_move_to"]:
        v = batch[k]
        ok = ((v >= 0) & (v <= 63)) | (v == -1)
        assert bool(ok.all().item()), f"{k} has invalid squares"

    for k in ["move_promo", "next_move_promo"]:
        v = batch[k]
        ok = ((v >= 0) & (v <= 4)) | (v == -1)
        assert bool(ok.all().item()), f"{k} has invalid promo id"

    print(f"✅ Batch sanity OK. B={B}, images={batch['images'].shape}, labels={batch['labels64'].shape}")


def main():
    resolution = 256

    # Put samples file next to your test (or wherever you prefer)
    samples_path = str(HERE / "games_samples_s42_r0.1_m32.json")

    ds = ChessGameSampleDataset(
        pgn_path="processed_games/games.pgn",
        index_path="processed_games/games_index.json",
        sprites_dir=str(SPRITES_DIR),
        resolution=resolution,
        sample_ratio=0.10,
        max_positions_per_game=32,
        seed=42,
        apply_augmentations=True,
        return_fen=False,
        samples_path=samples_path,
        cache_size=64,
    )
    print(f"Dataset length (positions): {len(ds)}")
    print(f"Using samples file: {samples_path}")

    num_workers = 4
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_positions_from_games,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    batch = next(iter(dl))
    sanity_check_batch(batch, resolution=resolution)

    for i in range(min(8, batch["images"].shape[0])):
        print(f"move_from: {batch["move_from"][i]}")
        print(f"move_to: {batch["move_to"][i]}")
        print(f"next_move_from: {batch["next_move_from"][i]}")
        print(f"next_move_to: {batch["next_move_to"][i]}")
        img = batch["images"][i].permute(1, 2, 0).cpu().numpy()
        plt.figure()
        plt.imshow(img)
        plt.title(
            f"i={i} game={int(batch['game_idx'][i])} ply={int(batch['ply_idx'][i])} "
            f"turn={int(batch['turn'][i])} ep={int(batch['ep_square'][i])}"
        )
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()