from twic_download import download_twic_zips, build_merged_pgn_from_zips, twic_range
from offsets import ensure_offsets
from torch.utils.data import DataLoader

from dataset import ChessGameSampleDataset
from collate import collate_positions_from_games

from pathlib import Path

HERE = Path(__file__).resolve().parent                 # .../dataset
PROJECT_ROOT = HERE.parent                             # .../project_multimodal_gen_ai
SPRITES_DIR = PROJECT_ROOT / "dataset" / "sprites"     # .../project_multimodal_gen_ai/dataset/sprites
print(f"Using sprites from {SPRITES_DIR}")





def main():
    # 0) Download + merge TWIC PGNs (do this once, or keep updating weekly)
    zips = download_twic_zips(twic_range(1601, 1610), out_dir="twic_zips")
    build_merged_pgn_from_zips(zips, out_pgn_path="games.pgn")

    # 1) Ensure offsets exist and match current games.pgn
    ensure_offsets("games.pgn", "games_index.json")

    # 2) Dataset
    ds = ChessGameSampleDataset(
        pgn_path="games.pgn",
        index_path="games_index.json",
        sprites_dir=str(SPRITES_DIR),
        resolution=256,
        sample_ratio=0.10,
        max_positions_per_game=32,
        seed=42,
        apply_augmentations=True,
        return_fen=False,
    )

    # 3) DataLoader
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_positions_from_games,
        pin_memory=False,  # MPS: avoids the warning (optional)
        persistent_workers=True,  # optional speed-up
    )

    batch = next(iter(dl))
    print(batch["images"].shape)
    print(batch["labels64"].shape)

    import matplotlib.pyplot as plt
    for i in range (10):
        img = batch["images"][i].permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    # If you ever freeze to an executable, you'd add freeze_support() here.
    main()


    #Ideas: turn the board depending on who's turn is it
    #Adding a pool of different sprites and backgrounds and choose them randomly to create variety