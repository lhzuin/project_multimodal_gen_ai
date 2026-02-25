"""build_stockfish_dataset.py

High-level convenience script/class to:
  1) Download TWIC PGNs (optional)
  2) Build merged PGN + offsets
  3) Build a *samples* file (game_idx, ply_idx)
  4) Build a sharded SQLite distillation DB with Stockfish top-K per sampled position

This file intentionally DOES NOT change your existing training datasets.
It only adds the new distillation database logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

"""Note on imports

In your repo, these helper modules might live either at top-level (e.g.
twic_download.py, offsets.py, dataset.py) or inside a package (e.g.
dataset/twic_download.py, dataset/offsets.py, dataset/dataset.py).

To avoid forcing you to rearrange files, we use small import fallbacks.
"""

from utils.twic_download import download_twic_zips, build_merged_pgn_from_zips, twic_range 

from dataset.offsets import ensure_offsets 

from dataset.dataset import build_samples_file 

from dataset.distillation_db import DistillationDBBuilder


@dataclass
class StockfishDistillationBuildConfig:
    # Source games
    start_week: int
    end_week: int
    out_pgn_path: str = "processed_games/games_stockfish.pgn"
    out_index_path: str = "processed_games/games_stockfish_index.json"

    # Sampling
    sample_ratio: float = 0.15
    max_positions_per_game: int = 32
    seed: int = 42
    samples_path: str = "processed_games/games_stockfish_samples.json"

    # Stockfish
    engine_path: str = "stockfish"
    movetime_ms: int = 100
    k: int = 5
    min_depth: int = 1

    # Output DB
    out_db_dir: str = "distill_db"
    num_workers: int = 4


class BuildStockfishDistillationDB:
    def __init__(self, cfg: StockfishDistillationBuildConfig) -> None:
        self.cfg = cfg

    def prepare_pgn(self) -> None:
        """Download + merge TWIC into one PGN, then build offsets."""
        zips = download_twic_zips(twic_range(self.cfg.start_week, self.cfg.end_week), out_dir="twic_zips")
        build_merged_pgn_from_zips(zips, out_pgn_path=self.cfg.out_pgn_path)
        ensure_offsets(self.cfg.out_pgn_path, self.cfg.out_index_path)

    def prepare_samples(self) -> None:
        """Build (game_list_idx, ply_idx) sample list used by the distillation builder."""
        if os.path.exists(self.cfg.samples_path):
            return
        build_samples_file(
            pgn_path=self.cfg.out_pgn_path,
            index_path=self.cfg.out_index_path,
            out_samples_path=self.cfg.samples_path,
            sample_ratio=self.cfg.sample_ratio,
            max_positions_per_game=self.cfg.max_positions_per_game,
            seed=self.cfg.seed,
        )

    def build_db(self, *, overwrite: bool = False) -> str:
        """Build the sharded SQLite distillation DB. Returns meta.json path."""
        builder = DistillationDBBuilder(
            pgn_path=self.cfg.out_pgn_path,
            index_path=self.cfg.out_index_path,
            samples_path=self.cfg.samples_path,
            engine_path=self.cfg.engine_path,
            out_dir=self.cfg.out_db_dir,
            movetime_ms=self.cfg.movetime_ms,
            k=self.cfg.k,
            min_depth=self.cfg.min_depth,
            num_workers=self.cfg.num_workers,
            seed=self.cfg.seed,
        )
        return builder.build(overwrite=overwrite)


if __name__ == "__main__":
    # Example usage (edit values in-place):
    cfg = StockfishDistillationBuildConfig(
        start_week=1463,
        end_week=1483,
        sample_ratio=0.15,
        max_positions_per_game=32,
        engine_path=os.environ.get("STOCKFISH_PATH", "stockfish"),
        movetime_ms=100,
        k=5,
        num_workers=10,
        out_db_dir="distill_db",
    )
    b = BuildStockfishDistillationDB(cfg)
    b.prepare_pgn()
    b.prepare_samples()
    meta = b.build_db(overwrite=False)
    print(f"[distill_db] meta: {meta}")