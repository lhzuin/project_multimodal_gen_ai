"""Explicit, serializable choices for the mixed Lichess RL dataset."""

from dataclasses import dataclass
from pathlib import Path


BUCKETS = ("tactics", "quiet", "endgame")
SHARES = {"tactics": 0.5, "quiet": 0.3, "endgame": 0.2}


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "processed_games/rl/lichess_mixed_v1"
    dataset_id: str = "Lichess/chess-puzzles-with-games"
    revision: str = "3510a01f320eaaab7b7775ebc73afcfe0b33b4c2"
    shards: tuple[str, ...] = ("data/train-00000-of-00014.parquet",)
    local_parquet: tuple[str, ...] = ()
    train_size: int = 10_000
    validation_size: int = 1_000
    test_size: int = 1_000
    seed: int = 42
    candidate_multiplier: int = 4
    min_player_elo: int = 1800
    min_puzzle_rating: int = 1200
    max_puzzle_rating: int = 2400
    max_rating_deviation: int = 85
    min_popularity: int = 70
    min_plays: int = 100
    max_solution_plies: int = 10  # includes the trigger, matching rollout budget
    quiet_min_ply: int = 20
    quiet_max_ply: int = 100
    quiet_max_abs_cp: int = 150
    quiet_max_delta_cp: int = 30
    confidence_checkpoint: str = "checkpoints/chess_encoder_seq_balanced_v1.best.pt"
    confidence_max_probability: float = 0.95
    confidence_batch_size: int = 8
    confidence_temperature: float = 1.0
    device: str = "cpu"
    torch_threads: int = 2
    exclusion_openings: tuple[str, ...] = ("utils/opening_suite_large.json",)
    stockfish: str = (
        "/Users/luiszuin/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
    )

    def __post_init__(self) -> None:
        for name in ("train_size", "validation_size", "test_size"):
            value = getattr(self, name)
            if value <= 0 or value % 20:
                raise ValueError(
                    f"{name} must be positive and divisible by 20 for exact bucket/color quotas"
                )
        if not 0 < self.confidence_max_probability < 1:
            raise ValueError("confidence_max_probability must be in (0, 1)")
        if (
            self.candidate_multiplier < 2
            or self.confidence_batch_size < 1
            or self.torch_threads < 1
        ):
            raise ValueError("invalid candidate/compute budget")
        if self.max_solution_plies < 4 or self.confidence_temperature <= 0:
            raise ValueError("invalid solution horizon or temperature")
        if self.min_puzzle_rating > self.max_puzzle_rating:
            raise ValueError("puzzle rating bounds are reversed")

    @property
    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_size,
            "validation": self.validation_size,
            "test": self.test_size,
        }

    def quotas(self) -> dict[tuple[str, str, str], int]:
        return {
            (split, bucket, color): int(size * SHARES[bucket]) // 2
            for split, size in self.split_sizes.items()
            for bucket in BUCKETS
            for color in ("white", "black")
        }


def resolve_sources(config: BuildConfig) -> list[Path]:
    """Reuse cached immutable shards; no implicit multi-GB downloads."""
    if config.local_parquet:
        paths = [Path(path).expanduser().resolve() for path in config.local_parquet]
    else:
        from huggingface_hub import hf_hub_download

        paths = [
            Path(
                hf_hub_download(
                    repo_id=config.dataset_id,
                    repo_type="dataset",
                    revision=config.revision,
                    filename=shard,
                    local_files_only=True,
                )
            )
            for shard in config.shards
        ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths
