
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


DATASET = "Lichess/chess-puzzles-with-games"
SHARD = "data/train-00000-of-00014.parquet"


def keep(row):
    """Select difficult middlegame puzzles from strong players."""
    return (
        row["WhiteElo"] >= 2400
        and row["BlackElo"] >= 2400
        and row["Rating"] >= 2200
        and "middlegame" in row["Themes"]
    )


def main():
    # Download one shard into the Hugging Face cache.
    # Subsequent runs reuse the downloaded file.
    path = hf_hub_download(
        repo_id=DATASET,
        filename=SHARD,
        repo_type="dataset",
    )

    # Read directly from the local file, avoiding HTTP streaming.
    with pq.ParquetFile(path) as parquet:
        count = 0

        for batch in parquet.iter_batches(
            batch_size=512,
            columns=[
                "WhiteElo",
                "BlackElo",
                "Rating",
                "Themes",
                "PuzzleId",
                "GameId",
                "FEN",
                "Moves",
                "movetext",
            ],
            use_threads=False,
        ):
            for row in batch.to_pylist():
                if not keep(row):
                    continue

                print(
                    f"Players: {row['WhiteElo']} / {row['BlackElo']} | "
                    f"Puzzle: {row['Rating']} | "
                    f"ID: {row['PuzzleId']}"
                )

                count += 1
                if count >= 10:
                    return

    print(f"Found {count} matching puzzles.")


if __name__ == "__main__":
    main()
    print("Completed successfully.")
