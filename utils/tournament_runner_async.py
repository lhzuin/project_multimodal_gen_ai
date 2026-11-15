#!/usr/bin/env python3
"""Original tournament CLI plus bounded game, model and Stockfish parallelism."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.async_tournament.runner import main


if __name__ == "__main__":
    main()
