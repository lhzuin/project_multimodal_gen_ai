"""Reuse decoder collation and apply its supervision mask to board targets."""

import torch

from dataset.collate_llm import CollateLLMMoves


class CollateLLMBoardMoves(CollateLLMMoves):
    """Pad ``[T,64]`` board labels without next-token shifting them.

    Reconstruction uses the same active positions as the move objective.
    Balanced-plan context-only tokens and batch padding contribute no board loss.
    """

    def __call__(self, batch):
        items = [b for b in batch if isinstance(b, dict) and b.get("valid", True)]
        out = super().__call__(items)
        if not out["valid"]:
            return out
        batch_size, length = out["input_ids"].shape
        labels = torch.full((batch_size, length, 64), -100, dtype=torch.long)
        for i, item in enumerate(items):
            target = item["board_labels"]
            if target.shape != (len(item["move_ids"]), 64):
                raise ValueError("board_labels must have shape [len(move_ids), 64]")
            if target.dtype != torch.long:
                raise ValueError("board_labels must have dtype torch.long")
            count = min(length, len(target))
            labels[i, :count] = target[:count]
        labels[out["labels"] == -100] = -100
        out["board_labels"] = labels
        return out
