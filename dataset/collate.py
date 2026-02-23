from typing import List
import torch
from typing import Dict, Any
from dataclasses import dataclass



def collate_positions_from_games(batch):
    batch = [b for b in batch if b.get("valid", False)]
    if len(batch) == 0:
        return {"images": torch.empty(0)}

    out = {
        "labels64": torch.stack([b["labels64"] for b in batch], dim=0),    # [B,64]

        "turn": torch.stack([b["turn"] for b in batch], dim=0),            # [B]
        "castling": torch.stack([b["castling"] for b in batch], dim=0),    # [B,4]
        "ep_square": torch.stack([b["ep_square"] for b in batch], dim=0),  # [B]

        "move_from": torch.stack([b["move_from"] for b in batch], dim=0),
        "move_to": torch.stack([b["move_to"] for b in batch], dim=0),
        "move_promo": torch.stack([b["move_promo"] for b in batch], dim=0),

        "next_move_from": torch.stack([b["next_move_from"] for b in batch], dim=0),
        "next_move_to": torch.stack([b["next_move_to"] for b in batch], dim=0),
        "next_move_promo": torch.stack([b["next_move_promo"] for b in batch], dim=0),

        "game_idx": torch.stack([b["game_idx"] for b in batch], dim=0),    # [B]
        "ply_idx": torch.stack([b["ply_idx"] for b in batch], dim=0),      # [B]
    }

    if "fen" in batch[0]:
        out["fen"] = [b["fen"] for b in batch]


    if "piece_probs" in batch[0]:
        out["piece_probs"] = torch.stack([b["piece_probs"] for b in batch], dim=0)  # [B,64,13]

    if "images" in batch[0]:
        out["images"] = torch.stack([b["images"] for b in batch], dim=0)  # [B,3,H,W]
    else:
        out["images"] = torch.empty(0)
    return out

@dataclass
class CollatePositionsFromGames:
    # put tunable params here if you have any
    # e.g. pad_to: int = 64

    def __call__(self, batch):
        # call your existing function, or inline its logic
        return collate_positions_from_games(batch)