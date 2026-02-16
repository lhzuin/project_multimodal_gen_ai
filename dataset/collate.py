from typing import List
import torch
from typing import Dict, Any

def collate_positions_from_games(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # drop invalid/empty samples
    batch = [b for b in batch if b.get("valid", False) and b["images"].shape[0] > 0]
    if len(batch) == 0:
        return {"images": torch.empty(0)}

    images = torch.cat([b["images"] for b in batch], dim=0)       # [B,3,H,W]
    labels64 = torch.cat([b["labels64"] for b in batch], dim=0)   # [B,64]

    turn = torch.cat([b["turn"] for b in batch], dim=0)           # [B]
    castling = torch.cat([b["castling"] for b in batch], dim=0)   # [B,4]
    ep_square = torch.cat([b["ep_square"] for b in batch], dim=0) # [B]

    move_from = torch.cat([b["move_from"] for b in batch], dim=0)
    move_to = torch.cat([b["move_to"] for b in batch], dim=0)
    move_promo = torch.cat([b["move_promo"] for b in batch], dim=0)

    next_move_from = torch.cat([b["next_move_from"] for b in batch], dim=0)
    next_move_to = torch.cat([b["next_move_to"] for b in batch], dim=0)
    next_move_promo = torch.cat([b["next_move_promo"] for b in batch], dim=0)

    # track origin
    game_idx = torch.cat([
        b["game_idx"].repeat(b["images"].shape[0]) for b in batch
    ], dim=0)  # [B]

    ply_idx = torch.cat([b["ply_idx"] for b in batch], dim=0)     # [B]

    out = {
        "images": images,
        "labels64": labels64,
        "turn": turn,
        "castling": castling,
        "ep_square": ep_square,
        "move_from": move_from,
        "move_to": move_to,
        "move_promo": move_promo,
        "next_move_from": next_move_from,
        "next_move_to": next_move_to,
        "next_move_promo": next_move_promo,
        "game_idx": game_idx,
        "ply_idx": ply_idx,
    }

    # optional fens if present
    if "fen" in batch[0]:
        fens = []
        for b in batch:
            fens.extend(b["fen"])
        out["fen"] = fens

    return out