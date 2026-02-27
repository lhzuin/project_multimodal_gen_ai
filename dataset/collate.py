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



def collate_game_sequences(batch):
    batch = [b for b in batch if b.get("valid", False)]
    if len(batch) == 0:
        return {"piece_probs": torch.empty(0)}

    # lengths
    lengths = torch.tensor([int(b["seq_len"].item()) for b in batch], dtype=torch.long)
    B = len(batch)
    T = int(lengths.max().item())

    # allocate padded tensors
    #piece_probs = torch.zeros((B, T, 64, 13), dtype=torch.float32)
    labels64 = torch.zeros((B, T, 64), dtype=torch.long)
    turn = torch.zeros((B, T), dtype=torch.long)
    castling = torch.zeros((B, T, 4), dtype=torch.long)
    ep_square = torch.zeros((B, T), dtype=torch.long)

    move_from = torch.full((B, T), -1, dtype=torch.long)
    move_to = torch.full((B, T), -1, dtype=torch.long)
    move_promo = torch.full((B, T), -1, dtype=torch.long)

    attn_mask = torch.zeros((B, T), dtype=torch.bool)

    fen = None
    if "fen" in batch[0]:
        fen = []

    game_idx = torch.stack([b["game_idx"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        Li = int(b["seq_len"].item())
        attn_mask[i, :Li] = True

        # if "piece_probs" in b:
        #     piece_probs[i, :Li] = b["piece_probs"]
        turn[i, :Li] = b["turn"]
        castling[i, :Li] = b["castling"]
        ep_square[i, :Li] = b["ep_square"]
        labels64[i, :Li] = b["labels64"]

        move_from[i, :Li] = b["move_from"]
        move_to[i, :Li] = b["move_to"]
        move_promo[i, :Li] = b["move_promo"]

        if fen is not None:
            fen.append(b["fen"])  # list[str] length Li

    out = {
        #"piece_probs": piece_probs,   # [B,T,64,13]
        "labels64": labels64,       # [B,T,64]
        "turn": turn,                 # [B,T]
        "castling": castling,         # [B,T,4]
        "ep_square": ep_square,       # [B,T]
        "move_from": move_from,       # [B,T]
        "move_to": move_to,           # [B,T]
        "move_promo": move_promo,     # [B,T]
        "attn_mask": attn_mask,       # [B,T] True where valid
        "seq_len": lengths,           # [B]
        "game_idx": game_idx,         # [B]
    }
    if fen is not None:
        out["fen"] = fen              # list[list[str]] (ragged)

    return out


@dataclass
class CollateGameSequences:
    def __call__(self, batch):
        return collate_game_sequences(batch)