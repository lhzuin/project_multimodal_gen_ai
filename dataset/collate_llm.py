# collate_llm.py
import torch

class CollateLLMMoves:
    def __init__(self, pad_id: int, piece_pad_id: int, max_len: int | None = None):
        self.pad_id = int(pad_id)
        self.p_pad_id = int(piece_pad_id)
        self.max_len = max_len

    def __call__(self, batch):
        # batch: list[dict] from PGNMoveDataset
        batch = [b for b in batch if isinstance(b, dict) and b.get("valid", True)]
        if len(batch) == 0:
            return {"valid": False}

        move_seqs = [b["move_ids"] for b in batch]
        piece_seqs = [b["piece_ids"] for b in batch]
        start_turn = torch.stack([b["start_turn"] for b in batch], dim=0)  # [B]

        lengths = [len(x) for x in move_seqs]
        T = max(lengths)
        if self.max_len is not None:
            T = min(T, int(self.max_len))

        B = len(move_seqs)

        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        piece_input_ids = torch.full((B, T), self.p_pad_id, dtype=torch.long)

        for i, (m, p) in enumerate(zip(move_seqs, piece_seqs)):
            m = m[:T]
            p = p[:T]
            input_ids[i, :len(m)] = m
            piece_input_ids[i, :len(p)] = p

        attention_mask = (input_ids != self.pad_id)

        # Next-token labels
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.pad_id
        labels[~attention_mask] = -100

        piece_labels = piece_input_ids.clone()
        piece_labels[:, :-1] = piece_input_ids[:, 1:]
        piece_labels[:, -1] = self.p_pad_id
        piece_labels[~attention_mask] = -100

        return {
            "valid": True,
            "input_ids": input_ids,
            "piece_input_ids": piece_input_ids,
            "labels": labels,
            "piece_labels": piece_labels,
            "attention_mask": attention_mask,
            "start_turn": start_turn,
        }