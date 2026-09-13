# collate_llm.py
import torch


class CollateLLMMoves:
    """Pad decoder sequences and build next-token labels.

    If dataset items contain ``loss_mask``, it must already be aligned with decoder
    label positions (same length as ``move_ids``).  Zero entries are converted to
    ``ignore_index=-100`` *after* next-token shifting; input and attention tensors are
    left untouched so masked moves remain available as causal context.

    Items without ``loss_mask`` follow the legacy behavior byte-for-byte, including
    the historical final PAD next-token target.  This preserves old config behavior.
    """

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

        has_loss_mask = ["loss_mask" in b for b in batch]
        if any(has_loss_mask) and not all(has_loss_mask):
            raise ValueError("cannot collate a mixture of planned and legacy decoder items")
        use_loss_mask = all(has_loss_mask)

        lengths = [len(x) for x in move_seqs]
        T = max(lengths)
        if self.max_len is not None:
            T = min(T, int(self.max_len))

        B = len(move_seqs)

        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        piece_input_ids = torch.full((B, T), self.p_pad_id, dtype=torch.long)
        supervision_mask = torch.zeros((B, T), dtype=torch.bool) if use_loss_mask else None

        for i, (m, p) in enumerate(zip(move_seqs, piece_seqs)):
            if use_loss_mask and len(m) > T:
                # A plan was generated for an explicit context limit.  Silently
                # truncating it would break absolute-ply/loss alignment, so fail fast.
                raise ValueError(
                    f"planned decoder item has {len(m)} tokens but collate max_len={T}"
                )

            m = m[:T]
            p = p[:T]
            input_ids[i, :len(m)] = m
            piece_input_ids[i, :len(p)] = p

            if supervision_mask is not None:
                item_mask = batch[i]["loss_mask"].bool()
                if item_mask.ndim != 1 or item_mask.numel() != len(m):
                    raise ValueError(
                        "decoder loss_mask must be one-dimensional and match move_ids; "
                        f"got mask={tuple(item_mask.shape)}, tokens={len(m)}"
                    )
                supervision_mask[i, :len(m)] = item_mask

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

        if supervision_mask is not None:
            # Crucially, only labels are suppressed.  input_ids, piece_input_ids and
            # attention_mask stay unchanged, preserving the full planned causal context.
            labels[~supervision_mask] = -100
            piece_labels[~supervision_mask] = -100

        out = {
            "valid": True,
            "input_ids": input_ids,
            "piece_input_ids": piece_input_ids,
            "labels": labels,
            "piece_labels": piece_labels,
            "attention_mask": attention_mask,
            "start_turn": start_turn,
        }
        if supervision_mask is not None:
            out["loss_mask"] = supervision_mask
        return out
