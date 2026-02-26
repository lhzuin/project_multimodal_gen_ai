# chess_decoder_player.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from models.chess_llm import ChessLLM


class ChessDecoderPlayer(nn.Module):
    def __init__(
        self,
        tokenizer_path: str,
        model_dim: int = 256,
        mlp_ratio: float = 4.0,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        tie_weights: bool = True,
        use_turn_embed: bool = True,
        cache_param_groups: bool = True,
    ):
        super().__init__()
        self._param_groups_cache = None if cache_param_groups else {}

        self.decoder = ChessLLM(
            tokenizer_path=tokenizer_path,
            model_dim=model_dim,
            mlp_ratio=mlp_ratio,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_rate=dropout,
            max_seq_len=max_seq_len,
        )

        tok = self.decoder.tokenizer
        self.vocab_size = self.decoder.vocab_size
        self.pad_id = tok.pad_id
        self.bos_id = tok.bos_id
        self.eos_id = tok.eos_id

        # Move LM head
        self.lm_head = nn.Linear(model_dim, self.vocab_size, bias=False)

        self.tie_weights = tie_weights
        if tie_weights:
            self.lm_head.weight = self.decoder.tok_emb.weight

        # Piece head (aux task)
        self.piece_vocab_size = len(tok.id2piece)
        self.piece_head = nn.Linear(model_dim, self.piece_vocab_size)

        # Turn embedding to handle random crops
        self.use_turn_embed = use_turn_embed
        if use_turn_embed:
            self.turn_emb = nn.Embedding(2, model_dim)  # 0 white, 1 black

    def forward(self, input_ids, attention_mask=None, start_turn=None, piece_input_ids=None):
        """
        input_ids: [B,T]
        start_turn: [B] 0/1 (optional)
        returns logits: [B,T,V]
        """
        h = self.decoder(input_ids=input_ids, attention_mask=attention_mask, piece_input_ids=piece_input_ids)  # [B,T,D]

        if self.use_turn_embed and start_turn is not None:
            B, T, D = h.shape
            t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
            parity = (start_turn.view(B, 1) + t) % 2
            h = h + self.turn_emb(parity)

        move_logits = self.lm_head(h)            # [B,T,Vmove]
        piece_logits = self.piece_head(h)        # [B,T,Vpiece]
        return {"logits": move_logits, "piece_logits": piece_logits}

    # ---------- Legal mask over vocab ----------
    def legal_token_mask_from_fen(self, fen_list, device=None):
        """
        returns [B,V] bool mask for move tokens only (UCI strings in tokenizer).
        """
        tok = self.decoder.tokenizer
        B = len(fen_list)
        mask = torch.zeros((B, self.vocab_size), dtype=torch.bool, device=device)
        for i, fen in enumerate(fen_list):
            board = chess.Board(fen)
            for mv in board.legal_moves:
                uci = mv.uci()
                tid = tok.move2id.get(uci, None)
                if tid is not None:
                    mask[i, tid] = True
        # keep specials illegal (optional)
        mask[:, tok.pad_id] = False
        mask[:, tok.bos_id] = False
        mask[:, tok.eos_id] = False
        mask[:, tok.unk_id] = False
        return mask

    @staticmethod
    def apply_token_mask(logits, mask):
        # logits: [B,V], mask: [B,V]
        neg = -1e9
        if logits.dtype in (torch.float16, torch.bfloat16):
            neg = -1e4
        return logits.masked_fill(~mask, neg)

    # ---------- Policy-like next-move sampling ----------
    @torch.no_grad()
    def sample_moves(
        self,
        input_ids,                 # [B,T] history tokens (already encoded)
        fen_list,                  # list[str] current position per batch
        attention_mask=None,        # [B,T]
        start_turn=None,            # [B] optional
        piece_input_ids=None,
        temperature: float = 1.0,
        topk: int | None = None,
        greedy: bool = False,
    ):
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask, start_turn=start_turn, piece_input_ids=piece_input_ids)
        next_logits = out["logits"][:, -1, :]  # [B,V]
        next_logits = next_logits / max(1e-6, float(temperature))

        legal = self.legal_token_mask_from_fen(fen_list, device=next_logits.device)
        next_logits = self.apply_token_mask(next_logits, legal)

        moves = []
        tok = self.decoder.tokenizer
        for i in range(next_logits.size(0)):
            if greedy:
                tid = int(next_logits[i].argmax().item())
            else:
                if topk is not None:
                    vals, idxs = torch.topk(next_logits[i], k=topk)
                    probs = F.softmax(vals, dim=-1)
                    pick = int(torch.multinomial(probs, 1).item())
                    tid = int(idxs[pick].item())
                else:
                    probs = F.softmax(next_logits[i], dim=-1)
                    tid = int(torch.multinomial(probs, 1).item())

            uci = tok.id2move[tid]
            moves.append(chess.Move.from_uci(uci))

        return moves
    
    @torch.no_grad()
    def next_policy_matrix(self, input_ids, fen_list, attention_mask=None, start_turn=None, piece_input_ids=None):
        """
        Returns:
        policy_logits_64: [B,64,64] where entry is logit for move from->to
        (promotion tokens mapped by taking max over promotions)
        """
        out = self.forward(
            input_ids=input_ids,
            piece_input_ids=piece_input_ids,
            attention_mask=attention_mask,
            start_turn=start_turn,
        )
        next_logits = out["logits"][:, -1, :]  # [B,V]

        # legal token mask
        legal = self.legal_token_mask_from_fen(fen_list, device=next_logits.device)
        next_logits = self.apply_token_mask(next_logits, legal)

        B, V = next_logits.shape
        mat = next_logits.new_full((B, 64, 64), float("-inf"))

        tok = self.decoder.tokenizer
        for tid, uci in enumerate(tok.id2move):
            # skip specials and malformed
            if len(uci) < 4 or uci[0] == "<":
                continue
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            f = mv.from_square
            t = mv.to_square
            mat[:, f, t] = torch.maximum(mat[:, f, t], next_logits[:, tid])

        return mat

    # ---------- Keep your param-group pattern ----------
    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = {
                "decoder": list(self.decoder.parameters()),
                "lm_head": list(self.lm_head.parameters()),
                "piece_head": list(self.piece_head.parameters()),
                **({"turn_emb": list(self.turn_emb.parameters())} if self.use_turn_embed else {}),
            }
            # If weights are tied, lm_head.weight is decoder.tok_emb.weight
            # Remove that shared Parameter from lm_head group
            if getattr(self, "tie_weights", False):
                tied = self.decoder.tok_emb.weight
                self._param_groups_cache["lm_head"] = [p for p in self._param_groups_cache["lm_head"] if p is not tied]
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        groups = self.get_param_groups()
        for p in self.parameters():
            p.requires_grad = False
        for g in group_names:
            for p in groups[g]:
                p.requires_grad = True
