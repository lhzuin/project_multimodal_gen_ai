import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from models.chessformer import ChessFormer
from models.chess_visual_classifier import ChessVisualClassifierTransformers


class Policy64x64Head(nn.Module):
    """
    Chessformer-style policy head:
      h: [B,64,D] -> policy_logits [B,64,64]
    Implemented as q@k^T / sqrt(d).
    """
    def __init__(self, model_dim=256):
        super().__init__()
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, h):
        q = self.q(h)  # [B,64,D]
        k = self.k(h)  # [B,64,D]
        scale = 1.0 / math.sqrt(q.size(-1))
        return torch.matmul(q, k.transpose(1, 2)) * scale  # [B,64,64]


class ChessEncoderPlayer(nn.Module):
    """
    policy_64x64 player.
    Can be called with:
      - images=[B,3,H,W] OR
      - piece_probs=[B,64,13] OR piece_logits=[B,64,13]
    Always requires metadata: turn [B], castling [B,4], ep_square [B]
    """
    def __init__(
        self,
        img_size: int,
        encoder_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        encoder_dropout: float = 0.1,
        ep_embed_dim: int = 16,
        vit_path: str | None = None,
        freeze_vit: bool = True,
        cache_param_groups: bool = True,
    ):
        super().__init__()
        self._param_groups_cache = None if cache_param_groups else {}

        # Visual model (optional)
        self.vit = ChessVisualClassifierTransformers(img_size=img_size)
        self.has_vit = True

        if vit_path is not None:
            state = torch.load(vit_path, map_location="cpu")
            self.vit.load_state_dict(state)

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Metadata embedding
        self.ep_embed = nn.Embedding(65, ep_embed_dim)  # ep_square -1..63 mapped to 64

        # token features: piece_probs(13) + turn(1) + castling(4) + ep_emb(ep_dim)
        self.token_dim = 13 + 1 + 4 + ep_embed_dim

        # Chessformer encoder
        self.encoder = ChessFormer(
            token_dim=self.token_dim,
            model_dim=encoder_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_rate=encoder_dropout,
        )

        # Policy head
        self.policy_head = Policy64x64Head(model_dim=encoder_dim)

        # Promotions: separate head (recommended)
        self.promo_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 5),  # 0 none, 1 N,2 B,3 R,4 Q
        )

    # --------------------
    # Token building
    # --------------------
    def _piece_probs_from_inputs(self, *, images=None, piece_probs=None, piece_logits=None):
        if piece_probs is not None:
            return piece_probs
        if piece_logits is not None:
            return F.softmax(piece_logits, dim=-1)
        if images is None:
            raise ValueError("Provide either images, piece_probs or piece_logits.")

        logits = self.vit(images)  # [B,64,13]
        return F.softmax(logits, dim=-1)

    def _tokens(self, piece_probs, turn, castling, ep_square):
        B = piece_probs.size(0)

        turn_f = turn.float().view(B, 1, 1).expand(B, 64, 1)           # [B,64,1]
        cast_f = castling.float().view(B, 1, 4).expand(B, 64, 4)       # [B,64,4]

        ep_idx = ep_square.clone()
        ep_idx = torch.where(ep_idx < 0, torch.full_like(ep_idx, 64), ep_idx)
        ep_emb = self.ep_embed(ep_idx)                                 # [B,ep_dim]
        ep_emb = ep_emb.view(B, 1, -1).expand(B, 64, ep_emb.size(-1))  # [B,64,ep_dim]

        x_tokens = torch.cat([piece_probs, turn_f, cast_f, ep_emb], dim=-1)  # [B,64,F]
        return x_tokens

    # --------------------
    # Forward
    # --------------------
    def forward(self, *, images=None, piece_probs=None, piece_logits=None, turn=None, castling=None, ep_square=None): 
        #Obs: white color=1 (different from decoder's turn encoding)
        if turn is None or castling is None or ep_square is None:
            raise ValueError("turn, castling, ep_square are required.")

        piece_probs = self._piece_probs_from_inputs(images=images, piece_probs=piece_probs, piece_logits=piece_logits)
        x_tokens = self._tokens(piece_probs, turn, castling, ep_square)     # [B,64,F]
        h = self.encoder(x_tokens)                                          # [B,64,D]
        policy_logits = self.policy_head(h)                                 # [B,64,64]

        pooled = h.mean(dim=1)                                              # [B,D]
        promo_logits = self.promo_head(pooled)                              # [B,5]

        return {"policy_logits": policy_logits, "promo_logits": promo_logits}

    # --------------------
    # Legal mask + sampling (built in)
    # --------------------
    @staticmethod
    def legal_mask_from_fen(fen_list: list[str], device=None):
        B = len(fen_list)
        mask = torch.zeros((B, 64, 64), dtype=torch.bool, device=device)
        for i, fen in enumerate(fen_list):
            board = chess.Board(fen)
            for mv in board.legal_moves:
                mask[i, mv.from_square, mv.to_square] = True
        return mask

    @staticmethod
    def apply_legal_mask(policy_logits, legal_mask):
        neg_inf = torch.finfo(policy_logits.dtype).min
        return policy_logits.masked_fill(~legal_mask, neg_inf)

    @torch.no_grad()
    def sample_moves(
        self,
        *,
        images=None,
        piece_probs=None,
        piece_logits=None,
        turn=None,
        castling=None,
        ep_square=None,
        fen=None,                     # list[str] length B, required for masking
        temperature: float = 1.0,
        topk: int | None = None,
        greedy: bool = False,
    ):
        """
        Returns a list[chess.Move] (length B).
        If fen is provided, masks illegal moves before sampling.
        Promotion selection uses promo_logits when multiple legal moves share (from,to).
        """
        out = self.forward(
            images=images,
            piece_probs=piece_probs,
            piece_logits=piece_logits,
            turn=turn,
            castling=castling,
            ep_square=ep_square,
        )
        policy_logits = out["policy_logits"]  # [B,64,64]
        promo_logits = out["promo_logits"]    # [B,5]
        B = policy_logits.size(0)

        if fen is None:
            raise ValueError("fen list is required for legal masking + sampling.")

        legal_mask = self.legal_mask_from_fen(fen, device=policy_logits.device)
        masked = self.apply_legal_mask(policy_logits, legal_mask)  # [B,64,64]

        flat = masked.view(B, -1)  # [B,4096]
        flat = flat / max(1e-6, float(temperature))

        moves_out = []
        for i in range(B):
            board = chess.Board(fen[i])

            if greedy:
                idx = int(flat[i].argmax().item())
            else:
                if topk is not None:
                    vals, idxs = torch.topk(flat[i], k=topk)
                    probs = F.softmax(vals, dim=-1)
                    pick = int(torch.multinomial(probs, 1).item())
                    idx = int(idxs[pick].item())
                else:
                    probs = F.softmax(flat[i], dim=-1)
                    idx = int(torch.multinomial(probs, 1).item())

            from_sq = idx // 64
            to_sq = idx % 64

            # match legal moves (promotion ambiguity)
            candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
            if not candidates:
                # fallback: choose best legal move
                best = int(flat[i].argmax().item())
                from_sq = best // 64
                to_sq = best % 64
                candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]

            if len(candidates) == 1:
                moves_out.append(candidates[0])
                continue

            # choose promotion with promo_logits if needed
            # promo id: 0 none, 1 N,2 B,3 R,4 Q
            promo_probs = F.softmax(promo_logits[i], dim=-1)
            promo_id = int(torch.argmax(promo_probs).item())

            promo_map = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
            desired = promo_map.get(promo_id, None)

            chosen = None
            for mv in candidates:
                if mv.promotion == desired:
                    chosen = mv
                    break
            if chosen is None:
                chosen = candidates[0]
            moves_out.append(chosen)

        return moves_out

    # --------------------
    # Param groups (keep your pattern)
    # --------------------
    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = self._build_param_groups()
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        groups = self.get_param_groups()
        for p in self.parameters():
            p.requires_grad = False
        for g in group_names:
            assert g in groups, f"Unknown param group: {g}"
            for p in groups[g]:
                p.requires_grad = True

    def _build_param_groups(self):
        groups = {
            "encoder": list(self.encoder.parameters()),
            "classifier": list(self.policy_head.parameters()) + list(self.promo_head.parameters()),
        }
        # only include vit if it’s trainable
        if any(p.requires_grad for p in self.vit.parameters()):
            groups["vit"] = list(self.vit.parameters())
        return groups







class ChessEncoderPlayerV2(nn.Module):
    """
    Efficient policy_64x64 player.

    Supports inputs:
      - labels64: LongTensor [B,64]   (fast path, recommended)
      - piece_probs: FloatTensor [B,64,13]
      - piece_logits: FloatTensor [B,64,13]
      - images: FloatTensor [B,3,H,W]

    Metadata required:
      - turn: LongTensor [B]
      - castling: LongTensor [B,4]
      - ep_square: LongTensor [B]
    """
    def __init__(
        self,
        img_size: int,
        encoder_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        encoder_dropout: float = 0.1,
        ep_embed_dim: int = 16,
        piece_embed_dim: int = 32,   # NEW: compact piece representation
        vit_path: str | None = None,
        freeze_vit: bool = True,
        cache_param_groups: bool = True,
    ):
        super().__init__()
        self._param_groups_cache = None if cache_param_groups else {}

        # Visual model (optional)
        self.vit = ChessVisualClassifierTransformers(img_size=img_size)
        self.has_vit = True

        if vit_path is not None:
            state = torch.load(vit_path, map_location="cpu")
            self.vit.load_state_dict(state)

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Metadata embedding
        self.ep_embed = nn.Embedding(65, ep_embed_dim)  # ep_square -1..63 mapped to 64

        # NEW: piece feature path
        # - If labels64 is provided: embedding lookup (fast)
        # - If probs/logits/images are provided: linear projection to same dim
        self.piece_embed = nn.Embedding(13, piece_embed_dim)
        self.piece_proj = nn.Linear(13, piece_embed_dim, bias=False)

        # token features: piece_feat(piece_embed_dim) + turn(1) + castling(4) + ep_emb(ep_dim)
        self.token_dim = piece_embed_dim + 1 + 4 + ep_embed_dim

        # Chessformer encoder
        self.encoder = ChessFormer(
            token_dim=self.token_dim,
            model_dim=encoder_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_rate=encoder_dropout,
        )

        # Policy head
        self.policy_head = Policy64x64Head(model_dim=encoder_dim)

        # Promotions
        self.promo_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 5),  # 0 none, 1 N,2 B,3 R,4 Q
        )

    # --------------------
    # Token building
    # --------------------
    def _piece_feat_from_inputs(self, *, labels64=None, images=None, piece_probs=None, piece_logits=None):
        # Fast path: labels64 -> embedding
        if labels64 is not None:
            # labels64: [B,64] long in [0..12]
            return self.piece_embed(labels64)  # [B,64,piece_embed_dim]

        # Otherwise, use probabilities/logits/images -> project to piece_embed_dim
        if piece_probs is not None:
            probs = piece_probs
        elif piece_logits is not None:
            probs = F.softmax(piece_logits, dim=-1)
        elif images is not None:
            logits = self.vit(images)  # [B,64,13]
            probs = F.softmax(logits, dim=-1)
        else:
            raise ValueError("Provide labels64 OR one of images/piece_probs/piece_logits.")

        return self.piece_proj(probs)  # [B,64,piece_embed_dim]

    def _tokens(self, piece_feat, turn, castling, ep_square):
        B = piece_feat.size(0)

        turn_f = turn.float().view(B, 1, 1).expand(B, 64, 1)           # [B,64,1]
        cast_f = castling.float().view(B, 1, 4).expand(B, 64, 4)       # [B,64,4]

        ep_idx = ep_square.clone()
        ep_idx = torch.where(ep_idx < 0, torch.full_like(ep_idx, 64), ep_idx)
        ep_emb = self.ep_embed(ep_idx)                                 # [B,ep_dim]
        ep_emb = ep_emb.view(B, 1, -1).expand(B, 64, ep_emb.size(-1))  # [B,64,ep_dim]

        x_tokens = torch.cat([piece_feat, turn_f, cast_f, ep_emb], dim=-1)  # [B,64,F]
        return x_tokens

    # --------------------
    # Forward
    # --------------------
    def forward(
        self,
        *,
        labels64=None,            # NEW recommended input: [B,64] long
        images=None,
        piece_probs=None,
        piece_logits=None,
        turn=None,
        castling=None,
        ep_square=None,
    ):
        # Obs: white color=1 (different from decoder's turn encoding)
        if turn is None or castling is None or ep_square is None:
            raise ValueError("turn, castling, ep_square are required.")

        piece_feat = self._piece_feat_from_inputs(
            labels64=labels64,
            images=images,
            piece_probs=piece_probs,
            piece_logits=piece_logits,
        )  # [B,64,piece_embed_dim]

        x_tokens = self._tokens(piece_feat, turn, castling, ep_square)  # [B,64,token_dim]
        h = self.encoder(x_tokens)                                      # [B,64,D]

        policy_logits = self.policy_head(h)                             # [B,64,64]
        pooled = h.mean(dim=1)                                          # [B,D]
        promo_logits = self.promo_head(pooled)                          # [B,5]

        return {"policy_logits": policy_logits, "promo_logits": promo_logits}

    # --------------------
    # Legal mask + sampling (still useful for inference)
    # --------------------
    @staticmethod
    def legal_mask_from_fen(fen_list: list[str], device=None):
        B = len(fen_list)
        mask = torch.zeros((B, 64, 64), dtype=torch.bool, device=device)
        for i, fen in enumerate(fen_list):
            board = chess.Board(fen)
            for mv in board.legal_moves:
                mask[i, mv.from_square, mv.to_square] = True
        return mask

    @staticmethod
    def apply_legal_mask(policy_logits, legal_mask):
        neg_inf = torch.finfo(policy_logits.dtype).min
        return policy_logits.masked_fill(~legal_mask, neg_inf)

    @torch.no_grad()
    def sample_moves(
        self,
        *,
        images=None,
        piece_probs=None,
        piece_logits=None,
        turn=None,
        castling=None,
        ep_square=None,
        fen=None,                     # list[str] length B, required for masking
        temperature: float = 1.0,
        topk: int | None = None,
        greedy: bool = False,
    ):
        """
        Returns a list[chess.Move] (length B).
        If fen is provided, masks illegal moves before sampling.
        Promotion selection uses promo_logits when multiple legal moves share (from,to).
        """
        out = self.forward(
            images=images,
            piece_probs=piece_probs,
            piece_logits=piece_logits,
            turn=turn,
            castling=castling,
            ep_square=ep_square,
        )
        policy_logits = out["policy_logits"]  # [B,64,64]
        promo_logits = out["promo_logits"]    # [B,5]
        B = policy_logits.size(0)

        if fen is None:
            raise ValueError("fen list is required for legal masking + sampling.")

        legal_mask = self.legal_mask_from_fen(fen, device=policy_logits.device)
        masked = self.apply_legal_mask(policy_logits, legal_mask)  # [B,64,64]

        flat = masked.view(B, -1)  # [B,4096]
        flat = flat / max(1e-6, float(temperature))

        moves_out = []
        for i in range(B):
            board = chess.Board(fen[i])

            if greedy:
                idx = int(flat[i].argmax().item())
            else:
                if topk is not None:
                    vals, idxs = torch.topk(flat[i], k=topk)
                    probs = F.softmax(vals, dim=-1)
                    pick = int(torch.multinomial(probs, 1).item())
                    idx = int(idxs[pick].item())
                else:
                    probs = F.softmax(flat[i], dim=-1)
                    idx = int(torch.multinomial(probs, 1).item())

            from_sq = idx // 64
            to_sq = idx % 64

            # match legal moves (promotion ambiguity)
            candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
            if not candidates:
                # fallback: choose best legal move
                best = int(flat[i].argmax().item())
                from_sq = best // 64
                to_sq = best % 64
                candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]

            if len(candidates) == 1:
                moves_out.append(candidates[0])
                continue

            # choose promotion with promo_logits if needed
            # promo id: 0 none, 1 N,2 B,3 R,4 Q
            promo_probs = F.softmax(promo_logits[i], dim=-1)
            promo_id = int(torch.argmax(promo_probs).item())

            promo_map = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
            desired = promo_map.get(promo_id, None)

            chosen = None
            for mv in candidates:
                if mv.promotion == desired:
                    chosen = mv
                    break
            if chosen is None:
                chosen = candidates[0]
            moves_out.append(chosen)

        return moves_out

    # --------------------
    # Param groups (keep your pattern)
    # --------------------
    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = self._build_param_groups()
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        groups = self.get_param_groups()
        for p in self.parameters():
            p.requires_grad = False
        for g in group_names:
            assert g in groups, f"Unknown param group: {g}"
            for p in groups[g]:
                p.requires_grad = True

    def _build_param_groups(self):
        groups = {
            "encoder": list(self.encoder.parameters()),
            "classifier": list(self.policy_head.parameters()) + list(self.promo_head.parameters()),
        }
        # only include vit if it’s trainable
        if any(p.requires_grad for p in self.vit.parameters()):
            groups["vit"] = list(self.vit.parameters())
        return groups