# models/chess_llm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.chess_tokenizer import ChessTokenizer


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)  # [max_len, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.dtype)


def causal_mask(T: int, device):
    # True where we should mask (upper triangle)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class ChessLLM(nn.Module):
    """
    Decoder-only Transformer for UCI-move tokens.
    Forward returns hidden states; caller adds lm_head.
    """
    def __init__(
        self,
        tokenizer_path: str,
        model_dim: int = 256,
        mlp_ratio: float = 4.0,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout_rate: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.tokenizer = ChessTokenizer(tokenizer_path)
        self.vocab_size = len(self.tokenizer.id2move)

        self.tok_emb = nn.Embedding(self.vocab_size, model_dim)
        self.piece_vocab_size = len(self.tokenizer.id2piece)
        self.piece_emb = nn.Embedding(self.piece_vocab_size, model_dim)
        self.pos_enc = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_seq_len)
        self.drop = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * model_dim),
                dropout=dropout_rate,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, piece_input_ids: torch.Tensor | None =None):
        """
        input_ids: [B,T] long
        attention_mask: [B,T] bool/0-1 where 1/True means "keep"
        """
        x = self.tok_emb(input_ids)  # [B,T,D]

        if piece_input_ids is not None:
            x = x + self.piece_emb(piece_input_ids)
        x = self.pos_enc(x)
        x = self.drop(x)

        T = input_ids.size(1)
        attn_mask = causal_mask(T, device=input_ids.device)  # [T,T] bool

        # key_padding_mask: True means "mask out" (pad)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(torch.bool)  # [B,T]

        for layer in self.layers:
            x = layer(x, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)

        x = self.norm(x)
        return x