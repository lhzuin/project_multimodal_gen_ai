import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    """
    ViT-style image tokenizer:
      x: (B, C, H, W)
      -> tokens: (B, N, D) where N = (H/P)*(W/P)
    """
    def __init__(self, img_size=256, patch_size=32, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size  # N

        # This conv is exactly equivalent to:
        # 1) extracting non-overlapping patches
        # 2) flattening them
        # 3) applying a linear projection
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self._init_params()



    def _init_params(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Conv init is fine by default, but you can also init explicitly if desired.

    def forward(self, x):
        """
        x: (B, C, H, W) must match img_size
        returns: (B, N , D)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input size must match img_size"

        # (B, D, H/P, W/P)
        x = self.proj(x)

        # (B, D, N)
        x = x.flatten(2)

        # (B, N, D)
        x = x.transpose(1, 2)

        x = x + self.pos_embed                          # add position info
        return x


class PatchToSquareTokens(nn.Module):
    """
    Convert patch tokens (B, N, D) laid out as (Gh, Gw) into exactly 8x8=64 square tokens.

    Works when patch grid is square-ish. For img_size=256:
      patch_size=16 -> grid 16x16 -> pool 2x2 -> 8x8
      patch_size=32 -> grid 8x8   -> pool 1x1 -> 8x8 (identity)
    """
    def __init__(self, img_size=256, patch_size=32, board_size=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.board_size = board_size
        self.grid = img_size // patch_size
        assert self.grid % board_size == 0, "patch grid must be divisible by board_size"
        self.pool = self.grid // board_size  # e.g. 16/8=2

    def forward(self, x):
        # x: [B, N, D] where N = grid*grid
        B, N, D = x.shape
        g = self.grid
        assert N == g * g, "Unexpected N for given img_size/patch_size"
        x = x.view(B, g, g, D)

        p = self.pool
        if p == 1:
            x = x
        else:
            # mean pool in non-overlapping p x p blocks
            x = x.view(B, self.board_size, p, self.board_size, p, D).mean(dim=(2,4))

        # x: [B, 8, 8, D] -> [B, 64, D]
        x = x.reshape(B, self.board_size * self.board_size, D)
        return x


class ChessVit(nn.Module):
    def __init__(self, img_size,  patch_size = 32, model_dim=256, mlp_ratio = 2, n_heads=4, n_layers=6, dropout_rate=0.1, in_channels=3):
        super().__init__()
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=int(mlp_ratio * model_dim),
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,   # strongly recommended for training from scratch stability
        )
            for _ in range(n_layers)
        ])
        self.tokenizer = PatchEmbeddings(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=model_dim)
        self.to_square_tokens = PatchToSquareTokens(img_size=img_size, patch_size=patch_size, board_size=8)

    
    def forward(self, x):
        x = self.tokenizer(x)          # [B, Npatch, D]
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.to_square_tokens(x)   # [B, 64, D]
        return x
    






# class SSLHead(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)