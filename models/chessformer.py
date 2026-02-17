import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_relpos_index_2d(seq_len=64, board_w=8, board_h=8, max_rel=7, device=None):
    assert seq_len == board_w * board_h
    # token i -> (x,y) with your chosen scan order; this is row-major by default:
    xs = torch.arange(seq_len, device=device) % board_w
    ys = torch.arange(seq_len, device=device) // board_w

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]

    dx = dx.clamp(-max_rel, max_rel) + max_rel
    dy = dy.clamp(-max_rel, max_rel) + max_rel

    # map (dx,dy) -> single id in [0, (2*max_rel+1)^2 - 1]
    n_bucket = 2 * max_rel + 1
    rel_index = dx * n_bucket + dy
    return rel_index  # [seq_len, seq_len]

class MultiHeadRelativeAttention(nn.Module):
    def __init__(self, x_to_dim, x_from_dim, hidden_dim, n_heads, seq_len=64, board_w=8, board_h=8, max_rel=7):
        super().__init__()
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.x_to_dim = x_to_dim
        self.x_from_dim = x_from_dim
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads

        self.Wq = nn.Linear(x_to_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(x_from_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(x_from_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, x_to_dim)

        # ---- Shaw et al. relative position representations (Eq. 6-7) ----
        self.seq_len = seq_len
        self.board_w = board_w
        self.board_h = board_h
        self.max_rel = max_rel
        self.n_bucket = 2 * max_rel + 1
        self.n_rel = self.n_bucket * self.n_bucket  # 225 for max_rel=7

        # Per-head learnable vectors a^Q_ij, a^K_ij, a^V_ij (stored as embeddings over relative buckets)
        self.rel_aq = nn.Embedding(self.n_rel, self.head_dim)
        self.rel_ak = nn.Embedding(self.n_rel, self.head_dim)
        self.rel_av = nn.Embedding(self.n_rel, self.head_dim)

        # Precompute (i,j)->bucket index as a buffer
        rel_index = build_relpos_index_2d(seq_len, board_w, board_h, max_rel, device=None)
        self.register_buffer("rel_index", rel_index, persistent=False)

    def forward(self, x_to, x_from=None, attn_mask=None):
        # x_to = [batch size, x_to_len, x_to_dim]
        # x_from = [batch size, x_from_len, x_from_dim]
        if x_from is None:
            x_from = x_to

        Q = self.Wq(x_to)     # [b, t, hidden_dim]
        K = self.Wk(x_from)   # [b, f, hidden_dim]
        V = self.Wv(x_from)   # [b, f, hidden_dim]

        batch_size, x_to_len, _ = Q.size()
        _, x_from_len, _ = K.size()

        Q = Q.view(batch_size, x_to_len, self.n_heads, self.head_dim).transpose(1, 2)  # [b,h,t,d]
        K = K.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1, 2)  # [b,h,f,d]
        V = V.view(batch_size, x_from_len, self.n_heads, self.head_dim).transpose(1, 2)  # [b,h,f,d]

        scale = math.sqrt(self.head_dim)

        # ---- build aQ_ij, aK_ij, aV_ij for all pairs (t,f) ----
        # expects fixed 64 tokens for chessboard; if you ever vary length, you must rebuild rel_index
        assert x_to_len == self.seq_len and x_from_len == self.seq_len, "Shaw 2D relpos here assumes fixed seq_len=64"

        rel_ids = self.rel_index[:x_to_len, :x_from_len]  # [t,f]
        aQ = self.rel_aq(rel_ids).unsqueeze(0)  # [1,t,f,d]
        aK = self.rel_ak(rel_ids).unsqueeze(0)  # [1,t,f,d]
        aV = self.rel_av(rel_ids).unsqueeze(0)  # [1,t,f,d]

        # Expand over heads by sharing the same a* across heads (simple + common).
        # If you want per-head a*, make rel_aq/rel_ak/rel_av have (n_heads*n_rel) and reshape.
        aQ = aQ.unsqueeze(1)  # [1,1,t,f,d]
        aK = aK.unsqueeze(1)  # [1,1,t,f,d]
        aV = aV.unsqueeze(1)  # [1,1,t,f,d]

        # ---- Shaw logits (Eq. 6): (Q + aQ_ij) dot (K + aK_ij) ----
        Qp = Q.unsqueeze(3) + aQ         # [b,h,t,1,d] + [1,1,t,f,d] -> [b,h,t,f,d]
        Kp = K.unsqueeze(2) + aK         # [b,h,1,f,d] + [1,1,t,f,d] -> [b,h,t,f,d]
        similarity = (Qp * Kp).sum(dim=-1) / scale  # [b,h,t,f]

        if attn_mask is not None:
            # Allow [t,f], [b,t,f], or [b,1,t,f]
            attn_mask = attn_mask.to(dtype=torch.bool, device=similarity.device)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,t,f]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [b,1,t,f]
            similarity = similarity.masked_fill(attn_mask, float('-inf'))

        attention_coefficients = F.softmax(similarity, dim=-1)  # [b,h,t,f]

        # ---- Shaw values (Eq. 7): sum_j α_ij (V_j + aV_ij) ----
        Vp = V.unsqueeze(2) + aV  # [b,h,1,f,d] + [1,1,t,f,d] -> [b,h,t,f,d]
        head_output = torch.einsum('bhtf,bhtfd->bhtd', attention_coefficients, Vp)  # [b,h,t,d]

        out = head_output.transpose(1, 2).contiguous().view(batch_size, x_to_len, self.hidden_dim)  # [b,t,hidden_dim]
        return self.Wo(out)
    
class LearnedAffinePositionalEncoder(nn.Module):
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.pos_add = nn.Parameter(torch.zeros(seq_len, model_dim))
        self.pos_mul = nn.Parameter(torch.ones(seq_len, model_dim))  # start as identity

    def forward(self, x):
        # x: [b, seq_len, model_dim]
        return x * self.pos_mul.unsqueeze(0) + self.pos_add.unsqueeze(0)

class FFN(nn.Sequential):
    def __init__(self, model_dim, dropout_rate=0.1, expansion_factor=2):
        super(FFN, self).__init__()
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.expansion_factor = expansion_factor
        self.fc1 = nn.Linear(model_dim, model_dim * expansion_factor)
        self.fc2 = nn.Linear(model_dim * expansion_factor, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, hidden_dim, mlp_ratio, n_heads, dropout_rate=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.self_attention = MultiHeadRelativeAttention(model_dim, model_dim, hidden_dim, n_heads)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.ffn = FFN(model_dim=model_dim, dropout_rate=dropout_rate, expansion_factor=mlp_ratio)
        self.layer_norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, x):
        # x = [batch size, x_len, hidden dim]
        attended_x = self.self_attention(x)
        x = self.layer_norm1(x + attended_x)
        ffn_x = self.ffn(x)
        x = self.layer_norm2(x + ffn_x)
        return x

class ChessFormer(nn.Module):
    def __init__(self, data_dim, model_dim=256, mlp_ratio = 2, n_heads=4, n_layers=6, dropout_rate=0.1):
        super().__init__()

        self.embedding = nn.Embedding(data_dim, model_dim) if data_dim != model_dim else nn.Identity()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
            model_dim=model_dim,
            hidden_dim=model_dim,   # keep same unless you explicitly want different
            mlp_ratio=mlp_ratio,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            #batch_first=True,
            #norm_first=True,   # strongly recommended for training from scratch stability
        )
            for _ in range(n_layers)
        ])

        self.positional_encoder = LearnedAffinePositionalEncoder(seq_len=64, model_dim=model_dim)

    def forward(self, x):
        x = self.embedding(x)              # [b,64] -> [b,64,model_dim] (if embedding)
        x = self.positional_encoder(x)     # learned add+mul per position

        for layer in self.encoder_layers:
            x = layer(x)
        return x