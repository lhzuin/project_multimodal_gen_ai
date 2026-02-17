
import torch
import torch.nn as nn
from models.chess_tokenizer import ChessTokenizer

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
    def forward(self, x):
        # x = [batch size, seq len, hidden dim]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)  # [seq_len]
        angle_rates = 1 / torch.pow(10000, (2 * (torch.arange(self.hidden_dim, device=x.device) // 2)) / self.hidden_dim)  # [hidden_dim]
        angle_rads = positions.unsqueeze(1) * angle_rates.unsqueeze(0)  # [seq_len, hidden_dim]
        pos_encoding = torch.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pos_encoding = pos_encoding.unsqueeze(0)  # [1, seq_len, hidden_dim]
        x = x + pos_encoding
        return x
    

class ChessLLM(nn.Module):
    def __init__(self, tokenizer_path, data_dim, model_dim=256, mlp_ratio=2, n_heads=4, n_layers=6, dropout_rate=0.1):
        super().__init__()

        self.embedding = nn.Embedding(data_dim, model_dim) if data_dim != model_dim else nn.Identity()
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=n_heads,
                dim_feedforward=int(mlp_ratio * model_dim),
                dropout=dropout_rate,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.positional_encoder = SinusoidalPositionalEncoding(data_dim)


        self.tokenizer = ChessTokenizer(tokenizer_path)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = self.embedding(x)

        for layer in self.decoder_layers:
            x = layer(x)
        return x
        


