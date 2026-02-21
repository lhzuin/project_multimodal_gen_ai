import torch
import torch.nn as nn
from models.chess_vit import ChessVit

class PieceClassifierHead(nn.Module):
    num_classes = 13
    def __init__(self, encoder_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.activation_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        # x: [B, 64, D]
        x = self.activation_func(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 64, 13]
        return x



class ChessVisualClassifierTransformers(nn.Module):
    def __init__(self, img_size,  patch_size = 32, encoder_dim=256, n_heads=4, n_layers=6, encoder_dropout=0.1, head_hidden_dim=512, head_dropout=0.1):
        super().__init__()
        self._param_groups_cache = None
        
        self.classifier = PieceClassifierHead(encoder_dim=encoder_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)
        self.encoder = ChessVit(img_size=img_size, patch_size=patch_size, model_dim=encoder_dim, n_heads=n_heads, n_layers=n_layers, dropout_rate=encoder_dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    

    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = self._build_param_groups()
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        """
        Generic, config-driven freezing:
        - group_names: list of strings, subset of keys in get_param_groups().
        """
        groups = self.get_param_groups()

        # default: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # enable only requested groups
        for g in group_names:
            assert g in groups, f"Unknown param group: {g}"
            for p in groups[g]:
                p.requires_grad = True
    
    def _build_param_groups(self):
        """
        Build a dict group_name -> list[Parameter].
        Called once and cached.
        """
        groups = {
            "encoder":    list(self.encoder.parameters()),
            "classifier": list(self.classifier.parameters()),
        }

        return groups