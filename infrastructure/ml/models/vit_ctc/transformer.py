import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, n_block, nhead, dim, drop_out):
        super().__init__()
        self.transformer = self._init_transformer(n_block, nhead, dim, drop_out)

    def _init_transformer(self, n_block, nhead, dim, drop_out):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim,
            activation='relu',
            dropout=drop_out,
            batch_first=True
        )
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_block,
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transformer(X)