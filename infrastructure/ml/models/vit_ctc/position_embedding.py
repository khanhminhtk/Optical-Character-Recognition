import torch
import torch.nn as nn


class PosEmbedding(nn.Module):
    def __init__(self, dim: int, num_patches: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X + self.pos_embedding