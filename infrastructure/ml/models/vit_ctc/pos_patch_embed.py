from typing import List, Any

import torch
import torch.nn as nn

from infrastructure.ml.models.vit_ctc.abstraction import IPatchEmbedding, IPosEmbedding

class PatchEmbedWithPos(nn.Module):
    def __init__(self, patch_embedding: IPatchEmbedding, pos_embedding: IPosEmbedding):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.pos_embedding = pos_embedding

    def forward(self, images: List[Any], rows, cols) -> torch.Tensor:
        patchs = self.patch_embedding.extract_patches(
            images=images,
            rows=rows,
            cols=cols
        )
        extract_feature = self.patch_embedding.forward(X=patchs)
        return self.pos_embedding.forward(X=extract_feature)
