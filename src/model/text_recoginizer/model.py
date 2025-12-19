from typing import List, Any

import torch.nn as nn

from src.model.text_recoginizer.pos_patch_embed import PatchEmbedWithPos
from src.model.text_recoginizer.transformer import TransformerBlock

class ModelTextRecoginizer(nn.Module):
    def __init__(
            self,
            patchembedwithpos: PatchEmbedWithPos,
            transformer: TransformerBlock,
            embedding_dim: int,
            num_classes: int = 26,
            dropout: float = 0.2,
            hidden_classifier: int = 512
        ):
        super().__init__()
        self.patchembedwithpos = patchembedwithpos
        self.transformer = transformer
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_classifier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_classifier, num_classes)
        )
    def forward(self, images: List[Any], rows, cols):
        out_patch = self.patchembedwithpos(images, rows, cols)
        out_transformer = self.transformer(out_patch)
        out_pool = out_transformer.mean(dim=1)
        out_logits = self.classifier(out_pool)
        return out_logits

