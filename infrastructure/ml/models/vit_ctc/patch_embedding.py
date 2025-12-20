import cv2

import torch
import torch.nn as nn

from infrastructure.ml.models.vit_ctc.abstraction import IBackbone

class PatchEmbedding(nn.Module):
    def __init__(self, backbone: IBackbone, num_patches: int = 8, rows: int = 4, cols: int = 2):
        super().__init__()
        self.backbone = backbone
        self.num_patches = num_patches
        self.rows = rows
        self.cols = cols
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, X: torch.Tensor):
        batch_size, num_patches, C, H, W = X.shape
        X = X.view(batch_size * num_patches, C, H, W)
        X = X.float() / 255.0
        X = (X - self.mean) / self.std
        embeddings = self.backbone.get_model(X)
        embedding_dim = embeddings.shape[-1]
        embeddings = embeddings.view(batch_size, num_patches, embedding_dim)
        
        return embeddings

    @staticmethod
    def extract_patches(images, rows, cols):
        patch_height = 224 // rows
        patch_width = 224 // cols
        images = [cv2.resize(src=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB), dsize=(224, 224)) for img in images]
        images_patchs = []
        for image in images:
            patchs = []
            for i in range(rows):
                for j in range(cols):
                    y_start = i * patch_height
                    y_end = (i + 1) * patch_height
                    x_start = j * patch_width
                    x_end = (j + 1) * patch_width
                    
                    patch = cv2.resize(src=image[y_start:y_end, x_start:x_end], dsize=(224, 224))
                    patchs.append(patch)
            patchs = torch.stack([torch.from_numpy(p).permute(2, 0, 1) for p in patchs])
            images_patchs.append(patchs)
        return torch.stack(images_patchs)
