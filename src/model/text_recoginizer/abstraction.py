from typing import List, Protocol, Any
import torch

class IBackbone(Protocol):
    def freeze_parameter(self, layers: List[int] = [], finetune_classifycation: bool = False) -> None:
        ...
    
    @property
    def get_model(self) -> torch.nn.Module:
        ...

class IPatchEmbedding(Protocol):
    def forward(self, X: torch.Tensor):
        ...

    def extract_patches(images, rows, cols):
        ...

class IPosEmbedding(Protocol):
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ...

class IPosPatchEmbedding(Protocol):
    def forward(self, images: List[Any], rows, cols) -> torch.Tensor:
        ...