from typing import List

import torch
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, mobilenet_v3_small

class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def freeze_parameter(self, layers: List[int] = [], finetune_classifycation: bool = False):
        for param in self._model.parameters():
            param.requires_grad = False

        if finetune_classifycation:
            for param in self._model.classifier.parameters():
                param.requires_grad = True

        if len(layers) > 0:
            for layer in layers:
                for param in ((self._model.features)[layer]).parameters():
                    param.requires_grad = True

    @property
    def get_model(self):
        return self._model
    
    def forward(self, X: torch.Tensor):
        return self._model.forward(X)


# model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights)

# backbone = Backbone(model=model)
# backbone.freeze_parameter(layers=[-1, -2], finetune_classifycation=True)

# print(backbone.get_model)

    


