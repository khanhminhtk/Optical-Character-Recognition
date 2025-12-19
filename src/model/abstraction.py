from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path

class ITrainer(ABC):
    @abstractmethod
    def load_model(self, weights_path: str):
        ...

    @abstractmethod
    def train(self, data_yaml: str):
        ...

    @abstractmethod
    def validate(self, **kwargs):
        ...

    @abstractmethod
    def predict(self, source: Union[str, Path, list], conf: float = 0.5, save: bool = False, **kwargs) -> list:
        ...
    
    @abstractmethod
    def save(self, save_path: str):
        ...

    @abstractmethod
    def export(self, format: str = "onnx", **kwargs) -> str:
        ...

    @property
    def model(self):
        ...

    @property
    def results(self):
        ...

    @property
    def metrics(self):
        ...

    @property
    def best_weights_path(self) -> Optional[str]:
        ...
