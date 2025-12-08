from typing import Optional, Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfigs:
    # Required
    model_name: str

    # Optional with sensible defaults (to keep dataclass field ordering valid)
    pretrained: bool = False
    weights: Optional[str] = None

    # Training schedule
    epochs: int = 200
    imgsz: int = 1024
    batch_size: int = 16
    patience: int = 10

    # Optimizer (defaults chosen as reasonable starting values)
    optimizer: str = "adamw"
    lr0: float = 1e-3
    weight_decay: float = 0.0

    # Augmentation
    augment: bool = True

    # Save
    project: str = "runs/train"
    name: str = "exp"
    save_period: int = 1

    # Device
    device: Optional[str] = None  # 'cpu' or 'cuda' or None for auto-select

    workers: int = 8

    @classmethod
    def from_dict(cls, dictionary: Dict[str, str]):
        lr0_val = dictionary.get("lr0", 1e-3)
        weight_decay_val = dictionary.get("weight_decay", 0.0)
        
        return cls(
            model_name = dictionary.get("model_name"),
            pretrained = dictionary.get("pretrained", False),
            weights = dictionary.get("weights"),
            epochs = int(dictionary.get("epochs", 200)),
            imgsz = int(dictionary.get("imgsz", 1024)),
            batch_size = int(dictionary.get("batch_size", 16)),
            patience = int(dictionary.get("patience", 10)),
            optimizer = dictionary.get("optimizer", "adamw"),
            lr0 = float(lr0_val) if lr0_val is not None else 1e-3,
            weight_decay = float(weight_decay_val) if weight_decay_val is not None else 0.0,
            augment = dictionary.get("augment", True),
            project = dictionary.get("project", "runs/train"),
            name = dictionary.get("name", "exp"),
            save_period = int(dictionary.get("save_period", 1)),
            device = dictionary.get("device"),
            workers = int(dictionary.get("workers", 8))
        )
