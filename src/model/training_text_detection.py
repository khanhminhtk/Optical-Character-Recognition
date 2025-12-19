import yaml
from dataclasses import asdict

from src.model.text_detection.config.trainconfig import TrainConfigs
from src.model.text_detection.trainer import Trainer



# if __name__ == "__main__":
#     path_train_config = "/home/minhtk/code/ai-project/Optical-Character-Recognition/model/text_detection/config/train_configs.yaml"
#     config = _load_training_config(path_train_config)

#     data_yaml = "/home/minhtk/code/ai-project/Optical-Character-Recognition/data_test/yolo_overfit/data.yaml"
#     trainer = Trainer(
#         config=config
#     )
#     trainer.load_model(weights_path=config.weights)
#     trainer.train(
#         data_yaml=data_yaml
#     )
