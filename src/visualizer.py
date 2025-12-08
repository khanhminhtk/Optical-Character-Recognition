from typing import Optional

import cv2

from src.model.abstraction import ITrainer
from src.model.text_detection.trainer import Trainer, load_training_config


class Visualizer:
    def __init__(self, trainer: ITrainer = None, stream_url = 0, weights_path: Optional[str] = None):
        self.trainer = trainer
        self.stream_url = stream_url
        self._load_model(weights_path=weights_path)
    
    def _load_model(self, weights_path):
        self.trainer.load_model(weights_path=weights_path)
    
    def _predict(self, frame, device, stream, verbose):
        result = self.trainer.predict(source=frame, device=device, stream=stream, verbose=verbose)
        return result

    def process_frame(self):
        cap = cv2.VideoCapture(self.stream_url)
        if cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS: ", fps)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(src=frame, dsize=(640, 640))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._predict(frame, "cuda", stream=True, verbose=True)
                
                display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            x1, y1, x2, y2 = map(int, bbox)                            
                            print(f"Bbox: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.4f}, Class: {cls}")
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text = f"Conf: {conf:.2f} hehe"
                            cv2.putText(display_frame, text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow("test", display_frame)
                    break


                if cv2.waitKey(1) == 27:
                    break

    

train_config_path = "/home/minhtk/code/ai-project/Optical-Character-Recognition/src/model/text_detection/config/train_configs.yaml"
weights_path = "/home/minhtk/code/ai-project/Optical-Character-Recognition/runs/detect/train3/weights/best.pt"


train_config = load_training_config(yaml_path=train_config_path)
train = Trainer(config=train_config)
# train.load_model(weights_path=weights_path)

# # Predict và lấy results
# results = train.predict(source="/home/minhtk/code/ai-project/Optical-Character-Recognition/data/train/images/1.jpg")

# # Lấy tọa độ bounding boxes
# bboxes = train.get_bboxes(results)
# print("\nDetected bounding boxes:")
# for i, bbox_info in enumerate(bboxes):
#     print(f"Detection {i+1}:")
#     print(f"  Bbox (x1, y1, x2, y2): {bbox_info['bbox']}")
#     print(f"  Confidence: {bbox_info['confidence']:.4f}")
#     print(f"  Class: {bbox_info['class_name']} (id: {bbox_info['class']})")

stream = Visualizer(trainer=train, weights_path=weights_path)
stream.process_frame()