from typing import Optional
import os

import cv2
import torch
import numpy as np

from src.model.abstraction import ITrainer
from src.model.text_detection.trainer import Trainer, load_training_config
from model.text_recoginizer.model_test import create_text_recognizer
from src.model.text_recoginizer.patch_embedding import PatchEmbedding


class Visualizer:
    def __init__(
            self, trainer: ITrainer = None, 
            stream_url = 0, 
            weights_path: Optional[str] = None, 
            ocr_model_path: Optional[str] = None, 
            device: str = "cpu"
        ):
        self.trainer = trainer
        self.stream_url = stream_url
        self.device = device
        self._load_model(weights_path=weights_path)
        self._load_ocr_model(ocr_model_path)

    def _load_model(self, weights_path):
        self.trainer.load_model(weights_path=weights_path)
    
    def _load_ocr_model(self, model_path):
        self.ocr_model = create_text_recognizer(num_classes=26)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.ocr_model.load_state_dict(checkpoint['model_state_dict'])

        self.ocr_model = self.ocr_model.to(self.device)
        self.ocr_model.eval()
    
    def _recognize_text(self, crop_img):
        try:
            patches = PatchEmbedding.extract_patches([crop_img], rows=4, cols=2)
            patches = patches.to(self.device)

            with torch.no_grad():
                outputs = self.ocr_model(patches)
                probs = torch.softmax(outputs, dim=-1)
                pred = torch.argmax(probs, dim=-1)
            char = chr(65 + pred.item())
            confidence = probs[0, pred.item()].item()
            return f"{char}({confidence:.2f})"
        except Exception as e:
            print(f"OCR Error: {e}")
            return "?"
    
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
                            ocr_text = ""
                            if self.ocr_model is not None:
                                crop = display_frame[y1:y2, x1:x2].copy()
                                if crop.size > 0:
                                    ocr_text = self._recognize_text(crop)
                            
                            print(f"Bbox: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.4f}, Class: {cls}, Text: {ocr_text}")
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if ocr_text:
                                text = f"{ocr_text} {conf:.2f}"
                            else:
                                text = f"Conf: {conf:.2f}"
                            
                            cv2.putText(display_frame, text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow("test", display_frame)
                    break


                if cv2.waitKey(1) == 27:
                    break

    

if __name__ == "__main__":
    train_config_path = "/home/minhtk/code/ai-project/Optical-Character-Recognition/src/model/text_detection/config/train_configs.yaml"
    weights_path = "/home/minhtk/code/ai-project/Optical-Character-Recognition/runs/detect/train3/weights/best.pt"
    ocr_model_path = "/home/minhtk/code/ai-project/Optical-Character-Recognition/checkpoints/best_model.pth"
    
    train_config = load_training_config(yaml_path=train_config_path)
    train = Trainer(config=train_config)

    stream = Visualizer(
        trainer=train, 
        weights_path=weights_path,
        ocr_model_path=ocr_model_path,
        use_ocr=True
    )
    
    stream.process_frame()