# Optical-Character-Recognition
Sử dụng bộ dữ liệu ICDAR 2003 với hơn 1000 bức ảnh xây dựng đường ống và dữ liệu cho phù hợp với format của yolo
Thực hiện fine-tuning mô hình YOLOv8 cho bài toán text detection.
Kết hợp fine-tuning của ResNet và LSTM để xây dựng một hệ thống nhận diện văn bản (text recognizer) và xây dựng hệ tống toàn diện end-to-end kết hợp 2 bài toán lại với nhau

# Kết quả thu được:
## Bài toán text detection
Thu được mô hình yolov8 sau khi pretrain với chỉ số mAP50 là 0.887 và chỉ số mAP50-95 là 0.681

![image](https://github.com/user-attachments/assets/b6ff3243-4881-4e13-9d47-50dbb364f118)
![image](https://github.com/user-attachments/assets/f47303a2-fb4a-4d98-9a91-ff2c4a99f13c)
## Bài toán text recognizer
Thu được mô hình CRNN với pretain mô hình Resnet với chỉ số validate loss và test lần lược là 1.1295359219823564 và 1.5311663842626981

![image](https://github.com/user-attachments/assets/9f88e11d-f173-4975-b3ac-9d3db5a3be03)
![image](https://github.com/user-attachments/assets/135a1c3b-18ba-4a64-b73d-c4f2919b24bf)
## End to end 2 mô hình cho bài toán OCR
![image](https://github.com/user-attachments/assets/39d1a7bb-bc83-4d6d-b9ba-87b235b4b72f)
