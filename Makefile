.PHONY: help install train-detection train-recognition docker-build docker-train export-onnx edge-build test clean

help:
	@echo "Available commands:"
	@echo "  make install              - Install dependencies"
	@echo "  make train-detection      - Train text detection model"
	@echo "  make train-recognition    - Train text recognition model"
	@echo "  make docker-build         - Build Docker images"
	@echo "  make docker-train         - Train using Docker"
	@echo "  make export-onnx          - Export models to ONNX"
	@echo "  make edge-build           - Build C++ edge runtime"
	@echo "  make test                 - Run tests"
	@echo "  make clean                - Clean build artifacts"

install:
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements_trainer.txt

train-detection:
	python application/text_detection/train_detector.py

train-recognition:
	python application/text_recognition/train_recognizer.py

docker-build:
	cd deployment/docker && \
	docker-compose build

docker-train-detection:
	cd deployment/docker && \
	docker-compose up train-detection

docker-train-recognition:
	cd deployment/docker && \
	docker-compose up train-recognition

docker-serve:
	cd deployment/docker && \
	docker-compose up serve

export-onnx:
	python deployment/edge/scripts/export_for_edge.py \
		--detection-checkpoint infrastructure/persistence/checkpoints/detection/best.pth \
		--recognition-checkpoint infrastructure/persistence/checkpoints/ctc/checkpoint_ctc_epoch_30.pth

edge-build:
	cd deployment/edge/cpp && \
	mkdir -p build && cd build && \
	cmake .. && make

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-e2e:
	pytest tests/e2e/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf deployment/edge/cpp/build
	rm -rf .pytest_cache
