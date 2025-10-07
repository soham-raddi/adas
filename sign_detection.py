import os
import cv2
from ultralytics import YOLO

cache_dir = os.path.join(os.getcwd(), 'ultralytics_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['ULTRALYTICS_HOME'] = cache_dir

class SignDetector:
    def __init__(self, model_path='runs/detect/train_custom3/weights/best.pt'):
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print("Custom-trained traffic sign model loaded successfully.")
            print(f"Model can detect: {list(self.class_names.values())}")
        except Exception as e:
            print(f"Error loading custom YOLO model: {e}")
            print(f"Please ensure the model path is correct: '{model_path}'")
            self.model = None

    def detect_signs(self, frame):
        if self.model is None:
            return frame
        
        results = self.model(frame, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()
        
        return annotated_frame