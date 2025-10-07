import os
import cv2
from ultralytics import YOLO

cache_dir = os.path.join(os.getcwd(), 'ultralytics_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['ULTRALYTICS_HOME'] = cache_dir

class HazardDetector:
    def __init__(self, model_path='yolov8s.pt'):
        try:
            self.model = YOLO(model_path)
            self.target_classes = ['car', 'truck', 'bus', 'motorbike', 'person']
            
            self.target_class_ids = [
                k for k, v in self.model.names.items()
                if v in self.target_classes
            ]
            print("YOLOv8 hazard model loaded successfully.")
            print(f"Detecting: {self.target_classes}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect_hazards(self, frame):
        if self.model is None:
            return frame
        
        results = self.model(frame, classes=self.target_class_ids, verbose=False)
        annotated_frame = results[0].plot()
        
        return annotated_frame