import os
import cv2
from ultralytics import YOLO

cache_dir = os.path.join(os.getcwd(), 'ultralytics_cache')
os.makedirs(cache_dir, exist_ok=True) 
os.environ['ULTRALYTICS_HOME'] = cache_dir

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
            print("YOLOv8 vehicle model loaded successfully.")
        except Exception as e:
            print(f"Error loading vehicle model: {e}")
            self.model = None

    def detect_vehicles(self, frame):
        if self.model is None: return frame
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                if class_name in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame