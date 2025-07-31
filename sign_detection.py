import os
import cv2
from ultralytics import YOLO

# This script uses the same local cache setup to avoid permission errors.
cache_dir = os.path.join(os.getcwd(), 'ultralytics_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['ULTRALYTICS_HOME'] = cache_dir

class SignDetector:
    """
    A class to encapsulate the YOLOv8 traffic sign detection logic.
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the SignDetector with a pre-trained YOLOv8 model.
        """
        try:
            # We use the same YOLOv8 model as it can detect multiple object types.
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            # --- Define the specific classes we are interested in ---
            # The default 'yolov8n.pt' model is trained on the COCO dataset,
            # which includes 'traffic light' and 'stop sign'.
            self.sign_classes = ['traffic light', 'stop sign']
            print("Sign detection model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model for sign detection: {e}")
            self.model = None

    def detect_signs(self, frame):
        """
        Takes a single video frame and returns it with detected signs boxed.
        """
        if self.model is None:
            return frame

        # Pass the frame to the model for inference
        results = self.model(frame)

        # Process the results
        for result in results:
            for box in result.boxes:
                # Get the class ID and name
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]

                # Filter for only the sign classes we care about
                if class_name in self.sign_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # --- Draw Bounding Box and Label ---
                    # Use a different color for signs (e.g., magenta) to distinguish them
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return frame