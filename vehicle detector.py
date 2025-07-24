import cv2

# --- Global Variables & Setup ---
# Load the pre-trained vehicle classifier from OpenCV.
# This XML file should be in the same directory as your scripts.
try:
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    if car_cascade.empty():
        # This error will be printed if the file is missing or corrupted.
        raise IOError("Could not load haarcascade_car.xml. Make sure the file is in your project directory.")
except Exception as e:
    print(f"Error: {e}")
    car_cascade = None

def detect_vehicles(frame):
    """
    Detects vehicles in a frame, draws bounding boxes, and adds a collision warning.
    
    Args:
        frame: The input video frame.
        
    Returns:
        The frame with vehicle detections and warnings drawn on it.
    """
    if car_cascade is None:
        # If the classifier failed to load, display an error on the frame and return it.
        cv2.putText(frame, "Error: Vehicle Classifier Not Loaded", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
        
    # Create a copy to draw on, leaving the original frame clean.
    output_frame = frame.copy()
    gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars using the cascade classifier.
    # The parameters (1.1, 4) can be tuned for sensitivity and performance.
    # 1.1 is the scaleFactor, 4 is minNeighbors.
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles and check for collision warning
    for (x, y, w, h) in cars:
        # Draw a blue rectangle around the detected vehicle
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # --- Forward Collision Warning (FCW) Logic ---
        # If the bottom of the vehicle's bounding box is in the lower
        # quarter of the screen, it's considered "close".
        if y + h > output_frame.shape[0] * 0.75:
            cv2.putText(output_frame, "WARNING: VEHICLE AHEAD", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
    return output_frame
