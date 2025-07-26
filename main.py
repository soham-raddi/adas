import cv2
import numpy as np

# --- Main Execution: Video Playback ---

# To make this project portable, the video file should be in the same
# directory as the Python script.
# If your video is named something else, change the filename here.
video_filename = 'C:\\Users\\Soham\\Downloads\\adas_sample_video.mp4'

print(f"Attempting to open video: {video_filename}")
cap = cv2.VideoCapture(video_filename)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"FATAL ERROR: Could not open video file '{video_filename}'")
    print("Please make sure the video file is in the same folder as this script.")
else:
    print("Video opened successfully. Starting playback...")
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Info: End of video file. Exiting.")
            break
        
        cv2.imshow("Video Feed", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): # Adjusted waitKey for more natural playback speed
            print("'q' pressed, exiting playback.")
            break

# --- Cleanup ---
print("Exiting program and cleaning up.")
cap.release()
cv2.destroyAllWindows()
