import cv2
import time
import lane_detection
import vehicle_detection 
import sign_detection

if __name__ == '__main__':
    video_filename = './sample videos/adas_sample_video1.mp4'
    cap = cv2.VideoCapture(video_filename)

    hd = vehicle_detection.HazardDetector()
    sd = sign_detection.SignDetector()

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_filename}'")
    else:
        print("Video opened successfully. Starting ADAS application...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            
            try:
                height, width = frame.shape[:2]
                scale = 1280 / width
                small_frame = cv2.resize(frame, (1280, int(height * scale)), interpolation=cv2.INTER_AREA)
                frame_with_lanes = lane_detection.process_frame(small_frame)
                frame_with_hazards = hd.detect_hazards(frame_with_lanes)
                final_frame = sd.detect_signs(frame_with_hazards)
                cv2.imshow("ADAS Feed", final_frame)

            except Exception as e:
                print(f"An error occurred during frame processing: {e}")
                cv2.imshow("ADAS Feed", frame) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()