import cv2
import time # For timing the processing speed

# importing the necessary modules
import lane_detection
import vehicle_detection
import sign_detection

# main video processing function
if __name__ == '__main__':
    video_filename = './adas/sample videos/adas_sample_video1.mp4'
    cap = cv2.VideoCapture(video_filename)

    # initialising the detectors
    vd = vehicle_detection.VehicleDetector()
    sd = sign_detection.SignDetector()

    # optimising the frame processing
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 1

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_filename}'")
    else:
        print("Video opened successfully. Starting ADAS application...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_counter += 1
            
            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    height, width = frame.shape[:2]
                    scale = 1280 / width
                    small_frame = cv2.resize(frame, (1280, int(height * scale)), interpolation=cv2.INTER_AREA)
                    frame_with_lanes = lane_detection.process_frame(small_frame)
                    frame_with_vehicles = vd.detect_vehicles(frame_with_lanes)
                    
                    final_frame = sd.detect_signs(frame_with_vehicles)
                    cv2.imshow("ADAS Feed", final_frame)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    cv2.imshow("ADAS Feed", frame) 
            else:
                cv2.imshow("ADAS Feed", frame)
            # to exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()