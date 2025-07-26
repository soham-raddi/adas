import cv2
import lane_detection

if __name__ == '__main__':
    video_filename = r"C:\Users\Soham\Downloads\adas_sample_video.mp4" 

    print(f"Attempting to open video: {video_filename}")
    cap = cv2.VideoCapture(video_filename)
    
    left_line = lane_detection.Line()
    right_line = lane_detection.Line()

    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file '{video_filename}'")
    else:
        print("Video opened successfully. Starting lane detection...")
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Info: End of video file. Exiting.")
                break
            
            try:
                processed_frame = lane_detection.process_frame(frame, left_line, right_line)
                
                # final result
                cv2.imshow("Lane Detection Feed", processed_frame)

            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow("Lane Detection Feed", frame)
            
            # exits program if 'q' is prrssed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("'q' pressed, exiting playback.")
                break

    print("Exiting program and cleaning up.")
    cap.release()
    cv2.destroyAllWindows()
