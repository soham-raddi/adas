import cv2
import lane_detection #module for lane detection

#video processing
if __name__ == '__main__':
    video_filename = "C:\\Users\\Soham\\Downloads\\adas_sample_video.mp4"
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_filename}'")
    else:
        print("Video opened successfully. Starting lane detection...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            processed_frame = lane_detection.process_frame(frame)
            cv2.imshow("Lane Detection", processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
