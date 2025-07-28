import cv2
import lane_detection #script for lane detection
import vehicle_detection #script for vehicle identification and detection

# main video processing
if __name__ == '__main__':
    video_filename = "C:\\Users\\Soham\\Downloads\\adas_sample_video2.mp4"
    cap = cv2.VideoCapture(video_filename)

    try:
        vd = vehicle_detection.VehicleDetector()
    except AttributeError:
        print("Error: Could not find 'VehicleDetector' class.")
        print("Please ensure you have a 'vehicle_detector.py' file with the correct class definition.")
        vd = None


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
                #lane detection logic
                frame_with_lanes = lane_detection.process_frame(frame)

                #vehicle detection logic
                if vd:
                    #if the detection module loads
                    final_frame = vd.detect_vehicles(frame_with_lanes)
                else:
                    #else only show the lane lines
                    final_frame = frame_with_lanes 

                # final result
                cv2.imshow("ADAS Feed", final_frame)

            except Exception as e:
                print(f"An error occurred: {e}")
                cv2.imshow("ADAS Feed", frame)

            # Exit on 'q' key press
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()