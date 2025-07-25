#!/usr/bin/env python3
import cv2
import argparse
import logging
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------
# Frame Processing Stub
# Replace the body of this function with your ADAS/TSR logic.
# ----------------------------------------
def process_frame(frame):
    """
    Example placeholder for ADAS/TSR processing.
    Currently returns the frame unchanged.
    """
    # TODO: insert traffic-sign recognition or other analysis here
    return frame 

# ----------------------------------------
# Argument Parsing
# ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Robust video player with ADAS/TSR processing hook"
    )
    parser.add_argument(
        "source",
        help="Video source: camera index (int) or file path",
    )
    parser.add_argument(
        "--backend",
        choices=["default", "ffmpeg", "dshow", "msmf"],
        default="default",
        help="OpenCV VideoCapture backend to use",
    )
    parser.add_argument(
        "--width", "-W", type=int, default=640, help="Frame width"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=480, help="Frame height"
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay FPS on the video",
    )
    parser.add_argument(
        "--window", "-w", default="ADAS Preview", help="Window name"
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()

# ----------------------------------------
# Configure Logging
# ----------------------------------------
def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------------------------
# Open Video Source with Selected Backend
# ----------------------------------------
def open_capture(source, backend_choice):
    # Determine backend flag
    backend_flags = {
        "default": cv2.CAP_ANY,
        "ffmpeg": cv2.CAP_FFMPEG,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }
    flag = backend_flags.get(backend_choice, cv2.CAP_ANY)

    # Interpret numeric camera index or file path
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src, flag)
    if not cap.isOpened():
        raise IOError(f"Unable to open source '{source}' with backend '{backend_choice}'")
    return cap

# ----------------------------------------
# Main Loop
# ----------------------------------------
def main():
    args = parse_args()
    configure_logging(args.loglevel)

    logging.info("Starting ADAS/TSR video player")
    logging.debug(f"Args: {args}")

    try:
        cap = open_capture(args.source, args.backend)
    except IOError as e:
        logging.critical(e)
        sys.exit(1)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    logging.info(f"Opened source; resolution set to {args.width}x{args.height}")

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of stream or cannot read frame; exiting.")
            break

        # Process frame (ADAS/TSR logic goes here)
        processed = process_frame(frame)

        # Compute and overlay FPS if requested
        if args.show_fps:
            frame_count += 1
            elapsed = time.time() - last_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_time = time.time()
            cv2.putText(
                processed,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(args.window, processed)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            logging.info("ESC pressed; exiting.")
            break
        elif key == ord("q"):
            logging.info("'q' pressed; exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Cleanup complete; program terminated.")

if __name__ == "__main__":
    main()