# adas
Advanced Driver Assistance System - Suitable for Indian Traffic Conditions
Real-Time Advanced Driver-Assistance System (ADAS)
This project is a Python-based Advanced Driver-Assistance System (ADAS) that uses computer vision and deep learning to analyze road footage in real-time. It identifies vehicles, recognizes traffic signs, and provides crucial information to the driver, simulating the core functionalities of modern ADAS technologies.

The system processes a video feed using a combination of the YOLOv8 object detection model for vehicle tracking and a custom-trained model for traffic sign recognition.

Features
Real-Time Vehicle Detection: Utilizes the pre-trained YOLOv8 model to accurately detect and track cars, trucks, and buses on the road.

Traffic Sign Recognition: Employs a custom-trained YOLO model to identify a wide range of traffic signs, including speed limits, stop signs, and traffic lights.

Performance Measurement: Calculates and displays the real-time frames per second (FPS) to monitor processing performance.

Modular Design: The code is structured into separate modules for vehicle detection, sign detection, and performance monitoring, making it easy to extend and maintain.

Installation
Follow these steps to set up the project environment on your local machine.

1. Prerequisites
Python 3.10

pip (Python package installer)

2. Setup Instructions
a. Clone the repository:

git clone <your-repository-url>
cd adas

b. Create and activate a virtual environment:

It is highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment (using Python 3.10)
py -3.10 -m venv .venv

# Activate the environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

c. Install dependencies:

The provided requirements.txt file contains all the necessary packages with compatible versions.

pip install -r requirements.txt

d. Place the Custom Model Weights:

This project requires a custom-trained model for traffic sign detection. You must place the weights file in the correct directory.

Place your best.pt file inside the following folder structure: runs/detect/train_custom/weights/

Usage
Once the installation is complete, you can run the main application.

Add your video file: Place the video you want to analyze into the sample_videos/ directory.

Update the script: Open the main.py file and update the video_filename variable to point to your video file.

# in main.py
video_filename = './sample_videos/your_video_name.mp4'

Run the application: Execute the main script from the root directory of the project.

python main.py

The application will start, display the processed video feed with bounding boxes for vehicles and traffic signs, and print relevant information to the console.