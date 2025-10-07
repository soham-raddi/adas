Real-Time Advanced Driver-Assistance System (ADAS)

This project is a Python-based Advanced Driver-Assistance System (ADAS) that uses computer vision and deep learning to analyze road footage in real time.
It identifies lane lines, detects hazards such as vehicles and pedestrians, and recognizes custom-trained traffic signs, simulating the key functionalities of modern driver-assistance technologies.

The system processes video input using a combination of OpenCV for lane detection and the YOLOv8 object detection model for hazard and sign recognition.

Key Features

Lane Detection:
Uses an OpenCV-based pipeline to detect and draw lane markings on the road.

Hazard Detection (Vehicles & Pedestrians):
Employs a pre-trained YOLOv8 model to detect and track cars, trucks, buses, motorbikes, and pedestrians.

Custom Traffic Sign Recognition:
Uses a custom-trained YOLOv8 model to identify traffic signs relevant to the driving environment.

Modular Design:
Each detection task is implemented in a separate module and orchestrated by a central main.py script.

System Architecture

The application follows a sequential frame processing pipeline:

Frame Input:
main.py reads a frame from the video source.

Lane Detection:
The frame is passed to the lane_detection module, which identifies and draws lane lines.

Hazard Detection:
The frame is passed to the hazard_detection module to detect vehicles and pedestrians.

Sign Detection:
The frame is passed to the sign_detection module to identify custom traffic signs.

Display:
The final annotated frame (lanes, hazards, and signs) is displayed to the user.

Installation (Recommended: GPU Setup)

For optimal performance, especially during training, a system with an NVIDIA GPU is recommended.

1. NVIDIA Driver and CUDA Setup

Go to the NVIDIA Driver Downloads
 page.

Select your driver:

Product Type: GeForce

Product Series: GeForce RTX 30 Series (Laptops)

Product: GeForce RTX 3050 Laptop GPU

Download Type: Studio Driver (SD)

Download and install the driver (use the Express installation option).

Restart your system after installation.

2. Create a Conda Environment

Use Miniconda to manage dependencies.

conda create -n adas_gpu python=3.10 -y
conda activate adas_gpu

3. Install Libraries

Install PyTorch with CUDA support and the required libraries.

# Install PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics (YOLOv8) and OpenCV
pip install ultralytics opencv-python

Verify GPU Installation

Run Python and check CUDA availability:

import torch
print(torch.cuda.is_available())
# Expected output: True

Installation (CPU-Only Fallback)

If you do not have a compatible GPU, the project can be run on CPU (training will be slower).

# Clone the repository
git clone <your-repository-url>
cd adas

# Create and activate a virtual environment
python -m venv .venv

# On Windows
.\.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Usage

Activate your environment:

# GPU setup
conda activate adas_gpu

# CPU setup
.\.venv\Scripts\activate


Navigate to the project directory:

cd path/to/your/project/folder


Run the application:

python main.py


The application will display the processed video feed with lane, hazard, and sign detections.
Press q to quit.

Training the Custom Sign Detector

The sign_detection module relies on a custom-trained YOLOv8 model.

1. Prepare Your Dataset

Organize your dataset as follows:

traffic_sign_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml


Example data.yaml:

train: ./traffic_sign_dataset/train/images
val: ./traffic_sign_dataset/val/images
names:
  0: speed_limit
  1: stop
  2: pedestrian_crossing

2. Train the Model

Edit the train_traffic.py script:

results = model.train(
    data='./traffic_sign_dataset/data.yaml',
    epochs=50,      # Increase for better performance
    imgsz=640,
    project='runs/detect',
    name='train_custom',
    freeze=10       # Freeze the backbone for faster fine-tuning
)


Run the training script:

python train_traffic.py


The best model will be saved in:

runs/detect/train_custom/weights/best.pt


Ensure this path is correctly referenced in sign_detection.py.

Troubleshooting and Improvements
Incorrect Sign Detections (Low Confidence)

Problem: The model misidentifies a sign with a low confidence score.
Cause: The model is underfit due to insufficient training or limited data.

Solutions:

Retrain with more epochs (e.g., 50+).

Increase the confidence threshold in sign_detection.py:

results = self.model(frame, conf=0.6, verbose=False)


Improve the dataset with more varied and balanced samples.

False Detections (e.g., "person" on a signpost)

Problem: The pre-trained YOLOv8n model incorrectly detects objects.
Cause: YOLOv8n is optimized for speed, not accuracy.

Solution:
Switch to a more accurate YOLOv8s model in hazard_detection.py:

def __init__(self, model_path='yolov8s.pt'):
    self.model = YOLO(model_path)


Ultralytics will automatically download the required model when first used.

Project Structure
ADAS/
├── main.py
├── lane_detection.py
├── hazard_detection.py
├── sign_detection.py
├── train_traffic.py
├── requirements.txt
└── traffic_sign_dataset/

License

This project is open source under the MIT License.
You may modify and use it for research or educational purposes.

Acknowledgments

Ultralytics YOLOv8

OpenCV

NVIDIA CUDA Toolkit