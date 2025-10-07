Real-Time Advanced Driver-Assistance System (ADAS)
This project is a Python-based ADAS that uses computer vision and deep learning to analyze road footage in real-time. It identifies lane lines, detects hazards like vehicles and pedestrians, and recognizes custom-trained traffic signs, simulating the core functionalities of modern driver-assistance technologies.

The system processes a video feed using a combination of OpenCV for lane detection and the YOLOv8 object detection model for hazard and sign recognition.

Key Features
Lane Detection: Utilizes an OpenCV-based pipeline to detect and draw lane markings on the road.

Hazard Detection (Vehicles & Pedestrians): Employs a pre-trained YOLOv8 model to accurately detect and track cars, trucks, buses, motorbikes, and people.

Custom Traffic Sign Recognition: Uses a custom-trained YOLOv8 model to identify a specific set of traffic signs relevant to the driving environment.

Modular Design: The code is structured into separate, easy-to-understand modules for each detection task, orchestrated by a central main.py script.

System Architecture
The application operates on a sequential processing pipeline for each frame of the video:

Frame Input: main.py reads a frame from the video source.

Lane Detection: The frame is passed to the lane_detection module, which draws lane lines.

Hazard Detection: The frame (now with lanes) is passed to the hazard_detection module to identify vehicles and pedestrians.

Sign Detection: The frame (now with lanes and hazards) is passed to the sign_detection module to find custom traffic signs.

Display: The final, fully annotated frame is displayed to the user.

Installation (Recommended: GPU Setup)
For optimal performance, especially for training, a computer with an NVIDIA GPU is highly recommended. This setup uses Conda to manage the complex CUDA dependencies.

1. NVIDIA Driver & CUDA Setup
First, ensure your system is ready for GPU acceleration.

Go to the NVIDIA Driver Downloads page: https://www.nvidia.com/Download/index.aspx

Select Your Driver:

Product Type: GeForce

Product Series: GeForce RTX 30 Series (Laptops)

Product: GeForce RTX 3050 Laptop GPU

Download Type: Studio Driver (SD)

Download and Install the driver using the Express option.

Restart your computer after the installation is complete.

2. Create a Conda Environment
We will use Miniconda to create an isolated environment for the project.

Install Miniconda if you don't have it: Miniconda Download Page.

Create and activate the environment by opening the Anaconda Prompt and running:

conda create -n adas_gpu python=3.10 -y
conda activate adas_gpu

3. Install Libraries
Inside the activated adas_gpu environment, install PyTorch with CUDA support and the other project libraries.

Install PyTorch:

pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Install Ultralytics and OpenCV:

pip install ultralytics opencv-python

Verify GPU Installation: Run python and enter the following commands. The output should be True.

import torch
print(torch.cuda.is_available())
# Expected output: True
exit()

Installation (CPU-Only Fallback)
If you do not have a compatible NVIDIA GPU, you can run the project on your CPU, though training will be very slow.

Clone the repository:

git clone <your-repository-url>
cd adas

Create and activate a virtual environment:

# Create the environment
python -m venv .venv
# Activate on Windows
.\.venv\Scripts\Activate
# Activate on macOS/Linux
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Usage
Activate your environment:

# If using GPU setup
conda activate adas_gpu
# If using CPU setup
.\.venv\Scripts\Activate

Navigate to the project directory:

cd path/to/your/project/folder

Run the application:

python main.py

The application will start, displaying the processed video feed with all detections. Press 'q' to quit.

Training the Custom Sign Detector
The sign_detection module relies on a custom-trained model.

1. Prepare Your Dataset
Place your labeled images and a data.yaml configuration file in a folder (e.g., traffic_sign_dataset/).

The data.yaml file should point to the training and validation image directories and list the class names.

2. Run the Training Script
Modify the train_traffic.py script to optimize training time and performance. Using freeze=10 is highly recommended to speed up training by only fine-tuning the final layers of the model.

# In train_traffic.py
results = model.train(
    data='./traffic_sign_dataset/data.yaml',
    epochs=50,      # A higher number of epochs for better confidence
    imgsz=640,
    project='runs/detect',
    name='train_custom',
    freeze=10       # Freeze the backbone for much faster training
)

Execute the training script from your activated environment:

python train_traffic.py

The best model will be saved as best.pt inside the runs/detect/train_custom/weights/ folder. Ensure this is the path used in sign_detection.py.

Troubleshooting & Improving Detections
If you encounter incorrect detections, here are the common causes and solutions.

Incorrect Sign Detections (Low Confidence)
Problem: The custom model misidentifies a sign with a very low confidence score (e.g., 0.44).

Cause: The model is "underfit" and uncertain due to insufficient training.

Solutions:

Retrain for more epochs: As shown above, increase epochs to 50 or more in train_traffic.py to build the model's confidence.

Increase Confidence Threshold: In sign_detection.py, raise the conf value to filter out weak guesses. This is best done after retraining.

# In sign_detection.py
results = self.model(frame, conf=0.6, verbose=False) # Requires 60% confidence

Improve Your Dataset: Add more varied images, especially of the signs the model struggles with.

False Detections (e.g., "person" on a signpost)
Problem: The pre-trained model "hallucinates" objects like people on inanimate objects.

Cause: The yolov8n.pt ("nano") model is optimized for speed, not accuracy. It can be easily confused by ambiguous shapes.

Solution: Switch to a more accurate model. The yolov8s.pt ("small") model is a great balance of speed and improved accuracy.

# In hazard_detection.py
# Change the model path in the __init__ method
def __init__(self, model_path='yolov8s.pt'): # Use the "small" model

Ultralytics will automatically download this model the first time you run the script.