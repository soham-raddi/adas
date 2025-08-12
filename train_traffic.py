import os
from ultralytics import YOLO

cache_dir = os.path.join(os.getcwd(), 'ultralytics_cache')
os.makedirs(cache_dir, exist_ok=True) 
os.environ['ULTRALYTICS_HOME'] = cache_dir

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # Training process.
    results = model.train(
        data='traffic_sign_dataset/data.yaml',
        epochs=20,                 # Number of training cycles
        imgsz=640,                 
        project='runs/detect',     
        name='train_custom'        
    )

    print("Training complete! The new model is saved in the 'runs/detect/train_custom' folder.")