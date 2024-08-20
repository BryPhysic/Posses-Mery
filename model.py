
from ultralytics import YOLO

def load_model(model_path='yolo/yolov8n-pose.pt'):
    """
    Función para cargar el modelo YOLOv8.
    """
    model = YOLO(model_path)
    return model