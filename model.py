
from ultralytics import YOLO



def load_model(model_path='yolov8.pt'):
    """
    Función para cargar el modelo YOLOv8.
    """
    model = YOLO(model_path)
    return model