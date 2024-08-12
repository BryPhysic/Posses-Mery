from ultralytics import YOLO

from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor
model = YOLO('Models/Best weights.pt')
print(type(model))


results = model(source='0',show=True)
print(results)