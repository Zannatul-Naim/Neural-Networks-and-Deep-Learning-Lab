
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")


model.model.fuse() 

for param in list(model.model.parameters())[:-10]:
    param.requires_grad = False

model.train(data="datasets/widerface/data.yaml", epochs=5, imgsz=640, batch=16)
