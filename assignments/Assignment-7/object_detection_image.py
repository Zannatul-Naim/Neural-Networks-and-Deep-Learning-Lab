
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

annotated_img = results[0].plot()

plt.imsave("bus_detected.png", annotated_img)