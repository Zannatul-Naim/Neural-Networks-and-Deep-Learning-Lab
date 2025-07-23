
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO("./yolov8n_100e.pt")

# Run prediction
results = model("../test_image/faces_img.jpg", save=False)


annotated_img = results[0].plot()

plt.imshow(annotated_img)
plt.axis('off')
plt.title("YOLOv8 Prediction")
plt.savefig("yusepp_face_prediction_output.png")
print("Prediction saved as yusepp_face_prediction_output.png")
