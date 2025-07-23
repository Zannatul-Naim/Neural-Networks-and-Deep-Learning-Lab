
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO("runs/detect/train/weights/best.pt")

results = model("test_image/faces_img.jpg", save=False)

metrics = model.val()

precision, recall, map50, map50_95 = metrics.box.mean_results()

results_text = f"""YOLOv8 Validation Metrics:
Precision: {precision:.4f}
Recall: {recall:.4f}
mAP@0.5: {map50:.4f}
mAP@0.5:0.95: {map50_95:.4f}
"""

# Save to metrics.txt
with open('yolov8_widerface_dataset_metrics.txt', 'w') as f:
    f.write(results_text)


annotated_img = results[0].plot()

plt.imshow(annotated_img)
plt.axis('off')
plt.title("YOLOv8 Prediction")
plt.savefig("face_prediction_output.png") 
print("Prediction saved as face_prediction_output.png")
