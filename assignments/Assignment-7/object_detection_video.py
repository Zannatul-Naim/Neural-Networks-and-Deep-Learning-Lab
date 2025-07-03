
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo12n.pt")

video_path = "manual_input_video.mp4"
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

out = cv2.VideoWriter("manual_output_video_yolov8.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
# out = cv2.VideoWriter("manual_output_video_yolov11.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
# out = cv2.VideoWriter("manual_output_video_yolov12.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR) 

    out.write(annotated_frame)
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
