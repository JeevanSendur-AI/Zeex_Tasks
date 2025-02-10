
import cv2
from ultralytics import YOLO

model = YOLO('Task 1/yolov8x_atm.pt') 

# Open the camera
camera_index = 0  # webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the processed frame in a single window
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
