import cv2
import requests
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 models
model1 = YOLO("Task 1\yolov8x_atm.pt")  # Replace with actual model path
model2 = YOLO("Task 1\yolov8x_atm.pt")  # Replace with actual model path

# Initialize Firebase
cred = credentials.Certificate(r"Task 2\task2-7eeef-firebase-adminsdk-fbsvc-eec570cf25.json")  # Replace with your Firebase credentials file
firebase_admin.initialize_app(cred)
db = firestore.client()

# URL of the video stream from the Flask server
stream_url = "http://192.168.174.93:5000/video"

# Open the video stream
cap = requests.get(stream_url, stream=True)
byte_stream = b""

def log_to_firebase(detected_objects):
    """Logs an incident to Firebase Firestore."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": timestamp,
        "description": f"Detected: {', '.join(detected_objects)}",
    }
    db.collection("incidents").add(data)
    print(f"[LOGGED] {timestamp} - {data['description']}")

for chunk in cap.iter_content(chunk_size=1024):
    byte_stream += chunk
    a = byte_stream.find(b'\xff\xd8')  # Start of JPEG frame
    b = byte_stream.find(b'\xff\xd9')  # End of JPEG frame
    
    if a != -1 and b != -1:
        jpg = byte_stream[a:b+2]  # Extract the JPEG frame
        byte_stream = byte_stream[b+2:]  # Remove processed bytes
        
        # Convert to NumPy array and decode
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Perform inference on both models
        results1 = model1(frame)
        results2 = model2(frame)

        detected_objects1 = [f"{model1.names[int(box.cls)]} ({box.conf:.2f})" for box in results1[0].boxes if box.conf > 0.8]
        detected_objects2 = [f"{model2.names[int(box.cls)]} ({box.conf:.2f})" for box in results2[0].boxes if box.conf > 0.8]

        # If both models detect objects with confidence > 0.8, log incident
        if detected_objects1 and detected_objects2:
            log_to_firebase(detected_objects1 + detected_objects2)

        # Display the frame
        cv2.imshow("Live Video Stream", frame)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

cv2.destroyAllWindows()
