import cv2
import requests
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import torch
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Limit GPU memory usage to 90%
if device == "cuda":
    torch.cuda.empty_cache()  # Clear unused memory
    torch.backends.cudnn.benchmark = True  # Optimized training
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)

# Load models
model1 = YOLO(r"Task 2\anomaly_classifier.pt")  # yolo11 classifier(normal, anomlous)
# model2 = YOLO(r"Task 1\yolov8x_atm.pt")  # custom yolov8 object detector
model2 = YOLO(r'Task 3\knife,pistol,rifle.pt') #smaller model(yolo11n) for faster inference,but different classes

import firebase_admin
from firebase_admin import credentials, firestore

firebase_json = r"Task 2\task2-7eeef-firebase-adminsdk-fbsvc-eec570cf25.json"

try:
    cred = credentials.Certificate(firebase_json)
    firebase_admin.initialize_app(cred, {
        'projectId': 'task2-7eeef',  
        # 'databaseURL': 'https://task2-7eeef.firebaseio.com' 
    })
    db = firestore.client()
    print("[INFO] Firebase Initialized Successfully")
except Exception as e:
    print(f"[ERROR] Firebase Initialization Failed: {e}")
    exit(1)


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
    a = byte_stream.find(b'\xff\xd8')  
    b = byte_stream.find(b'\xff\xd9')  
    
    if a != -1 and b != -1:
        jpg = byte_stream[a:b+2]  # Extract the JPEG frame
        byte_stream = byte_stream[b+2:]  # Remove processed bytes
        
        # Convert to NumPy array and decode
        frame_bgr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame_bgr is None:
            print("Error: Couldn't decode frame")
            continue

        # Convert frame from BGR (OpenCV default) to RGB (YOLO expects this)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image for YOLOv8
        image = Image.fromarray(frame_rgb)

        # Perform inference on both models
        results1 = model1(image)
        detected_objects1 = results1[0].probs.data[1].item()
        if detected_objects1 < 0.8:
            detected_objects1 = None
        else:
            detected_objects1 = 1
        image = results1[0].plot()
        results2 = model2(image)
        detected_objects2 = [model2.names[int(box.cls)] for box in results2[0].boxes if box.conf > 0.2]
        model2_frame = results2[0].plot()

        if detected_objects1 == 1 and detected_objects2 != None:
            log_to_firebase(list(set(detected_objects2)))

        # Display the annotated frame
        cv2.imshow("Live Video Stream", model2_frame)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

cv2.destroyAllWindows()
