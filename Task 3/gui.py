import tkinter as tk
from tkinter import messagebox
import json

# Sample detections (Normally, this would come from your YOLOv8 model)
detections = [
    {"id": 1, "object": "knife", "confidence": 0.85},
    {"id": 2, "object": "helmet", "confidence": 0.92}
]

# Confidence thresholds dictionary
confidence_thresholds = {"knife": 0.8, "helmet": 0.8}
feedback_log = []

def update_confidence_threshold(obj, correct):
    if correct:
        confidence_thresholds[obj] += 0.02  # Increase confidence threshold
    else:
        confidence_thresholds[obj] -= 0.02  # Decrease confidence threshold
    confidence_thresholds[obj] = max(0.5, min(0.95, confidence_thresholds[obj]))
    

def submit_feedback(detection_id, correct):
    global feedback_log
    detection = next((d for d in detections if d["id"] == detection_id), None)
    
    if detection:
        obj = detection["object"]
        update_confidence_threshold(obj, correct)
        feedback_log.append({"id": detection_id, "object": obj, "correct": correct})
        messagebox.showinfo("Feedback Received", f"Updated threshold for {obj}: {confidence_thresholds[obj]:.2f}")
    
    print("Updated thresholds:", json.dumps(confidence_thresholds, indent=2))
    

def create_gui():
    root = tk.Tk()
    root.title("Object Detection Feedback")
    
    tk.Label(root, text="Review Detection Results", font=("Arial", 14, "bold")).pack(pady=10)
    
    for detection in detections:
        frame = tk.Frame(root, pady=5)
        frame.pack()
        
        tk.Label(frame, text=f"Detected: {detection['object']} (Confidence: {detection['confidence']:.2f})", font=("Arial", 12)).pack(side=tk.LEFT)
        
        tk.Button(frame, text="Correct", command=lambda d=detection["id"]: submit_feedback(d, True)).pack(side=tk.RIGHT)
        tk.Button(frame, text="Incorrect", command=lambda d=detection["id"]: submit_feedback(d, False)).pack(side=tk.RIGHT)
    
    root.mainloop()

create_gui()
