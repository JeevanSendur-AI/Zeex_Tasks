import cv2
import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# ---------------------------- Configuration ---------------------------- #
MODEL_PATH = "Task 3\knife,pistol,rifle.pt"  # Path to your YOLO model
DATASET_PATH = "dataset/"  # Folder to store images and labels
CONFIDENCE_THRESHOLD = 0.2  # Initial confidence threshold

# Ensure dataset directories exist
os.makedirs(f"{DATASET_PATH}/images", exist_ok=True)
os.makedirs(f"{DATASET_PATH}/labels", exist_ok=True)

# Load YOLOv11 model
model = YOLO(MODEL_PATH)

CLASS_NAMES = {0: "Knife", 1: "Pistol", 2: "Rifle"}  



# ---------------------------- Video Frame Extraction ---------------------------- #
def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(1, frame_rate // fps)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            frames.append(frame)
        frame_id += 1

    cap.release()
    return frames

# ---------------------------- Image Processing ---------------------------- #
def run_inference(image):
    results = model(image)  
    parsed_results = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  
        confs = r.boxes.conf.cpu().numpy()  
        classes = r.boxes.cls.cpu().numpy()  

        for i in range(len(boxes)):
            if confs[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = boxes[i]
                parsed_results.append({
                    "xmin": int(x1),
                    "ymin": int(y1),
                    "xmax": int(x2),
                    "ymax": int(y2),
                    "class": int(classes[i]),
                    "class_name": CLASS_NAMES.get(int(classes[i]), "Unknown"),
                    "confidence": float(confs[i])
                })

    return parsed_results  

# ---------------------------- GUI Class ---------------------------- #
class BoundingBoxReviewer:
    def __init__(self, images):
        self.root = tk.Tk()
        self.root.title("Bounding Box Feedback")
        self.images = images
        self.current_idx = 0
        self.annotations = []

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.submit_button = tk.Button(self.root, text="Submit Feedback", command=self.save_annotations)
        self.report_missing_button = tk.Button(self.root, text="Report Missing Object", command=self.report_missing)

        self.prev_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.RIGHT)
        self.submit_button.pack(side=tk.BOTTOM)
        self.report_missing_button.pack(side=tk.BOTTOM)

        self.load_image()
        self.root.mainloop()

    def load_image(self):
        self.canvas.delete("all")
        image_path = self.images[self.current_idx]
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Run inference
        self.results = run_inference(image_path)
        self.bbox_feedback = {}

        for i, result in enumerate(self.results):
            x1, y1, x2, y2 = result["xmin"], result["ymin"], result["xmax"], result["ymax"]
            predicted_class = result["class"]
            class_name = result["class_name"]

            bbox_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
            self.canvas.create_text(x1, y1 - 10, text=f"{class_name} ({predicted_class})", fill="red", anchor=tk.W)

            self.bbox_feedback[bbox_id] = tk.IntVar(value=0)
            options = [
                (f"Correct ({class_name})", 1),
                (f"Should be {CLASS_NAMES[(predicted_class + 1) % 3]}", 2),
                (f"Should be {CLASS_NAMES[(predicted_class + 2) % 3]}", 3),
                ("Wrong (Background)", 4),
                ("Incorrect BBox", 5)
            ]

            frame = tk.Frame(self.root)
            for text, val in options:
                tk.Radiobutton(frame, text=text, variable=self.bbox_feedback[bbox_id], value=val).pack(side=tk.LEFT)
            frame.pack()

    def report_missing(self):
        x1 = simpledialog.askinteger("Bounding Box", "Enter x1:")
        y1 = simpledialog.askinteger("Bounding Box", "Enter y1:")
        x2 = simpledialog.askinteger("Bounding Box", "Enter x2:")
        y2 = simpledialog.askinteger("Bounding Box", "Enter y2:")
        class_id = simpledialog.askinteger("Class", "Enter class ID (0,1,2):")

        if None in (x1, y1, x2, y2, class_id):
            return

        self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2)
        self.annotations.append({
            "image": self.images[self.current_idx],
            "bbox": [x1, y1, x2, y2],
            "correct_class": class_id
        })

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def next_image(self):
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_image()

    def save_annotations(self):
        image_name = os.path.basename(self.images[self.current_idx])
        label_path = f"{DATASET_PATH}/labels/{image_name.replace('.jpg', '.txt')}"

        with open(label_path, "w") as label_file:
            for result in self.results:
                x_center = (result["xmin"] + result["xmax"]) / 2 / self.pil_image.width
                y_center = (result["ymin"] + result["ymax"]) / 2 / self.pil_image.height
                width = (result["xmax"] - result["xmin"]) / self.pil_image.width
                height = (result["ymax"] - result["ymin"]) / self.pil_image.height
                class_id = int(result["class"])

                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        for annotation in self.annotations:
            with open(label_path, "a") as label_file:
                x1, y1, x2, y2 = annotation["bbox"]
                x_center = (x1 + x2) / 2 / self.pil_image.width
                y_center = (y1 + y2) / 2 / self.pil_image.height
                width = (x2 - x1) / self.pil_image.width
                height = (y2 - y1) / self.pil_image.height
                label_file.write(f"{annotation['correct_class']} {x_center} {y_center} {width} {height}\n")

        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD += 0.05  

        messagebox.showinfo("Saved", "Feedback saved successfully!")

# ---------------------------- Run GUI ---------------------------- #
if __name__ == "__main__":
    file_path = filedialog.askopenfilename(title="Select Image or Video")
    images = [file_path] if file_path.endswith((".jpg", ".png")) else extract_frames(file_path)
    BoundingBoxReviewer(images)
