import os
import csv
import cv2
import numpy as np
import torch
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# --- Attendance setup (minimal additions) ---
attendance_folder = os.path.join(os.getcwd(), "attendence_sheet")
os.makedirs(attendance_folder, exist_ok=True)

def get_attendance_file_for_today():
    """Return full path for today's attendance CSV (one file per day)."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(attendance_folder, f"attendance_{date_str}.csv")

def read_names_from_file(att_file):
    """Return a set of names already present in the given attendance file."""
    names = set()
    if os.path.isfile(att_file):
        try:
            with open(att_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # skip empty rows or malformed lines
                    if 'Name' in row and row['Name'].strip():
                        names.add(row['Name'].strip())
        except Exception:
            # If file is corrupted or unreadable, treat as empty
            pass
    return names

def log_attendance(name, logged_names_set):
    """
    Append name + date + time to today's CSV if not already present.
    logged_names_set is the per-run memory (to avoid repeated writes during a run).
    """
    # normalize name string
    name = str(name).strip()
    if not name or name == 'Not Recognized':
        return False

    att_file = get_attendance_file_for_today()
    today = datetime.now()
    date = today.strftime("%Y-%m-%d")
    time = today.strftime("%H:%M:%S")

    # If already logged in this run, skip
    if name in logged_names_set:
        return False

    # Check file for existing name (so across restarts duplicates are prevented)
    existing_names = read_names_from_file(att_file)
    if name in existing_names:
        # mark in-run to avoid re-checking
        logged_names_set.add(name)
        return False

    # Write header if file doesn't exist
    file_exists = os.path.isfile(att_file)
    with open(att_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date, time])

    # Remember that we've logged this name in this run
    logged_names_set.add(name)
    return True

# Keep track of who was already logged this run (avoid duplicate lines per run)
logged_names = set()
# store current date to clear logged_names when day changes during a long run
current_date = datetime.now().strftime("%Y-%m-%d")
# ------------------------------------------------

# Function to detect and encode faces
# --- Auto-load known faces from Images folder (minimal change) ---
# Try the project-specific folder first, otherwise use ./Images

def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode specific images with predefined names
def encode_known_faces(known_faces):
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        known_image = cv2.imread(image_path)
        if known_image is not None:
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.append(encodings[0])  # Assuming one face per image
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Define known faces with explicit names
# --- Auto-load known faces from Images folder (minimal change) ---
# Try the project-specific folder first, otherwise use ./Images

images_folder = os.path.join(os.getcwd(), "images")  # default

# Supported image extensions
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Build known_faces dict: filename (no ext) -> full path
known_faces = {}
for fname in os.listdir(images_folder):
    name, ext = os.path.splitext(fname)
    if ext.lower() in img_exts:
        # normalize name (strip whitespace)
        person_name = name.strip()
        if person_name:  # skip empty names
            full_path = os.path.join(images_folder, fname)
            # avoid overwriting if duplicate filenames exist
            if person_name in known_faces:
                # if duplicate, append an index to name to keep both
                idx = 1
                new_name = f"{person_name}_{idx}"
                while new_name in known_faces:
                    idx += 1
                    new_name = f"{person_name}_{idx}"
                person_name = new_name
            known_faces[person_name] = full_path

# Encode known faces (unchanged usage)
known_face_encodings, known_face_names = encode_known_faces(known_faces)
# ------------------------------------------------------------------

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# Start video capture
cap = cv2.VideoCapture(0)
threshold = 0.6

while cap.isOpened():
    # --- daily rollover handling: clear in-run logged_names when date changes ---
    now_date = datetime.now().strftime("%Y-%m-%d")
    if now_date != current_date:
        logged_names.clear()
        current_date = now_date
    # ------------------------------------------------------------------------

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)

    if test_face_encodings and known_face_encodings:
        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)
        boxes = mtcnn.detect(frame_rgb)[0]  # reuse detection for drawing
        for name, box in zip(names, boxes if boxes is not None else []):
            if box is not None:
                (x1, y1, x2, y2) = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Attendance logging: write once per day (and once per run)
                log_attendance(name, logged_names)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
