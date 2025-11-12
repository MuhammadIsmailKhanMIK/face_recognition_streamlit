import streamlit as st
import face_recognition
import pickle
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- Configuration ---
UPLOAD_FOLDER = "static/uploads"
ENCODINGS_FILE = "encodings.pkl"
REGISTERED_FILE = "registered_users.txt"
DETECTION_METHOD = "hog"
TOLERANCE = 0.5
REGISTERED_STATUS = "Registered (Access Granted)"
NOT_REGISTERED_STATUS = "Not Registered (Access Denied)"

# Helper function to load registered names
def load_registered_names(filepath):
    names = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    names.append(name)
    return names

# Process image
def process_image(image_path):
    # Load encodings
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading encodings: {e}")
        return None

    REGISTERED_PERSON_NAMES = load_registered_names(REGISTERED_FILE)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        st.error("Failed to load image")
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)
    statuses = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
        identified_name = "Not Found"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            identified_name = max(counts, key=counts.get)
        final_status = REGISTERED_STATUS if identified_name in REGISTERED_PERSON_NAMES else NOT_REGISTERED_STATUS
        statuses.append(final_status)

    # Draw results
    for ((top, right, bottom, left), status) in zip(boxes, statuses):
        color = (0, 255, 0) if status == REGISTERED_STATUS else (255, 0, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        y = top - 10 if top - 10 > 10 else top + 10
        cv2.putText(image, status, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Streamlit UI ---
st.title("Face Recognition App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    processed_image = process_image(tmp_path)
    if processed_image is not None:
        st.image(processed_image, caption="Processed Image", use_column_width=True)
