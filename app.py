import streamlit as st
import face_recognition
import pickle
import cv2
import numpy as np
import os
from PIL import Image

# --- CONFIGURATION ---
ENCODINGS_FILE = "encodings.pkl"
REGISTERED_FILE = "registered_users.txt"
DETECTION_METHOD = "hog"
TOLERANCE = 0.5
REGISTERED_STATUS = "Registered (Access Granted)"
NOT_REGISTERED_STATUS = "Not Registered (Access Denied)"

# Helper: Load registered names
def load_registered_names(filepath):
    names = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    names.append(name)
    return names

# Core image processing
def process_image(image_np):
    try:
        data = pickle.loads(open(ENCODINGS_FILE, "rb").read())
    except Exception as e:
        st.error(f"Error loading encodings.pkl: {e}")
        return None

    REGISTERED_PERSON_NAMES = load_registered_names(REGISTERED_FILE)
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)
    statuses = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
        identified_name = "Not Found"
        if True in matches:
            matchedIdxs = [i for i, b in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                matched_name = data["names"][i]
                counts[matched_name] = counts.get(matched_name, 0) + 1
            identified_name = max(counts, key=counts.get)

        final_status = REGISTERED_STATUS if identified_name in REGISTERED_PERSON_NAMES else NOT_REGISTERED_STATUS
        statuses.append(final_status)

    # Draw boxes and labels
    for ((top, right, bottom, left), status) in zip(boxes, statuses):
        color = (0, 255, 0) if status == REGISTERED_STATUS else (0, 0, 255)
        cv2.rectangle(image_np, (left, top), (right, bottom), color, 2)
        y = top - 10 if top - 10 > 10 else top + 10
        cv2.putText(image_np, status, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return image_np

# --- STREAMLIT UI ---
st.title("Face Recognition Access System")
st.write("Upload a group photo to detect registered faces and access status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        result_img = process_image(img)
        if result_img is not None:
            # Convert BGR to RGB for display in Streamlit
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Analyzed Image", use_column_width=True)
            st.success("Analysis Complete!")
        else:
            st.error("Failed to process the image.")
    else:
        st.error("Invalid image file.")
