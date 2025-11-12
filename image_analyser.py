import face_recognition
import pickle
import cv2
import os
import argparse
import numpy as np

# --- Configuration ---
ENCODINGS_FILE = "encodings.pkl"
REGISTERED_FILE = "registered_users.txt"
DETECTION_METHOD = "hog" 
TOLERANCE = 0.5 # Lower this number (e.g., 0.4) for stricter matching
REGISTERED_STATUS = "Registered"
NOT_REGISTERED_STATUS = "Not Registered"

# Function to load registered names from the text file
def load_registered_names(filepath):
    """Reads names from a text file, one name per line, and cleans the list."""
    names = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith('#'):
                    names.append(name)
    return names

def analyze_image(image_path):
    # 1. Load Data
    print("[INFO] Loading encodings and registered users...")
    try:
        data = pickle.loads(open(ENCODINGS_FILE, "rb").read())
    except FileNotFoundError:
        print(f"[ERROR] Encodings file not found: {ENCODINGS_FILE}. Run encode_faces.py first!")
        return

    REGISTERED_PERSON_NAMES = load_registered_names(REGISTERED_FILE)
    if not REGISTERED_PERSON_NAMES:
        print("[WARNING] No registered users found in text file. All faces will be 'Not Registered'.")

    # 2. Load Input Image
    print(f"[INFO] Analyzing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image at path: {image_path}. Check file path and format.")
        return

    # Convert BGR (OpenCV) to RGB (dlib/face_recognition)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Detect ALL Faces (Crucial for Group Pics)
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    print(f"[INFO] Found {len(boxes)} face(s) in the image.")

    statuses = []

    # 4. Analyze & Classify Each Face
    for encoding in encodings:
        
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
        identified_name = "Not Found" 

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                matched_name = data["names"][i]
                counts[matched_name] = counts.get(matched_name, 0) + 1

            identified_name = max(counts, key=counts.get)
            
        # Determine final status based on the Registered Users List
        if identified_name in REGISTERED_PERSON_NAMES:
            final_status = REGISTERED_STATUS
        else:
            final_status = NOT_REGISTERED_STATUS
        
        statuses.append(final_status)

    # 5. Output: Draw Results on the Image
    for ((top, right, bottom, left), status) in zip(boxes, statuses):
        # Determine color (Green for Registered, Red for Not Registered)
        color = (0, 255, 0) if status == REGISTERED_STATUS else (0, 0, 255) 
        
        # Draw bounding box
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        # Draw status label
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, status, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    

    # Display the analyzed image
    cv2.imshow("Image Analyzer Result", image)
    # --- Option to save the image ---
    # cv2.imwrite("analyzed_result.jpg", image) 
    # -------------------------------
    cv2.waitKey(0) # Wait until a key is pressed to close the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Setup command line argument to accept the image path
    parser = argparse.ArgumentParser(description="Analyze a group image for registered faces.")
    # The 'image' argument is REQUIRED and is the path to the photo.
    parser.add_argument("image", help="Path to the input image (e.g., group_photo.jpg)")
    args = parser.parse_args()
    
    analyze_image(args.image)
    