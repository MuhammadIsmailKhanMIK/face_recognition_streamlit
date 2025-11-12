# Imports for file/directory handling and face recognition
from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time

# --- Configuration ---
# '.' refers to the current directory where the script is run (E:\1st Task)
DATASET_PATH = "." 
# The path to the file where the face encodings will be stored
ENCODINGS_FILE = "encodings.pkl"
# Method to use for face detection (hog is faster on CPU)
DETECTION_METHOD = "hog" 

# === Key Update: Define all acceptable image extensions in lowercase ===
# This list is used to find all common image types (.jpg, .jpeg, .png, etc.), regardless of case.
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# =====================================================================


print("[INFO] Starting face quantification...")
start_time = time.time()

# Initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# --- UPDATED LOGIC TO HANDLE MULTIPLE EXTENSIONS ---
# Use list_files to get a recursive list of all files and then filter by extension.
imagePaths = []
for imagePath in paths.list_files(DATASET_PATH):
    # Check if the file ends with any of the valid extensions (case-insensitively)
    if imagePath.lower().endswith(VALID_IMAGE_EXTENSIONS):
        # We must also exclude the Group Images folder if it contains mixed images of the registered people.
        # However, for now, we will assume all folders are for registration.
        # If 'Group Images' contains images of the people you registered, you can leave it.
        if "group images" not in imagePath.lower():
            imagePaths.append(imagePath)
# ----------------------------------------------------------------------


# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path (e.g., "Muhammad Ismail")
    # This assumes the structure is DATASET_PATH/PERSON_NAME/IMAGE_FILE
    name = imagePath.split(os.path.sep)[-2]
    
    # Process image
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}: {name} ({imagePath.split(os.path.sep)[-1]})")

    # Load the input image and convert it from BGR (OpenCV ordering) to RGB (dlib ordering)
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not load image: {imagePath}. Skipping.")
        continue
        
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the coordinates of the faces in the input image
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)

    # Compute the 128-d face embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings (if multiple faces are found, it takes the encoding for the first face)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

end_time = time.time()
print(f"[INFO] Face quantification complete. Time taken: {end_time - start_time:.2f} seconds.")


# --- Serialization ---
# Dump the facial encodings and names to disk
print(f"[INFO] Serializing {len(knownEncodings)} encodings to disk...")
data = {"encodings": knownEncodings, "names": knownNames}

with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))
    
print(f"[INFO] Encodings saved as {ENCODINGS_FILE}")