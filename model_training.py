import os
from imutils import paths
import face_recognition
import pickle
import cv2

# Configuration
dataset_dir = "dataset"
processed_dir = "model_processed_images"
os.makedirs(processed_dir, exist_ok=True)

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images(dataset_dir))
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
    
    # Save processed image
    for (top, right, bottom, left) in boxes:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    processed_image_path = os.path.join(processed_dir, f"{name}_{i + 1}.jpg")
    cv2.imwrite(processed_image_path, image)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("face_recognition.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")