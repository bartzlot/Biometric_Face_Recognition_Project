import face_recognition
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("face_recognition.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Create a directory for saving compared images
compared_dir = "compared_images"
os.makedirs(compared_dir, exist_ok=True)

def capture_picture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open MacBook camera")

    print("Press SPACE to capture photo, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return filename
        
        elif key == ord('q'):  # Q key
            cap.release()
            cv2.destroyAllWindows()
            return None

def select_picture():
    filename = input("Enter the path to the image file: ")
    if os.path.exists(filename):
        return filename
    else:
        print("File not found.")
        return None

def compare_picture(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        percentage = 0.0
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            percentage = (1 - face_distances[best_match_index]) * 100
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name} ({percentage:.2f}%)", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)
        
        # Draw additional information below the face frame
        info_text = f"Name: {name}\nProbability: {percentage:.2f}%\nDistance: {face_distances[best_match_index]:.4f}"
        y0, dy = bottom + 40, 30
        for i, line in enumerate(info_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Save the compared image
    compared_image_path = os.path.join(compared_dir, f"compared_{os.path.basename(image_path)}")
    cv2.imwrite(compared_image_path, image)
    print(f"Compared image saved: {compared_image_path}")
    
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Take a picture")
    print("2. Select a picture")
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        image_path = capture_picture()
    elif choice == '2':
        image_path = select_picture()
    else:
        print("Invalid choice.")
        image_path = None
    
    if image_path:
        compare_picture(image_path)