import cv2
import face_recognition
import json
import numpy as np
from datetime import datetime
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import pyzxing

# Directory for temporary files and results
temp_dir = "temp_validation"
results_dir = "qr_code_validation_results"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def detect_qr_code(image_path):
    """Detect and decode QR code in the provided image using pyzxing."""
    reader = pyzxing.BarCodeReader()
    result = reader.decode(image_path)
    if result and result[0].get('parsed'):
        data = result[0]['parsed']
        print(f"[INFO] QR code detected: {data}")
        return data
    else:
        raise ValueError("No QR code found in the image.")

def load_biometric_data_from_qr(data):
    """Load biometric data from the decoded QR code data."""
    biometric_data = json.loads(data)
    print(f"[INFO] Loaded biometric data from QR code: {biometric_data}")
    return biometric_data

def capture_face_picture():
    """Capture a picture using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    print("Press SPACE to capture photo, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space key to capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(temp_dir, f"captured_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Photo saved: {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return image_path

        elif key == ord('q'):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            return None

def compare_faces(biometric_data, captured_image_path, threshold):
    """Compare the face in the captured image with the biometric data."""
    captured_image = cv2.imread(captured_image_path)
    rgb_captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

    # Detect faces and extract encodings
    print("[INFO] Detecting faces in captured image...")
    captured_face_locations = face_recognition.face_locations(rgb_captured_image, model="hog")
    captured_face_encodings = face_recognition.face_encodings(rgb_captured_image, captured_face_locations)

    if not captured_face_encodings:
        print("[ERROR] No faces detected in the captured image.")
        return False

    pil_image = Image.fromarray(rgb_captured_image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    match_found = False

    # Compare each captured face encoding with the saved biometric encodings
    for captured_encoding, (top, right, bottom, left) in zip(captured_face_encodings, captured_face_locations):
        for biometric_entry in biometric_data:
            known_encoding = np.array(biometric_entry["face_encoding"])
            match = face_recognition.compare_faces([known_encoding], captured_encoding, tolerance=threshold)
            distance = face_recognition.face_distance([known_encoding], captured_encoding)[0]
            probability = (1 - distance) * 100
            if match[0]:
                draw.rectangle([(left, top), (right, bottom)], outline="green", width=3)
                text = f"Match: Probability: {probability:.2f}%\nThreshold: {threshold}\nDistance: {distance:.2f}"
                draw.text((left, bottom + 10), text, fill="lightgreen", font=font)
                match_found = True
                break

    if not match_found:
        for (top, right, bottom, left) in captured_face_locations:
            draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
            text = f"No match found\nThreshold: {threshold}"
            draw.text((left, bottom + 10), text, fill="red", font=font)

    result_image_path = os.path.join(results_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    pil_image.save(result_image_path)
    pil_image.show()
    print(f"[INFO] Result saved to {result_image_path}")

    return match_found

def main():
    qr_code_path = input("Enter the path to the QR code image file: ").strip()
    threshold = float(input("Enter the face recognition threshold (e.g., 0.6): ").strip())

    try:
        print("[INFO] Detecting QR code in provided image...")
        qr_data = detect_qr_code(qr_code_path)

        print("[INFO] Loading biometric data from QR code...")
        biometric_data = load_biometric_data_from_qr(qr_data)

        print("[INFO] Capturing picture for validation...")
        captured_image_path = capture_face_picture()
        if not captured_image_path:
            print("[ERROR] Picture capture aborted.")
            return

        print("[INFO] Comparing captured image with biometric data...")
        is_valid = compare_faces(biometric_data, captured_image_path, threshold)

        if is_valid:
            print("[SUCCESS] The person matches the biometric data.")
        else:
            print("[FAILURE] The person does not match the biometric data.")

    except Exception as e:
        print(f"[ERROR] {str(e)}")

    finally:
        # Delete the temp_validation folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"[INFO] Temporary directory {temp_dir} deleted.")

if __name__ == "__main__":
    main()