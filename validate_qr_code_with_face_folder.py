import cv2
import face_recognition
import json
import numpy as np
from datetime import datetime
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
import pyzxing

# Directory for results
results_dir = "qr_code_validation_results"
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

def compare_faces(biometric_data, image_path, threshold):
    """Compare the face in the image with the biometric data."""
    captured_image = cv2.imread(image_path)
    rgb_captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

    # Detect faces and extract encodings
    print(f"[INFO] Detecting faces in image {image_path}...")
    captured_face_locations = face_recognition.face_locations(rgb_captured_image, model="hog")
    captured_face_encodings = face_recognition.face_encodings(rgb_captured_image, captured_face_locations)

    if not captured_face_encodings:
        print("[ERROR] No faces detected in the image.")
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
            text = f"No match found\nThreshold: {threshold}\nProbability: {probability:.2f}%\nDistance: {distance:.2f}"
            draw.text((left, bottom + 10), text, fill="red", font=font)

    return pil_image, match_found

def process_folder(biometric_data, folder_path, threshold, run_results_dir):
    """Process all images in the folder and save results in a single subfolder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            pil_image, match_found = compare_faces(biometric_data, image_path, threshold)

            result_image_path = os.path.join(run_results_dir, f"result_{os.path.basename(image_path)}")
            pil_image.save(result_image_path)
            print(f"[INFO] Result saved to {result_image_path}")

def main():
    qr_code_path = input("Enter the path to the QR code image file: ").strip()
    folder_path = input("Enter the path to the folder containing images: ").strip()
    threshold = float(input("Enter the face recognition threshold (e.g., 0.6): ").strip())

    try:
        print("[INFO] Detecting QR code in provided image...")
        qr_data = detect_qr_code(qr_code_path)

        print("[INFO] Loading biometric data from QR code...")
        biometric_data = load_biometric_data_from_qr(qr_data)

        # Create a subfolder for this run's results
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_results_dir = os.path.join(results_dir, f"run_{run_timestamp}")
        os.makedirs(run_results_dir, exist_ok=True)

        print("[INFO] Comparing images in the folder with biometric data...")
        process_folder(biometric_data, folder_path, threshold, run_results_dir)

    except Exception as e:
        print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()