import cv2
import face_recognition
import numpy as np
import qrcode
from PIL import Image, ImageDraw
import os
import json
from datetime import datetime

# Directory for saving QR codes and results
OUTPUT_DIR = "biometric_qr_codes"

def capture_picture():
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
            filename = f"captured_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
            cap.release()
            cv2.destroyAllWindows()
            return filename
        
        elif key == ord('q'):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            return None

def generate_biometric_qr(image_path):
    """Generate a biometric QR code based on a provided image."""
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face locations
    print("[INFO] Detecting faces...")
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    if not face_locations:
        print("No faces detected.")
        return

    # Extract face encodings
    print("[INFO] Extracting face encodings...")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    biometric_data = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Prepare biometric data (reduce encoding precision)
        biometric_data.append({
            "face_encoding": [round(num, 4) for num in face_encoding],
            "location": [top, right, bottom, left]
        })

    # Create a subfolder with a human-readable date format
    human_readable_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = os.path.join(OUTPUT_DIR, human_readable_timestamp)
    os.makedirs(result_folder, exist_ok=True)

    # Save biometric data to JSON
    json_path = os.path.join(result_folder, f"biometric_data_{human_readable_timestamp}.json")
    with open(json_path, "w") as json_file:
        json.dump(biometric_data, json_file, indent=4)
    print(f"Biometric data saved to: {json_path}")

    # Generate the QR code with the biometric data
    print("[INFO] Generating QR code...")
    qr = qrcode.QRCode(
        version=None,  # Automatically fit the data
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(json.dumps(biometric_data))  # Embed biometric data directly into the QR code
    qr.make(fit=True)

    qr_img = qr.make_image(fill="black", back_color="white")
    qr_code_path = os.path.join(result_folder, f"biometric_qr_{human_readable_timestamp}.png")
    qr_img.save(qr_code_path)
    print(f"QR code saved: {qr_code_path}")

    # Overlay QR code onto the image
    qr_img = qr_img.resize((150, 150))
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left) in face_locations:
        # Draw rectangle around the face
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
    
    # Paste QR code onto the image
    pil_image.paste(qr_img, (10, 10))
    final_image_path = os.path.join(result_folder, f"final_image_{human_readable_timestamp}.png")
    pil_image.save(final_image_path)
    print(f"Final image with QR code saved: {final_image_path}")


def main():
    print("Choose an option:")
    print("1. Capture a picture")
    print("2. Use an existing picture")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        image_path = capture_picture()
    elif choice == "2":
        image_path = input("Enter the path to the image: ").strip()
        if not os.path.exists(image_path):
            print("File not found.")
            return
    else:
        print("Invalid choice.")
        return

    if image_path:
        generate_biometric_qr(image_path)
        os.remove(image_path)

if __name__ == "__main__":
    main()
