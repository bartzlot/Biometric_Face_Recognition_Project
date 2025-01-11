# Biometric Face Recognition Project

This project is built using Python 3.10.16. Follow the instructions below to install the necessary dependencies.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/bartzlot/Biometric_Face_Recognition_Project.git
    cd Biometric_Face_Recognition_Project
    ```

2. **Create a virtual environment:**

    ```bash
    python3.10 -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Make sure to activate the virtual environment before running any scripts:

```bash
source venv/bin/activate
```

### Capturing Images

The `image_capture.py` script allows you to capture images using your computer's camera and save them in a structured folder format. Follow the steps below to use the script:

1. **Run the script:**

    ```bash
    python image_capture.py
    ```

2. **Enter the name of the person:**

    When prompted, enter the name of the person for whom you want to capture images. This will create a folder with the person's name inside the `dataset` directory.

3. **Capture photos:**

    - Press the `SPACE` key to capture a photo.
    - Press the `q` key to quit the photo capture session.

The captured photos will be saved in the `dataset/<person_name>` folder with a timestamp in the filename.

### Example

```bash
python image_capture.py
```

When prompted, enter the name:

```plaintext
Enter the name of the person: JohnDoe
```

Press `SPACE` to capture photos and `q` to quit. The photos will be saved in the `dataset/JohnDoe` directory.

### Training Model

The `model_training.py` script processes a dataset of images to extract facial encodings and save them for later use in face recognition tasks. It uses the `face_recognition` library to detect faces and compute encodings, and the `cv2` library to handle image processing.

1. **Configuration:**
    - `dataset_dir`: Directory containing the dataset of images.
    - `processed_dir`: Directory where processed images with detected faces will be saved.

2. **Processing Faces:**
    - The script iterates over all images in the dataset directory.
    - For each image, it extracts the name of the person from the directory structure.
    - It reads the image using `cv2.imread` and converts it to RGB format.
    - It detects face locations in the image using `face_recognition.face_locations`.
    - It computes facial encodings for each detected face using `face_recognition.face_encodings`.
    - The encodings and corresponding names are stored in lists `knownEncodings` and `knownNames`.

3. **Saving Processed Images:**
    - For each detected face, a rectangle is drawn around the face in the image.
    - The processed image is saved to the `processed_dir` with a filename indicating the person's name and image index.

4. **Serializing Encodings:**
    - The script creates a dictionary `data` containing the encodings and names.
    - It serializes this dictionary to a pickle file named `face_recognition.pickle` using the `pickle` module.

### Pickle Files

- Pickle files are used to serialize and deserialize Python objects. In this script, the facial encodings and names are serialized to a pickle file for later use.
- The structure of the pickle file is a dictionary with two keys:
  - `encodings`: A list of facial encodings.
  - `names`: A list of names corresponding to the encodings.

### How `cv2` Takes Biometric Data from Photos

- The `cv2` library is used to read images and convert them to the RGB format required by the `face_recognition` library.
- It also provides functionality to draw rectangles around detected faces and save the processed images.

### Live Facial Recognition

The `live_facial_recognition.py` script performs real-time facial recognition using a webcam. It leverages pre-trained facial encodings to identify faces in the video feed.

#### Script Overview

1. **Loading Encodings:**
    - The script loads pre-trained face encodings from a pickle file named `face_recognition.pickle`.

2. **Initializing the Camera:**
    - The script initializes the webcam for capturing video frames.

3. **Setting the Threshold:**
    - The user is prompted to input a threshold value for face recognition. This value determines the tolerance for matching faces.

4. **Processing Frames:**
    - Each frame from the webcam is resized and converted to RGB format.
    - The script detects face locations and computes face encodings for each detected face.
    - It compares the computed encodings with the known encodings to identify faces.

5. **Drawing Results:**
    - The script draws rectangles around detected faces and labels them with names and recognition probabilities.
    - Additional information, such as the threshold, name, probability, and distance, is displayed below each face.

6. **Displaying Video Feed:**
    - The processed frames with drawn results are displayed in a window.
    - The script runs in a loop until the user presses the `q` key to quit.

#### Usage

1. **Run the script:**

    ```bash
    python live_facial_recognition.py
    ```

2. **Enter the threshold value:**

    When prompted, enter a threshold value for face recognition (e.g., `0.6`).

3. **View the video feed:**

    The webcam feed will be displayed with detected faces labeled with names and recognition probabilities.

4. **Quit the script:**

    Press the `q` key to quit the video feed and stop the script.

#### Example

```bash
python live_facial_recognition.py
```

When prompted, enter the threshold:

```plaintext
Enter the threshold for face recognition (e.g., 0.6): 0.6
```

The video feed will display detected faces with labels and additional information.

#### Notes

- Ensure that the `face_recognition.pickle` file is present in the same directory as the script.
- Adjust the `cv_scaler` value to balance performance and accuracy. A lower value increases accuracy but decreases performance.
- The threshold value determines the strictness of face matching. A lower value means stricter matching.

### Facial Picture Recognition

The `facial_picture_recognition.py` script allows you to capture a picture using your webcam or select an existing picture, and then compare the face(s) in the picture with pre-trained face encodings to identify known individuals.

#### Script Overview

1. **Loading Encodings:**
    - The script loads pre-trained face encodings from a pickle file named `face_recognition.pickle`.

2. **Capturing or Selecting a Picture:**
    - You can either capture a new picture using your webcam or select an existing picture file from your filesystem.

3. **Comparing Faces:**
    - The script detects faces in the selected or captured picture.
    - It computes face encodings for each detected face and compares them with the known face encodings.
    - If a match is found, the face is labeled with the corresponding name and recognition probability.

4. **Saving Compared Images:**
    - The script saves the image with drawn rectangles and labels around detected faces in a directory named `compared_images`.

#### Usage

1. **Run the script:**

    ```bash
    python facial_picture_recognition.py
    ```

2. **Choose an option:**
    - Enter `1` to capture a new picture using your webcam.
    - Enter `2` to select an existing picture file.

3. **Capture or select a picture:**
    - If you chose to capture a picture, press the `SPACE` key to take a photo or `q` to quit.
    - If you chose to select a picture, enter the path to the image file when prompted.

4. **View the results:**
    - The script will display the image with detected faces labeled with names and recognition probabilities.
    - The compared image will be saved in the `compared_images` directory.

#### Example

```bash
python facial_picture_recognition.py
```

When prompted, choose an option:

```plaintext
Choose an option:
1. Take a picture
2. Select a picture
Enter 1 or 2: 1
```

If you chose to capture a picture, press `SPACE` to take a photo or `q` to quit. The results will be displayed and saved in the `compared_images` directory.

#### Notes

- Ensure that the `face_recognition.pickle` file is present in the same directory as the script.
- The script uses the `cv2` library to handle image capture and processing.
- The `face_recognition` library is used to detect faces and compute face encodings.
- The script saves the compared images with drawn rectangles and labels in the `compared_images` directory.
- The recognition probability is displayed along with the name of the identified person.

### Creating Biometric QR Codes

The `create_biometric_qr_code.py` script captures a picture using your webcam or uses an existing picture to generate a QR code containing biometric data. This QR code can be used for various biometric identification purposes.

#### Script Overview

1. **Capture or Select a Picture:**
    - You can either capture a new picture using your webcam or select an existing picture file from your filesystem.

2. **Detecting Faces:**
    - The script uses the `face_recognition` library to detect faces in the image.

3. **Extracting Face Encodings:**
    - It computes face encodings for each detected face, which are unique numerical representations of the facial features.

4. **Generating Biometric Data:**
    - The face encodings and face locations are stored in a JSON file.

5. **Creating a QR Code:**
    - The biometric data is embedded into a QR code, which is saved as an image file.

6. **Overlaying QR Code on Image:**
    - The QR code is overlaid onto the original image, and the final image is saved.

#### Usage

1. **Run the script:**

    ```bash
    python create_biometric_qr_code.py
    ```

2. **Choose an option:**
    - Enter `1` to capture a new picture using your webcam.
    - Enter `2` to select an existing picture file.

3. **Capture or select a picture:**
    - If you chose to capture a picture, press the `SPACE` key to take a photo or `q` to quit.
    - If you chose to select a picture, enter the path to the image file when prompted.

4. **View the results:**
    - The script will generate a QR code with the biometric data and overlay it on the image.
    - The results will be saved in the `biometric_qr_codes` directory.

#### Example

```bash
python create_biometric_qr_code.py
```

When prompted, choose an option:

```plaintext
Choose an option:
1. Capture a picture
2. Use an existing picture
Enter 1 or 2: 1
```

If you chose to capture a picture, press `SPACE` to take a photo or `q` to quit. The results will be displayed and saved in the `biometric_qr_codes` directory.

#### How Biometric Data is Taken from the Picture

- The script uses the `cv2` library to capture or read an image.
- The image is converted to RGB format, which is required by the `face_recognition` library.
- The `face_recognition.face_locations` function detects the locations of faces in the image.
- The `face_recognition.face_encodings` function computes the face encodings for each detected face.
- These encodings are numerical representations of the facial features and are stored in a JSON file.

#### Conversion to QR Code

- The biometric data (face encodings and locations) is serialized to a JSON format.
- The `qrcode` library is used to generate a QR code containing the serialized biometric data.
- The QR code is saved as an image file and overlaid onto the original image using the `PIL` library.

#### Notes

- Ensure that the `face_recognition` and `qrcode` libraries are installed.
- The script saves the QR code and the final image with the QR code overlaid in the `biometric_qr_codes` directory.
- The face encodings are rounded to four decimal places to reduce the size of the QR code.

### Validating QR Code with Face Script

The `validate_qr_code_with_face.py` script validates the identity by comparing the face in a captured picture with biometric data extracted from a QR code image file. The QR code must be clear and generated using the `create_biometric_qr_code.py` script.

#### Script Overview

1. **Detecting QR Code:**
    - The script uses the `pyzxing` library to detect and decode a QR code in the provided image file.

2. **Loading Biometric Data:**
    - The biometric data (face encodings and locations) is extracted from the decoded QR code data.

3. **Capturing Face Picture:**
    - The script captures a picture using the webcam for face comparison.

4. **Comparing Faces:**
    - The face in the captured image is compared with the biometric data from the QR code.
    - If a match is found, the face is labeled with the recognition probability and other details.

5. **Saving and Displaying Results:**
    - The script saves the image with drawn rectangles and labels around detected faces in a directory named `qr_code_validation_results`.

#### Usage

1. **Run the script:**

    ```bash
    python validate_qr_code_with_face.py
    ```

2. **Enter the path to the QR code image file:**

    When prompted, enter the path to the image file containing the QR code.

3. **Enter the face recognition threshold:**

    When prompted, enter a threshold value for face recognition (e.g., `0.6`).

4. **Capture a picture:**

    - Press the `SPACE` key to take a photo using the webcam.
    - Press the `q` key to quit the photo capture session.

5. **View the results:**

    The script will display the captured image with detected faces labeled with recognition probabilities and other details. The results will be saved in the `qr_code_validation_results` directory.

#### Example

```bash
python validate_qr_code_with_face.py
```

When prompted, enter the path to the QR code image file:

```plaintext
Enter the path to the QR code image file: path/to/qr_code_image.jpg
```

Enter the threshold for face recognition:

```plaintext
Enter the face recognition threshold (e.g., 0.6): 0.6
```

Press `SPACE` to capture a photo or `q` to quit. The results will be displayed and saved in the `qr_code_validation_results` directory.

#### Notes

- Ensure that the `face_recognition`, `cv2`, `pyzxing`, and `PIL` libraries are installed.
- The script saves the captured images and results in the `temp_validation` and `qr_code_validation_results` directories.
- The threshold value determines the strictness of face matching. A lower value means stricter matching.
- The script deletes the temporary directory `temp_validation` after execution.
- The QR code must be clear and generated using the `create_biometric_qr_code.py` script.
