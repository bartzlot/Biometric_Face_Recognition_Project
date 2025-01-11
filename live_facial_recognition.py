import face_recognition
import cv2
import numpy as np
import pickle

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("face_recognition.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open MacBook camera")

# Prompt user to input the threshold value
threshold = float(input("Enter the threshold for face recognition (e.g., 0.6): "))

# Initialize our variables
cv_scaler = 4  # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
face_percentages = []
face_distances = []

def process_frame(frame):
    global face_locations, face_encodings, face_names, face_percentages, face_distances
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    face_percentages = []
    face_distances = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
        name = "Unknown"
        percentage = 0.0
        distance = 1.0
        
        # Use the known face with the smallest distance to the new face
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            distance = distances[best_match_index]
            percentage = (1 - distance) * 100
        face_names.append(name)
        face_percentages.append(percentage)
        face_distances.append(distance)
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name, percentage, distance in zip(face_locations, face_names, face_percentages, face_distances):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} ({percentage:.2f}%)", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        # Draw additional information below the face frame
        info_text = f"Threshold: {threshold}\nName: {name}\nProbability: {percentage:.2f}%\nDistance: {distance:.4f}"
        y0, dy = bottom + 20, 20
        for i, line in enumerate(info_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

while True:
    # Capture a frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame with the function
    processed_frame = process_frame(frame)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cap.release()
cv2.destroyAllWindows()