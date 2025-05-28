import cv2
import numpy as np
import os
from datetime import datetime
import json
from Listen_Speak import Speak_en

def update_attendance_record(student_id, status):
    ATTENDANCE_FILE = 'attendance.json'
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                records = json.load(f)
        else:
            records = {}

        if today not in records:
            records[today] = {'present': [], 'absent': []}

        # Remove student from both lists first
        records[today]['present'] = [s for s in records[today]['present'] if s != student_id]
        records[today]['absent'] = [s for s in records[today]['absent'] if s != student_id]

        # Add to appropriate list
        if status == 'Present':
            if student_id not in records[today]['present']:
                records[today]['present'].append(student_id)
        else:
            if student_id not in records[today]['absent']:
                records[today]['absent'].append(student_id)

        with open(ATTENDANCE_FILE, 'w') as f:
            json.dump(records, f, indent=4)
        return True
    except Exception as e:
        print(f"Error updating attendance record: {str(e)}")
        return False

def face_recognition():
    try:
        # Get current directory and set up paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        trainer_path = os.path.join(current_dir, 'trainer.yml')
        cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')

        # Check if required files exist
        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"Trainer file not found at: {trainer_path}")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found at: {cascade_path}")

        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
        faceCascade = cv2.CascadeClassifier(cascade_path)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Configure based on your trained model
        names = ['unknown']  # Start with unknown at index 0
        # Add names for all student IDs found in samples directory
        if os.path.exists('samples'):
            student_ids = set()
            for filename in os.listdir('samples'):
                if filename.startswith(('face.', 'Student.')):
                    try:
                        student_id = filename.split('.')[1]
                        student_ids.add(int(student_id))  # Convert to int for sorting
                    except (IndexError, ValueError):
                        continue
            
            # Add IDs to names list in order
            for id in sorted(student_ids):
                names.append(str(id))  # Add each student ID to names list

        print("Loaded student IDs:", names[1:])  # Debug print

        # Initialize camera
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise Exception("Failed to open camera")

        cam.set(3, 1280)  # Width
        cam.set(4, 720)   # Height

        # Set minimum face size
        minW = int(0.1 * cam.get(3))
        minH = int(0.1 * cam.get(4))
        
        flag = 0
        recognized_name = "unknown"
        confidence_str = ""

        # Get all registered student IDs for marking absent later
        student_ids = set()
        if os.path.exists('samples'):
            for filename in os.listdir('samples'):
                if filename.startswith(('face.', 'Student.')):
                    try:
                        student_id = filename.split('.')[1]
                        student_ids.add(student_id)
                    except IndexError:
                        continue

        recognized_students = set()  # Keep track of recognized students

        while True:
            ret, img = cam.read()
            if not ret:
                raise Exception("Failed to grab frame from camera")

            # Convert to grayscale
            converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces with adjusted parameters
            faces = faceCascade.detectMultiScale(
                converted_image,
                scaleFactor=1.1,  # Reduced from 1.2 for better detection
                minNeighbors=5,
                minSize=(minW, minH)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Predict face with improved preprocessing
                face_roi = converted_image[y:y + h, x:x + w]
                # Add histogram equalization for better recognition
                face_roi = cv2.equalizeHist(face_roi)
                
                label, confidence = recognizer.predict(face_roi)

                # Adjust confidence threshold and handling
                if confidence < 70:  # Increased threshold for better matching
                    if label < len(names):  # Verify label is valid
                        recognized_name = names[label]
                        confidence_str = f"{round(100 - confidence)}% Match"
                        print(f"Recognized: {recognized_name} with confidence: {confidence}")  # Debug print
                        
                        if recognized_name != "unknown":
                            flag = 1
                            Speak_en(f"Welcome sir ")
                            recognized_students.add(recognized_name)
                            update_attendance_record(recognized_name, 'Present')
                            break
                    else:
                        print(f"Invalid label detected: {label}")  # Debug print
                else:
                    recognized_name = "unknown"
                    confidence_str = f"{round(100 - confidence)}% Match"
                    
                # Display recognition results
                cv2.putText(img, recognized_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_str, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # Show the image
            cv2.imshow('Face Recognition', img)

            # Check for exit conditions
            k = cv2.waitKey(10) & 0xff
            if k == 27 or flag == 1:  # ESC key or face recognized
                break

        # Mark absent for unrecognized students
        for student_id in student_ids:
            if student_id not in recognized_students:
                update_attendance_record(student_id, 'Absent')

        # Clean up
        cam.release()
        cv2.destroyAllWindows()
        return flag

    except Exception as e:
        print(f"Error in face recognition: {str(e)}")
        return 0

def face_unlock():
    return 1 if face_recognition() else 0

if __name__ == "__main__":
    # Test the face recognition system
    result = face_unlock()
    print(f"Authentication {'Successful' if result else 'Failed'}")