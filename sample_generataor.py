import cv2
import os

def sample_generataor(student_id=None, output_dir='samples'):
    """
    Generate face samples for a student
    Args:
        student_id (str): ID of the student
        output_dir (str): Directory to save samples
    Returns:
        int: Number of samples taken
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height

    # Load face detector
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # If student_id is not provided, ask for input
    if student_id is None:
        student_id = input("Enter numeric user ID here: ")
    
    if not str(student_id).isdigit():
        raise ValueError("Student ID must be a number")

    print(f"\nStarting capture for student ID: {student_id}")
    print("Looking for face... Press ESC to quit")
    
    count = 0
    total_samples = 100  # Number of samples to take

    while count < total_samples:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1

            # Save the captured face
            file_name = f"Student.{student_id}.{count}.jpg"
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])

            # Display progress
            cv2.putText(img, f"Sample: {count}/{total_samples}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Capturing Faces', img)

        # Wait for 100ms or key press
        key = cv2.waitKey(100) & 0xff
        if key == 27:  # ESC key
            break

    print(f"\nSamples Captured: {count}")
    cam.release()
    cv2.destroyAllWindows()
    
    return count

if __name__ == "__main__":
    sample_generataor()  # Run standalone with manual ID input