import cv2
import numpy as np
import os
from typing import Tuple, List

def Images_And_Labels(path: str) -> None:
    """
    Train face recognition model using images from the specified path
    """
    print("Training faces it will take few seconds.........")
    
    # Initialize face detector and recognizer
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces: List[np.ndarray] = []
    ids: List[int] = []
    
    # Validate directory
    if not os.path.exists(path):
        os.makedirs(path)
        raise ValueError(f"Created empty directory '{path}'. Please add face images.")
    
    # Get valid image files
    image_files = [f for f in os.listdir(path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise ValueError(f"No images found in directory '{path}'")
    
    # Process each image
    for image_file in image_files:
        try:
            image_path = os.path.join(path, image_file)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Warning: Could not read image {image_file}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract ID from filename (format: user.1.jpg)
            try:
                id = int(os.path.splitext(image_file)[0].split('.')[1])
            except (IndexError, ValueError):
                print(f"Warning: Invalid filename format for {image_file}")
                continue
            
            # Detect faces
            face_rects = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in face_rects:
                faces.append(gray[y:y+h, x:x+w])
                ids.append(id)
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    if not faces:
        raise ValueError("No faces detected in any of the provided images")
    
    # Train the model
    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")
    print("<<< Model Trained >>>")