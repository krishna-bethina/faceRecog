import cv2
import numpy as np
import pickle
import os
from mtcnn import MTCNN
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# Load VGG19 model for feature extraction
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

detector = MTCNN()

# Create a folder to save images
if not os.path.exists("dataset/"):
    os.makedirs("dataset/")

# Load known faces if exists
def load_known_faces():
    
    if os.path.exists("models/known_faces.pkl"):
        with open("models/known_faces.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def extract_features(image):
    img_resized = cv2.resize(image, (224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    return features.flatten()  # Flatten for storing

def register_person(person_name):
    cap = cv2.VideoCapture(0)
    known_faces = load_known_faces()
    image_count = 0

    while image_count < 5:  # Capture 5 images per person
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size > 0:
                cv2.imwrite(f"dataset/{person_name}_{image_count}.jpg", face_img)
                embedding = extract_features(face_img)
                known_faces[person_name] = embedding
                image_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Capturing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save updated face embeddings
    with open("models/known_faces.pkl", "wb") as f:
        pickle.dump(known_faces, f)

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ {person_name} registered successfully!")

# Run registration
name = input("Enter person name: ")
register_person(name)
