import cv2
import numpy as np
import pickle
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
import os

# ✅ Load VGG19 for feature extraction
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# ✅ Load MTCNN for face detection
detector = MTCNN()

# ✅ Load registered face embeddings
def load_known_faces():
    if not os.path.exists("models/known_faces.pkl"):
        print("❌ No registered faces found! Run register_faces.py first.")
        return {}
    with open("models/known_faces.pkl", "rb") as f:
        return pickle.load(f)

known_faces = load_known_faces()

# ✅ Load trained liveness model
liveness_model = load_model("models/arg_liveness_model.h5")

def extract_features(image):
    img_resized = cv2.resize(image, (224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array)
    return features.flatten()

def predict_liveness(face_img):
    img_resized = cv2.resize(face_img, (224, 224)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = liveness_model.predict(img_resized)
    return prediction[0][0]  # 1 = Real, 0 = Fake

def recognize_face(face_img):
    face_embedding = extract_features(face_img)
    min_distance = 0.5
    recognized_name = "Unidentified"

    for name, saved_embedding in known_faces.items():
        distance = cosine(face_embedding, saved_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_name = name

    if min_distance >= 0.5:
        recognized_name = "Unidentified"

    return recognized_name


# ✅ Open webcam for real-time multi-face recognition with liveness detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']
        face_img = frame[y:y+h, x:x+w]

        if face_img.size > 0:
            liveness_score = predict_liveness(face_img)

            if liveness_score < 0.5:  # Fake face detected
                label = "Liveliness not detected"
                color = (0, 0, 255)  # Red for fake
            else:
                person_name = recognize_face(face_img)
                label = person_name
                color = (0, 255, 0)  # Green for real

            # Draw rectangle and display name
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Multi Face Recognition + Liveness", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
