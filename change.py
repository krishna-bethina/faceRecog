import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load trained model
model_path = "models/arg_liveness_model.h5"  # Change path if needed
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded successfully from {model_path}")

# ✅ Load dataset for evaluation
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = data_gen.flow_from_directory(
    "liveness_dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False  # Keeps correct label ordering
)

# ✅ Get true labels and predictions
y_true = val_data.classes  # Actual labels
raw_preds = model.predict(val_data)
y_pred = (raw_preds > 0.5).astype(int)  # Convert probabilities to binary classes

# ✅ Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ✅ Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ✅ Print classification report
report = classification_report(y_true, y_pred, target_names=["Fake", "Real"])
print("\n Classification Report:\n")
print(report)
