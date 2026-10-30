import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.model_config import CLASS_NAMES


# Load trained classifier
model = joblib.load("models/ecg_svm_classifier.pkl")

# Load test features
data = np.load("data/features/test_features.npz")

X_test = data["X"]
y_test = data["y"]

# Predict test labels
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\nFINAL TEST RESULTS")
print("========================")

print(f"Test images: {len(y_test)}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test Macro-F1: {macro_f1:.4f}")

print("\nTEST CLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nTEST CONFUSION MATRIX")

print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))