import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix

from src.model_config import CLASS_NAMES


# Load the saved SVM classifier
model = joblib.load("models/ecg_svm_classifier.pkl")

# Load validation features
data = np.load("data/features/validation_features.npz")

X_validation = data["X"]
y_validation = data["y"]

# Predict validation labels
y_pred = model.predict(X_validation)

# Display classification report
print("\nCLASS-WISE VALIDATION REPORT")

print(
    classification_report(
        y_validation,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# Display confusion matrix
print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_validation,
        y_pred,
        labels=[0, 1, 2]
    )
)