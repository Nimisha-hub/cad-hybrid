
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.model_config import CLASS_NAMES


# Load calibrated classifier
model = joblib.load(
    "models/ecg_calibrated_classifier.pkl"
)

# Load previously extracted test features
data = np.load("data/features/test_features.npz")

X_test = data["X"]
y_test = data["y"]

# Predict using calibrated classifier
y_pred = model.predict(X_test)

print("\nCALIBRATED SAVED-FEATURE EVALUATION")
print("====================================")

print(f"Test images: {len(y_test)}")

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Macro-F1: "
    f"{f1_score(y_test, y_pred, average='macro'):.4f}"
)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )
)
