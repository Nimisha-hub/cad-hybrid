
import numpy as np
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.model_config import CLASS_NAMES


# 1. Load the trained SVM classifier
svm_model = joblib.load("models/ecg_svm_classifier.pkl")

# 2. Load validation features and labels
data = np.load("data/features/validation_features.npz")

X_validation = data["X"]
y_validation = data["y"]

print("Validation features:", X_validation.shape)
print("Validation labels:", y_validation.shape)


# 3. Create the calibrated classifier
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(svm_model),
    method="sigmoid"
)

# 4. Fit the calibration using validation data
calibrated_model.fit(X_validation, y_validation)

print("\nProbability calibration completed.")


# 5. Generate calibrated probabilities
probabilities = calibrated_model.predict_proba(X_validation)

# Find the CAD-positive class index
cad_index = list(calibrated_model.classes_).index(0)

cad_probabilities = probabilities[:, cad_index]

print("\nSample CAD-positive probabilities:")

for i in range(min(5, len(cad_probabilities))):
    print(f"Image {i + 1}: {cad_probabilities[i]:.4f}")


# 6. Save the calibrated classifier
joblib.dump(
    calibrated_model,
    "models/ecg_calibrated_classifier.pkl"
)

print("\nCalibrated classifier saved successfully.")
print("File: models/ecg_calibrated_classifier.pkl")
