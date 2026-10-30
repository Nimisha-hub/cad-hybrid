
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score
)

# Load calibrated classifier
model = joblib.load("models/ecg_calibrated_classifier.pkl")

# Load test features and duplicate group IDs
data = np.load("data/features/test_features.npz")

X_test = data["X"]
y_test = data["y"]
groups = data["groups"]

# Keep only the first image from each duplicate group
seen_groups = set()
unique_indices = []

for i, group in enumerate(groups):
    group = str(group)

    if group not in seen_groups:
        seen_groups.add(group)
        unique_indices.append(i)

# Select unique test samples
X_unique = X_test[unique_indices]
y_unique = y_test[unique_indices]

print("Original test images:", len(y_test))
print("Unique test groups:", len(unique_indices))
print("Repeated images excluded:", len(y_test) - len(unique_indices))

# Predict probabilities
probabilities = model.predict_proba(X_unique)

# Predict classes
predictions = model.predict(X_unique)

# CAD-positive probability
cad_index = list(model.classes_).index(0)
cad_probabilities = probabilities[:, cad_index]

# Binary CAD-positive targets
y_cad = (y_unique == 0).astype(int)

# Calculate metrics
print("\nUNIQUE-GROUP TEST RESULTS")
print("==============================")

print(f"Accuracy: {accuracy_score(y_unique, predictions):.4f}")
print(f"Macro-F1: {f1_score(y_unique, predictions, average='macro'):.4f}")
print(f"Brier Score: {brier_score_loss(y_cad, cad_probabilities):.4f}")
print(f"Log Loss: {log_loss(y_cad, cad_probabilities):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_cad, cad_probabilities):.4f}")
