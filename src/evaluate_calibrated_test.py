
import numpy as np
import joblib

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)

# 1. Load calibrated classifier
model = joblib.load("models/ecg_calibrated_classifier.pkl")

# 2. Load untouched test features
data = np.load("data/features/test_features.npz")

X_test = data["X"]
y_test = data["y"]

# 3. Predict class probabilities
probabilities = model.predict_proba(X_test)

# 4. Find CAD-positive class (label 0)
cad_index = list(model.classes_).index(0)
cad_probabilities = probabilities[:, cad_index]

# Convert labels into binary CAD-positive targets
y_cad = (y_test == 0).astype(int)

# 5. Evaluate probability quality
brier = brier_score_loss(y_cad, cad_probabilities)
logloss = log_loss(y_cad, cad_probabilities)
auc = roc_auc_score(y_cad, cad_probabilities)

print("\nCALIBRATED MODEL - TEST RESULTS")
print("================================")

print(f"Test images: {len(y_test)}")
print(f"Brier Score: {brier:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"ROC-AUC: {auc:.4f}")

print("\nSAMPLE CAD-POSITIVE PROBABILITIES")

for i in range(min(10, len(y_test))):
    print(
        f"Image {i + 1}: "
        f"Actual label = {y_test[i]}, "
        f"CAD probability = {cad_probabilities[i]:.4f}"
    )
