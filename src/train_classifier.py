
from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.model_config import CLASS_NAMES


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load extracted features and group IDs
# --------------------------------------------------

print("\nLoading CNN features...")

train_data = np.load(FEATURE_DIR / "train_features.npz")
validation_data = np.load(FEATURE_DIR / "validation_features.npz")

X_train = train_data["X"]
y_train = train_data["y"]
groups_train = train_data["groups"]

X_validation = validation_data["X"]
y_validation = validation_data["y"]

print("Training features:", X_train.shape)
print("Training groups:", groups_train.shape)

print("Validation features:", X_validation.shape)

if len(X_train) != len(groups_train):
    raise ValueError("Training features and groups do not match.")

if len(set(groups_train)) < 5:
    raise ValueError("Not enough unique groups for 5-fold CV.")


# --------------------------------------------------
# 3. Define SVM pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "svm",
        SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=42
        )
    )
])


# --------------------------------------------------
# 4. Hyperparameter search
# --------------------------------------------------

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.1, 0.01, 0.001]
}


# --------------------------------------------------
# 5. Group-aware cross-validation
# --------------------------------------------------

print("\nStarting group-aware cross-validation...")

group_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=group_cv,
    scoring="f1_macro",
    n_jobs=1,
    refit=True,
    error_score="raise",
    verbose=1
)

search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\nGroup-aware hyperparameter tuning completed.")

print("Best parameters:", search.best_params_)
print(f"Best group-aware CV macro-F1: {search.best_score_:.4f}")


# --------------------------------------------------
# 6. Train final classifier
# --------------------------------------------------

print("\nTraining final SVM classifier...")

best_C = search.best_params_["svm__C"]
best_gamma = search.best_params_["svm__gamma"]

classifier = Pipeline([
    ("scaler", StandardScaler()),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=best_C,
            gamma=best_gamma,
            probability=False,
            class_weight="balanced",
            random_state=42
        )
    )
])

classifier.fit(X_train, y_train)

print("Final SVM training completed.")


# --------------------------------------------------
# 7. Validation evaluation
# --------------------------------------------------

print("\nEvaluating validation dataset...")

validation_predictions = classifier.predict(X_validation)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

validation_macro_f1 = f1_score(
    y_validation,
    validation_predictions,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

training_accuracy = classifier.score(X_train, y_train)

print(f"\nTraining Accuracy: {training_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {validation_macro_f1:.4f}")

print("\nValidation Classification Report:")

print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

print("\nValidation Confusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_predictions,
        labels=[0, 1, 2]
    )
)


# --------------------------------------------------
# 8. Save model and results
# --------------------------------------------------

model_path = MODEL_DIR / "ecg_svm_classifier.pkl"

joblib.dump(classifier, model_path)

results = {
    "best_parameters": search.best_params_,
    "group_aware_cv_macro_f1": float(search.best_score_),
    "training_accuracy": float(training_accuracy),
    "validation_accuracy": float(validation_accuracy),
    "validation_macro_f1": float(validation_macro_f1),
    "class_names": CLASS_NAMES,
    "cross_validation": "StratifiedGroupKFold",
    "number_of_folds": 5
}

results_path = MODEL_DIR / "training_results.json"

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n===================================")
print("GROUP-AWARE SVM TRAINING COMPLETED")
print("===================================")

print("Best parameters:", search.best_params_)
print(f"Group-aware CV macro-F1: {search.best_score_:.4f}")
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {validation_macro_f1:.4f}")

print("\nSaved classifier:", model_path)
print("Saved results:", results_path)
