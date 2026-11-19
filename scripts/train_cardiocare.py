
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import sklearn

from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_A = (
    ROOT
    / ".."
    / "cardiocare-app"
    / "extention_of_Z-Alizadeh_sani_dataset.xlsx"
).resolve()

DATA_B = (
    ROOT
    / ".."
    / "cardiocare-app"
    / "heart_failure_clinical_records_dataset.csv"
).resolve()

OUTPUT_DIR = ROOT / "models" / "agents" / "cardiocare"

UPSTREAM_COMMIT = "eb11f8c321638c04f34ac6df7012e3a819f23880"
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 2000


# ---------------------------------------------------------
# IMPORT PROJECT MODEL FUNCTIONS
# ---------------------------------------------------------

sys.path.insert(0, str(ROOT))

from cardiocare.model import train_a, train_b


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def bootstrap_auc_ci(
    y_true,
    probabilities,
    samples=2000,
    random_state=42,
):
    """Stratified row-level bootstrap for Track A."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(random_state)

    negative_indices = np.flatnonzero(y_true == 0)
    positive_indices = np.flatnonzero(y_true == 1)

    scores = []

    for _ in range(samples):
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        indices = np.concatenate([
            sampled_negative,
            sampled_positive,
        ])

        score = roc_auc_score(
            y_true[indices],
            probabilities[indices],
        )

        scores.append(score)

    return {
        "lower_95": float(np.percentile(scores, 2.5)),
        "upper_95": float(np.percentile(scores, 97.5)),
        "valid_samples": len(scores),
        "method": "stratified_row_bootstrap",
    }


def cluster_bootstrap_auc_ci(
    patient_ids,
    y_true,
    probabilities,
    samples=2000,
    random_state=42,
):
    """Bootstrap complete patients, retaining all their intervals."""

    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    unique_patients = np.unique(patient_ids)

    patient_indices = {
        patient: np.flatnonzero(patient_ids == patient)
        for patient in unique_patients
    }

    rng = np.random.default_rng(random_state)
    scores = []

    for _ in range(samples):
        sampled_patients = rng.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )

        indices = np.concatenate([
            patient_indices[patient]
            for patient in sampled_patients
        ])

        sampled_y = y_true[indices]

        if len(np.unique(sampled_y)) < 2:
            continue

        score = roc_auc_score(
            sampled_y,
            probabilities[indices],
        )

        scores.append(score)

    if not scores:
        raise ValueError(
            "No valid Track B bootstrap samples."
        )

    return {
        "lower_95": float(np.percentile(scores, 2.5)),
        "upper_95": float(np.percentile(scores, 97.5)),
        "valid_samples": len(scores),
        "method": "patient_cluster_bootstrap_interval_level_auc",
    }


def convert_cv_results(results):
    output = {}

    for name, scores in results.items():
        output[name] = {
            "mean_auc": float(np.mean(scores)),
            "std_auc": float(np.std(scores)),
            "fold_scores": [
                float(score) for score in scores
            ],
        }

    return output


# ---------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("CARDIOCARE MODEL TRAINING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_A.exists():
        raise FileNotFoundError(
            f"Track A dataset not found: {DATA_A}"
        )

    if not DATA_B.exists():
        raise FileNotFoundError(
            f"Track B dataset not found: {DATA_B}"
        )

    metadata = {
        "project": "CardioCare",
        "upstream_commit": UPSTREAM_COMMIT,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "random_state": RANDOM_STATE,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "datasets": {},
        "track_a": {},
        "track_b": {},
    }

    # -----------------------------------------------------
    # TRACK A - CAD
    # -----------------------------------------------------

    print("\n[1/2] Training Track A - CAD")

    result_a = train_a(
        DATA_A,
        random_state=RANDOM_STATE,
    )

    ci_a = bootstrap_auc_ci(
        result_a["test_y"],
        result_a["test_probabilities"],
        samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_STATE,
    )

    # IMPORTANT:
    # Save the OOF Platt calibrator along with the model.
    # predict_a() requires both model and calibrator.

    bundle_a = {
        "model": result_a["model"],
        "calibrator": result_a["calibrator"],
        "features": result_a["features"],
        "training_prior": result_a["training_prior"],
        "global_importance": result_a["global_importance"],
        "importance_mode": result_a["importance_mode"],
    }

    joblib.dump(
        bundle_a,
        OUTPUT_DIR / "cardiocare_track_a.joblib",
    )

    metadata["datasets"]["track_a"] = {
        "filename": DATA_A.name,
        "sha256": sha256_file(DATA_A),
    }

    metadata["track_a"] = {
        "selected_model": result_a["best_name"],
        "test_roc_auc": result_a["test_auc"],
        "brier_score": result_a["brier_score"],
        "training_prior": result_a["training_prior"],
        "calibration_method": result_a["calibration_method"],
        "calibration_folds": result_a["calibration_cv"],
        "bootstrap_auc_ci": ci_a,
        "cv_results": convert_cv_results(
            result_a["cv_results"]
        ),
    }

    print("Selected model:", result_a["best_name"])
    print("ROC-AUC:", round(result_a["test_auc"], 4))
    print("Brier score:", round(result_a["brier_score"], 4))
    print("Bootstrap CI:", ci_a)
    print("Calibration:", result_a["calibration_method"])

    # -----------------------------------------------------
    # TRACK B - HEART FAILURE
    # -----------------------------------------------------

    print("\n[2/2] Training Track B - Heart Failure")

    result_b = train_b(
        DATA_B,
        random_state=RANDOM_STATE,
    )

    ci_b = cluster_bootstrap_auc_ci(
        result_b["test_patient_ids"],
        result_b["test_y"],
        result_b["test_probabilities"],
        samples=BOOTSTRAP_SAMPLES,
        random_state=RANDOM_STATE,
    )

    bundle_b = {
        "model": result_b["model"],
        "features": result_b["features"],
        "model_columns": result_b["model_columns"],
        "bin_size": result_b["bin_size"],
        "n_bins": result_b["n_bins"],
    }

    joblib.dump(
        bundle_b,
        OUTPUT_DIR / "cardiocare_track_b.joblib",
    )

    metadata["datasets"]["track_b"] = {
        "filename": DATA_B.name,
        "sha256": sha256_file(DATA_B),
    }

    metadata["track_b"] = {
        "selected_model": result_b["best_name"],
        "interval_level_test_roc_auc": result_b["test_auc"],
        "training_patients": result_b["training_patients"],
        "test_patients": result_b["test_patients"],
        "bin_size_days": result_b["bin_size"],
        "number_of_bins": result_b["n_bins"],
        "bootstrap_auc_ci": ci_b,
        "cv_results": convert_cv_results(
            result_b["cv_results"]
        ),
    }

    print("Selected model:", result_b["best_name"])
    print(
        "Interval-level ROC-AUC:",
        round(result_b["test_auc"], 4),
    )
    print("Bootstrap CI:", ci_b)

    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\n" + "=" * 60)
    print("CARDIOCARE TRAINING COMPLETED")
    print("=" * 60)

    print("Output directory:", OUTPUT_DIR)
    print("Track A model: cardiocare_track_a.joblib")
    print("Track B model: cardiocare_track_b.joblib")
    print("Metadata: metadata.json")


if __name__ == "__main__":
    main()
