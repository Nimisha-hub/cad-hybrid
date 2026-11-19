from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# TRACK A: CORONARY ARTERY DISEASE
# =========================================================

FEATURES_A = [
    "Age",
    "Sex",
    "BMI",
    "DM",
    "HTN",
    "Current Smoker",
    "FH",
    "Typical Chest Pain",
    "Atypical",
    "Dyspnea",
    "BP",
    "PR",
    "LDL",
    "HDL",
    "TG",
    "HB",
    "ESR",
    "EF-TTE",
]


# =========================================================
# TRACK B: HEART FAILURE SURVIVAL
# =========================================================

FEATURES_B = [
    "age",
    "sex",
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "smoking",
    "ejection_fraction",
    "serum_creatinine",
    "serum_sodium",
]

BIN_SIZE = 30
N_BINS = 10


# =========================================================
# SHARED MODEL CANDIDATES
# =========================================================

def build_candidate_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=0.5,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            random_state=random_state,
        ),
        "Support Vector Machine (RBF)": SVC(
            probability=True,
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
        ),
    }


def run_model_bakeoff(
    X_train,
    y_train,
    preprocessor,
    cv_folds=5,
    random_state=42,
    groups=None,
):
    """Compare candidate models using stratified or grouped CV."""

    candidates = build_candidate_models(random_state)

    if groups is None:
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        cv_splits = list(cv.split(X_train, y_train))

    else:
        groups = np.asarray(groups)

        if len(groups) != len(X_train):
            raise ValueError(
                "The number of groups must match training rows."
            )

        cv = StratifiedGroupKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        cv_splits = list(
            cv.split(X_train, y_train, groups=groups)
        )

        for train_indices, validation_indices in cv_splits:
            train_groups = set(groups[train_indices])
            validation_groups = set(groups[validation_indices])

            if train_groups.intersection(validation_groups):
                raise RuntimeError(
                    "Patient leakage detected between CV folds."
                )

    pipelines = {
        name: Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ])
        for name, model in candidates.items()
    }

    cv_results = {
        name: cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring="roc_auc",
        )
        for name, pipeline in pipelines.items()
    }

    ranked = sorted(
        cv_results.items(),
        key=lambda item: np.mean(item[1]),
        reverse=True,
    )

    top_two_names = [ranked[0][0], ranked[1][0]]

    ensemble = VotingClassifier(
        estimators=[
            (name, clone(candidates[name]))
            for name in top_two_names
        ],
        voting="soft",
    )

    ensemble_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", ensemble),
    ])

    ensemble_name = (
        f"Ensemble ({top_two_names[0]} + {top_two_names[1]})"
    )

    cv_results[ensemble_name] = cross_val_score(
        ensemble_pipeline,
        X_train,
        y_train,
        cv=cv_splits,
        scoring="roc_auc",
    )

    pipelines[ensemble_name] = ensemble_pipeline

    best_name = max(
        cv_results,
        key=lambda name: np.mean(cv_results[name]),
    )

    best_model = clone(pipelines[best_name])
    best_model.fit(X_train, y_train)

    return best_model, best_name, cv_results


# =========================================================
# TRACK A: DATA PREPARATION
# =========================================================

def _prepare_track_a(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Track A dataset not found: {path}"
        )

    data = pd.read_excel(path)

    missing_columns = [
        column
        for column in FEATURES_A + ["Cath"]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Track A dataset is missing columns: {missing_columns}"
        )

    X = data[FEATURES_A].copy()

    # Preserve upstream categorical encodings.
    X = X.replace({"Y": 1, "N": 0})

    X["Sex"] = X["Sex"].map({
        "Male": 1,
        "Fmale": 0,
    })

    y = data["Cath"].map({
        "CAD": 1,
        "Normal": 0,
    })

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError(
            "Track A contains missing or unrecognized values."
        )

    X = X.astype(float)
    y = y.astype(int)

    return X, y


# =========================================================
# TRACK A: OOF PLATT CALIBRATION
# =========================================================

def _probability_to_logit(probabilities):
    """
    Convert probabilities to logits for Platt calibration.

    Clipping prevents infinite values when probabilities
    are exactly zero or one.
    """

    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        probabilities / (1.0 - probabilities)
    ).reshape(-1, 1)


def _fit_oof_platt_calibrator(
    estimator,
    X,
    y,
    random_state=42,
    n_splits=5,
):
    """
    Generate one out-of-fold prediction per training row.

    Fit the Platt calibrator using only OOF predictions.
    Finally, refit the selected estimator on all training rows.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probabilities = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    for train_indices, validation_indices in cv.split(X, y):
        fold_model = clone(estimator)

        fold_model.fit(
            X.iloc[train_indices],
            y.iloc[train_indices],
        )

        oof_probabilities[validation_indices] = (
            fold_model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]
        )

    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError(
            "OOF calibration did not produce finite predictions."
        )

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        _probability_to_logit(oof_probabilities),
        y,
    )

    final_model = clone(estimator)

    final_model.fit(
        X,
        y,
    )

    return (
        final_model,
        calibrator,
        oof_probabilities,
    )


# =========================================================
# TRACK A: TRAINING
# =========================================================

def train_a(dataset_path, random_state=42):
    X, y = _prepare_track_a(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = StandardScaler()

    model, best_name, cv_results = run_model_bakeoff(
        X_train,
        y_train,
        preprocessor=preprocessor,
        cv_folds=5,
        random_state=random_state,
    )

    # Fit explicit OOF Platt calibration.
    final_model, calibrator, oof_probabilities = (
        _fit_oof_platt_calibrator(
            estimator=model,
            X=X_train,
            y=y_train,
            random_state=random_state,
            n_splits=5,
        )
    )

    # Predict using the final model, then calibrate its scores.
    raw_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    probabilities = calibrator.predict_proba(
        _probability_to_logit(raw_probabilities)
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    brier_score = brier_score_loss(
        y_test,
        probabilities,
    )

    # Global permutation importance for the final estimator.
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=random_state,
        scoring="roc_auc",
    )

    global_importance = pd.Series(
        importance.importances_mean,
        index=FEATURES_A,
    ).sort_values(
        key=abs,
        ascending=False,
    )

    return {
        "model": final_model,
        "calibrator": calibrator,
        "features": FEATURES_A,
        "best_name": best_name,
        "cv_results": cv_results,
        "test_auc": float(test_auc),
        "brier_score": float(brier_score),
        "training_prior": float(y_train.mean()),
        "importance_mode": "global",
        "global_importance": global_importance,
        "test_indices": X_test.index.tolist(),
        "test_y": y_test.to_numpy().tolist(),
        "test_probabilities": probabilities.tolist(),
        "oof_probabilities": oof_probabilities.tolist(),
        "calibration_method": "OOF Platt scaling",
        "calibration_cv": 5,
        "random_state": random_state,
    }


# =========================================================
# TRACK A: PREDICTION
# =========================================================

def predict_a(bundle, values):
    model = bundle["model"]
    calibrator = bundle["calibrator"]
    features = bundle["features"]

    missing = [
        feature
        for feature in features
        if feature not in values
    ]

    if missing:
        raise ValueError(
            f"Missing Track A inputs: {missing}"
        )

    row = pd.DataFrame([
        {
            feature: values[feature]
            for feature in features
        }
    ])

    row = row.replace({
        "Y": 1,
        "N": 0,
    })

    row["Sex"] = row["Sex"].replace({
        "Male": 1,
        "Fmale": 0,
    })

    try:
        row = row.astype(float)

    except (ValueError, TypeError):
        raise ValueError(
            "Track A inputs must be numeric or valid categories."
        )

    if not np.isfinite(
        row.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Track A inputs must contain only finite values."
        )

    raw_probability = float(
        model.predict_proba(row)[0, 1]
    )

    probability = float(
        calibrator.predict_proba(
            _probability_to_logit([raw_probability])
        )[0, 1]
    )

    contributions = bundle["global_importance"]

    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "contributions": contributions,
        "explanation_type": "global_permutation_importance",
    }


# =========================================================
# TRACK B: TIME-BIN EXPANSION
# =========================================================

def _expand_to_bins(data):
    data = data.copy()

    if "_patient_id" not in data.columns:
        data = data.reset_index(drop=True)
        data["_patient_id"] = np.arange(len(data))

    data["bin_last"] = (
        (data["time"] // BIN_SIZE)
        .clip(upper=N_BINS - 1)
        .astype(int)
    )

    rows = []

    for _, patient in data.iterrows():
        for bin_number in range(
            int(patient["bin_last"]) + 1
        ):
            row = {
                feature: patient[feature]
                for feature in FEATURES_B
            }

            # Used for grouped CV, never as a model input.
            row["patient_id"] = patient["_patient_id"]

            row["bin"] = bin_number
            row["bin_squared"] = bin_number ** 2

            row["event"] = int(
                bin_number == patient["bin_last"]
                and patient["DEATH_EVENT"] == 1
            )

            rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# TRACK B: TRAINING
# =========================================================

def train_b(dataset_path, random_state=42):
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Track B dataset not found: {path}"
        )

    data = pd.read_csv(path)

    # Assign patient IDs before splitting.
    data = data.reset_index(drop=True)
    data["_patient_id"] = np.arange(len(data))

    required_columns = FEATURES_B + [
        "time",
        "DEATH_EVENT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Track B dataset is missing columns: {missing_columns}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError(
            "Track B dataset contains missing values."
        )

    if not set(data["DEATH_EVENT"].unique()).issubset({0, 1}):
        raise ValueError(
            "Track B DEATH_EVENT must contain only 0 and 1."
        )

    # Split patients before expanding into intervals.
    train_patients, test_patients = train_test_split(
        data,
        test_size=0.15,
        stratify=data["DEATH_EVENT"],
        random_state=random_state,
    )

    train_expanded = _expand_to_bins(train_patients)
    test_expanded = _expand_to_bins(test_patients)

    model_columns = FEATURES_B + [
        "bin",
        "bin_squared",
    ]

    X_train = train_expanded[model_columns]
    X_test = test_expanded[model_columns]

    y_train = train_expanded["event"].astype(int)
    y_test = test_expanded["event"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "clinical",
                StandardScaler(),
                FEATURES_B,
            ),
            (
                "time",
                "passthrough",
                ["bin", "bin_squared"],
            ),
        ]
    )

    model, best_name, cv_results = run_model_bakeoff(
        X_train,
        y_train,
        preprocessor=preprocessor,
        cv_folds=3,
        random_state=random_state,
        groups=train_expanded["patient_id"].to_numpy(),
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    return {
        "model": model,
        "features": FEATURES_B,
        "model_columns": model_columns,
        "best_name": best_name,
        "cv_results": cv_results,
        "test_auc": float(test_auc),
        "bin_size": BIN_SIZE,
        "n_bins": N_BINS,
        "training_patients": len(train_patients),
        "test_patients": len(test_patients),

        # Held-out evaluation data.
        "test_patient_ids": (
            test_expanded["patient_id"].tolist()
        ),
        "test_y": y_test.tolist(),
        "test_probabilities": probabilities.tolist(),

        # Reproducibility.
        "random_state": random_state,
    }


# =========================================================
# TRACK B: PREDICTION
# =========================================================

def predict_b(bundle, values):
    features = bundle["features"]

    missing = [
        feature
        for feature in features
        if feature not in values
    ]

    if missing:
        raise ValueError(
            f"Missing Track B inputs: {missing}"
        )

    clinical_values = {
        feature: values[feature]
        for feature in features
    }

    try:
        clinical_row = pd.DataFrame(
            [clinical_values]
        ).astype(float)

    except (ValueError, TypeError):
        raise ValueError(
            "Track B inputs must be numeric."
        )

    if not np.isfinite(
        clinical_row.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Track B inputs must contain only finite values."
        )

    survival = 1.0
    days = [0]
    survival_values = [1.0]

    for bin_number in range(bundle["n_bins"]):
        row = clinical_row.copy()
        row["bin"] = bin_number
        row["bin_squared"] = bin_number ** 2

        hazard = float(
            bundle["model"].predict_proba(row)[0, 1]
        )

        hazard = float(
            np.clip(hazard, 0.0, 1.0)
        )

        survival *= 1.0 - hazard

        days.append(
            (bin_number + 1) * bundle["bin_size"]
        )

        survival_values.append(float(survival))

    risk_90d = 1.0 - float(
        np.interp(90, days, survival_values)
    )

    risk_300d = 1.0 - survival_values[-1]

    return {
        "days": days,
        "survival": survival_values,
        "risk_90d": float(
            np.clip(risk_90d, 0.0, 1.0)
        ),
        "risk_300d": float(
            np.clip(risk_300d, 0.0, 1.0)
        ),
    }