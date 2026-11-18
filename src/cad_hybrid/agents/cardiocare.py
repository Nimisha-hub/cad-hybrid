
"""CardioCare adapters for CAD prediction and HF survival.

Uses locally trained CardioCare model bundles and the shared cad-agentic
AgentReport contract. Does not modify or import code from the upstream app.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from cad_hybrid.agents.base import (
    AgentReport,
    PatientInputs,
    error,
    rejected,
)

VERSION = "1.0.0"

TRACK_A_FIELDS = {
    "Age": "age",
    "Sex": "sex_male",
    "BMI": "bmi",
    "DM": "diabetes",
    "HTN": "hypertension",
    "Current Smoker": "current_smoker",
    "FH": "family_history",
    "Typical Chest Pain": "typical_chest_pain",
    "Atypical": "atypical_chest_pain",
    "Dyspnea": "dyspnea",
    "BP": "bp_mmhg",
    "PR": "pulse_rate",
    "LDL": "ldl",
    "HDL": "hdl",
    "TG": "triglycerides",
    "HB": "hemoglobin",
    "ESR": "esr",
    "EF-TTE": "ejection_fraction",
}

TRACK_B_FIELDS = {
    "age": "age",
    "sex": "sex_male",
    "anaemia": "anaemia",
    "diabetes": "diabetes",
    "high_blood_pressure": "hypertension",
    "smoking": "current_smoker",
    "ejection_fraction": "ejection_fraction",
    "serum_creatinine": "creatinine",
    "serum_sodium": "sodium",
}

HF_DIAGNOSIS_FIELD = "heart_failure_diagnosed"

CAD_CAVEATS = [
    "The CAD model was trained on the Iranian Z-Alizadeh Sani dataset.",
    "External validation in other populations is required.",
    "EF-TTE measurement timing relative to CAD diagnosis has not been established.",
    "The reported test ROC-AUC is not evidence of clinical readiness.",
]

HF_CAVEATS = [
    "The HF model was trained on a small Pakistani cohort of 299 patients.",
    "The reported ROC-AUC is interval-level, not a patient-level C-index.",
    "This model estimates survival outcomes in diagnosed HF patients; it does not diagnose HF.",
    "External validation is required before clinical use.",
]


def _number(value: Any, field_name: str) -> float:
    """Convert a canonical numerical input to a finite float."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not Boolean.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")

    return result


def _binary(value: Any, field_name: str) -> int:
    """Convert a binary canonical field to 0 or 1."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "yes", "true", "male"}:
            return 1
        if normalized in {"0", "no", "false", "female"}:
            return 0

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be binary (0 or 1)."
        ) from exc

    if number not in (0.0, 1.0):
        raise ValueError(f"{field_name} must be binary (0 or 1).")

    return int(number)


def _model_values(
    inputs: PatientInputs,
    mapping: dict[str, str],
    track: str,
) -> dict[str, Any]:
    """Map canonical patient fields to original CardioCare feature names."""
    values = {}

    for model_field, canonical_field in mapping.items():
        raw_value = inputs.fields.get(canonical_field)

        if raw_value is None:
            raise ValueError(
                f"Missing required field: {canonical_field}"
            )

        if model_field == "Sex" or model_field == "sex":
            male = _binary(raw_value, canonical_field)
            values[model_field] = (
                "Male" if track == "A" and male else
                "Fmale" if track == "A" else male
            )

        elif model_field in {
            "DM", "HTN", "Current Smoker", "FH",
            "Typical Chest Pain", "anaemia", "diabetes",
            "high_blood_pressure", "smoking",
        }:
            values[model_field] = _binary(
                raw_value, canonical_field
            )

        elif model_field in {"Atypical", "Dyspnea"}:
            binary_value = _binary(raw_value, canonical_field)
            values[model_field] = "Y" if binary_value else "N"

        else:
            values[model_field] = _number(
                raw_value, canonical_field
            )

    return values


def _artifact_directory() -> Path:
    """Locate saved bundles relative to this repository."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "models" / "agents" / "cardiocare"


class _CardioCareBase:
    """Shared lazy-loading and metadata functionality."""

    def __init__(self):
        self._bundle = None
        self._metadata = None

    def _load(self, filename: str):
        """Load the local bundle and metadata once per agent instance."""
        if self._bundle is not None:
            return self._bundle, self._metadata

        import joblib

        directory = _artifact_directory()
        bundle_path = directory / filename
        metadata_path = directory / "metadata.json"

        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"CardioCare model bundle not found: {bundle_path}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"CardioCare metadata not found: {metadata_path}"
            )

        bundle = joblib.load(bundle_path)

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        self._bundle = bundle
        self._metadata = metadata

        return self._bundle, self._metadata

    @staticmethod
    def _provenance(
        metadata: dict,
        track: str,
        metric_name: str,
    ) -> dict:
        section = metadata.get(track, {})

        return {
            "project": metadata.get("project"),
            "upstream_commit": metadata.get("upstream_commit"),
            "dataset": metadata.get("datasets", {}),
            "selected_model": section.get("selected_model"),
            "metric_name": metric_name,
            "metric_value": section.get(metric_name),
            "bootstrap_auc_ci": section.get("bootstrap_auc_ci"),
            "calibration_method": section.get("calibration_method"),
            "calibration_folds": section.get("calibration_folds"),
        }


class CardiocareCADAgent(_CardioCareBase):
    """Predict CAD probability using CardioCare Track A."""

    name = "cardiocare_cad"
    target = "cad"
    input_description = (
        "18 clinical fields required by the CardioCare CAD model."
    )

    def can_handle(self, inputs: PatientInputs) -> bool:
        if not isinstance(inputs, PatientInputs):
            return False

        return all(
            inputs.fields.get(field) is not None
            for field in TRACK_A_FIELDS.values()
        )

    def assess(self, inputs: PatientInputs) -> AgentReport:
        if not isinstance(inputs, PatientInputs):
            return rejected(
                self.name, VERSION, self.target,
                "Expected PatientInputs.",
            )

        missing = [
            field for field in TRACK_A_FIELDS.values()
            if inputs.fields.get(field) is None
        ]

        if missing:
            return rejected(
                self.name, VERSION, self.target,
                "Required CAD input fields are missing.",
                missing_fields=missing,
            )

        try:
            bundle, metadata = self._load(
                "cardiocare_track_a.joblib"
            )

            from cardiocare.model import predict_a

            values = _model_values(inputs, TRACK_A_FIELDS, "A")
            result = predict_a(bundle, values)

            probability = float(result["probability"])
            prior = float(bundle["training_prior"])

            if not 0.0 <= probability <= 1.0:
                raise ValueError("CAD probability is outside [0, 1].")

            if not 0.0 < prior < 1.0:
                raise ValueError("CAD training prior is outside (0, 1).")

            raw_importance = result.get("contributions", {})

            if hasattr(raw_importance, "to_dict"):
                raw_importance = raw_importance.to_dict()

            importance = {
                str(key): float(value)
                for key, value in raw_importance.items()
            }

            importance = dict(
                sorted(
                    importance.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:10]
            )

            return AgentReport(
                agent=self.name,
                version=VERSION,
                target=self.target,
                status="ok",
                probability=probability,
                train_prior=prior,
                outputs={
                    "prediction": int(result["prediction"]),
                    "threshold": 0.5,
                },
                explanation={
                    "type": result.get("explanation_type"),
                    "global_permutation_importance": importance,
                },
                provenance=self._provenance(
                    metadata, "track_a", "test_roc_auc"
                ),
                caveats=list(CAD_CAVEATS),
            )

        except (ValueError, TypeError, KeyError) as exc:
            return rejected(
                self.name, VERSION, self.target,
                f"Invalid CAD input or model output: {exc}",
            )
        except Exception as exc:
            return error(
                self.name, VERSION, self.target,
                f"CAD model failure: {type(exc).__name__}: {exc}",
            )


class CardiocareHFAgent(_CardioCareBase):
    """Estimate survival outcomes using CardioCare Track B."""

    name = "cardiocare_hf"
    target = "hf_survival"
    input_description = (
        "Confirmed HF diagnosis and nine clinical fields required "
        "by the CardioCare HF survival model."
    )

    def can_handle(self, inputs: PatientInputs) -> bool:
        if not isinstance(inputs, PatientInputs):
            return False

        if not self._diagnosis_confirmed(
            inputs.fields.get(HF_DIAGNOSIS_FIELD)
        ):
            return False

        return all(
            inputs.fields.get(field) is not None
            for field in TRACK_B_FIELDS.values()
        )

    @staticmethod
    def _diagnosis_confirmed(value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {
                "1", "yes", "true", "confirmed", "diagnosed"
            }

        try:
            return float(value) == 1.0
        except (TypeError, ValueError):
            return False

    def assess(self, inputs: PatientInputs) -> AgentReport:
        if not isinstance(inputs, PatientInputs):
            return rejected(
                self.name, VERSION, self.target,
                "Expected PatientInputs.",
            )

        diagnosis = inputs.fields.get(HF_DIAGNOSIS_FIELD)

        if diagnosis is None:
            return rejected(
                self.name, VERSION, self.target,
                "Confirmed HF diagnosis is required.",
                missing_fields=[HF_DIAGNOSIS_FIELD],
            )

        if not self._diagnosis_confirmed(diagnosis):
            return rejected(
                self.name, VERSION, self.target,
                "HF survival assessment is restricted to patients "
                "with a confirmed HF diagnosis.",
            )

        missing = [
            field for field in TRACK_B_FIELDS.values()
            if inputs.fields.get(field) is None
        ]

        if missing:
            return rejected(
                self.name, VERSION, self.target,
                "Required HF input fields are missing.",
                missing_fields=missing,
            )

        try:
            bundle, metadata = self._load(
                "cardiocare_track_b.joblib"
            )

            from cardiocare.model import predict_b

            values = _model_values(inputs, TRACK_B_FIELDS, "B")
            result = predict_b(bundle, values)

            days = [int(value) for value in result["days"]]
            survival = [
                float(value) for value in result["survival"]
            ]

            if len(days) != len(survival) or not days:
                raise ValueError(
                    "Invalid survival output dimensions."
                )

            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in survival
            ):
                raise ValueError(
                    "Survival values must be finite and in [0, 1]."
                )

            if any(
                survival[index] > survival[index - 1] + 1e-12
                for index in range(1, len(survival))
            ):
                raise ValueError(
                    "Survival probability must be non-increasing."
                )

            risk_90d = float(result["risk_90d"])
            risk_300d = float(result["risk_300d"])

            if not (
                0.0 <= risk_90d <= 1.0
                and 0.0 <= risk_300d <= 1.0
            ):
                raise ValueError(
                    "Survival risks must be within [0, 1]."
                )

            return AgentReport(
                agent=self.name,
                version=VERSION,
                target=self.target,
                status="ok",
                probability=None,
                train_prior=None,
                outputs={
                    "days": days,
                    "survival": survival,
                    "risk_90d": risk_90d,
                    "risk_300d": risk_300d,
                },
                explanation={
                    "type": "discrete_time_survival",
                    "interval_days": bundle["bin_size"],
                    "number_of_intervals": bundle["n_bins"],
                },
                provenance=self._provenance(
                    metadata,
                    "track_b",
                    "interval_level_test_roc_auc",
                ),
                caveats=list(HF_CAVEATS),
            )

        except (ValueError, TypeError, KeyError) as exc:
            return rejected(
                self.name, VERSION, self.target,
                f"Invalid HF input or model output: {exc}",
            )
        except Exception as exc:
            return error(
                self.name, VERSION, self.target,
                f"HF model failure: {type(exc).__name__}: {exc}",
            )
