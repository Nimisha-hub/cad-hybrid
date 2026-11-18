
"""Automated tests for CardioCare CAD and HF agents.

All test inputs are synthetic. No real patient data is used.
"""

import pytest

from cad_hybrid.agents.base import PatientInputs
from cad_hybrid.agents.cardiocare import (
    CardiocareCADAgent,
    CardiocareHFAgent,
)
from cad_hybrid.agents.testing import check_agent_contract


@pytest.fixture
def cad_inputs():
    """Synthetic inputs for CardioCare Track A."""
    return PatientInputs(
        fields={
            "age": 55,
            "sex_male": 1,
            "bmi": 26,
            "diabetes": 0,
            "hypertension": 1,
            "current_smoker": 0,
            "family_history": 0,
            "typical_chest_pain": 1,
            "atypical_chest_pain": 0,
            "dyspnea": 0,
            "bp_mmhg": 130,
            "pulse_rate": 75,
            "ldl": 110,
            "hdl": 45,
            "triglycerides": 140,
            "hemoglobin": 14,
            "esr": 15,
            "ejection_fraction": 60,
        }
    )


@pytest.fixture
def hf_inputs():
    """Synthetic inputs for CardioCare Track B."""
    return PatientInputs(
        fields={
            "heart_failure_diagnosed": True,
            "age": 65,
            "sex_male": 1,
            "anaemia": 0,
            "diabetes": 1,
            "hypertension": 1,
            "current_smoker": 0,
            "ejection_fraction": 35,
            "creatinine": 1.2,
            "sodium": 137,
        }
    )


def test_cad_agent_follows_shared_contract(cad_inputs):
    agent = CardiocareCADAgent()

    report = check_agent_contract(agent, ok_inputs=cad_inputs)

    assert report.status == "ok"
    assert report.target == "cad"
    assert report.probability is not None
    assert 0.0 <= report.probability <= 1.0
    assert 0.0 < report.train_prior < 1.0


def test_hf_agent_follows_shared_contract(hf_inputs):
    agent = CardiocareHFAgent()

    report = check_agent_contract(agent, ok_inputs=hf_inputs)

    assert report.status == "ok"
    assert report.target == "hf_survival"
    assert report.probability is None
    assert report.train_prior is None


def test_cad_missing_field_is_rejected(cad_inputs):
    agent = CardiocareCADAgent()

    fields = dict(cad_inputs.fields)
    fields.pop("esr")

    incomplete = PatientInputs(fields=fields)

    assert agent.can_handle(incomplete) is False

    report = agent.assess(incomplete)

    assert report.status == "rejected"
    assert "esr" in report.missing_fields


def test_hf_missing_diagnosis_is_rejected(hf_inputs):
    agent = CardiocareHFAgent()

    fields = dict(hf_inputs.fields)
    fields.pop("heart_failure_diagnosed")

    incomplete = PatientInputs(fields=fields)

    assert agent.can_handle(incomplete) is False

    report = agent.assess(incomplete)

    assert report.status == "rejected"
    assert "heart_failure_diagnosed" in report.missing_fields


def test_hf_unconfirmed_diagnosis_is_rejected(hf_inputs):
    agent = CardiocareHFAgent()

    fields = dict(hf_inputs.fields)
    fields["heart_failure_diagnosed"] = False

    unconfirmed = PatientInputs(fields=fields)

    assert agent.can_handle(unconfirmed) is False

    report = agent.assess(unconfirmed)

    assert report.status == "rejected"
    assert "confirmed HF diagnosis" in report.reason


def test_hf_missing_clinical_field_is_rejected(hf_inputs):
    agent = CardiocareHFAgent()

    fields = dict(hf_inputs.fields)
    fields.pop("sodium")

    incomplete = PatientInputs(fields=fields)

    assert agent.can_handle(incomplete) is False

    report = agent.assess(incomplete)

    assert report.status == "rejected"
    assert "sodium" in report.missing_fields


def test_cad_invalid_binary_value_is_rejected(cad_inputs):
    agent = CardiocareCADAgent()

    fields = dict(cad_inputs.fields)
    fields["diabetes"] = 5

    invalid = PatientInputs(fields=fields)

    report = agent.assess(invalid)

    assert report.status == "rejected"
    assert report.reason is not None


def test_cad_probability_is_valid(cad_inputs):
    report = CardiocareCADAgent().assess(cad_inputs)

    assert report.status == "ok"
    assert isinstance(report.probability, float)
    assert 0.0 <= report.probability <= 1.0
    assert 0.0 < report.train_prior < 1.0


def test_hf_survival_values_are_valid(hf_inputs):
    report = CardiocareHFAgent().assess(hf_inputs)

    assert report.status == "ok"

    days = report.outputs["days"]
    survival = report.outputs["survival"]

    assert len(days) == len(survival)
    assert len(days) > 0

    assert all(0.0 <= value <= 1.0 for value in survival)

    assert all(
        survival[i] <= survival[i - 1]
        for i in range(1, len(survival))
    )


def test_hf_risk_matches_survival_curve(hf_inputs):
    report = CardiocareHFAgent().assess(hf_inputs)

    assert report.status == "ok"

    days = report.outputs["days"]
    survival = report.outputs["survival"]

    survival_by_day = dict(zip(days, survival))

    risk_90d = report.outputs["risk_90d"]
    risk_300d = report.outputs["risk_300d"]

    assert 0.0 <= risk_90d <= 1.0
    assert 0.0 <= risk_300d <= 1.0

    assert risk_90d == pytest.approx(
        1.0 - survival_by_day[90]
    )

    assert risk_300d == pytest.approx(
        1.0 - survival_by_day[300]
    )

    assert risk_300d >= risk_90d


def test_hf_survival_has_expected_time_points(hf_inputs):
    report = CardiocareHFAgent().assess(hf_inputs)

    assert report.status == "ok"

    assert report.outputs["days"] == [
        0, 30, 60, 90, 120, 150,
        180, 210, 240, 270, 300,
    ]


def test_agents_reject_empty_inputs():
    empty = PatientInputs()

    cad_agent = CardiocareCADAgent()
    hf_agent = CardiocareHFAgent()

    assert cad_agent.can_handle(empty) is False
    assert hf_agent.can_handle(empty) is False

    assert cad_agent.assess(empty).status == "rejected"
    assert hf_agent.assess(empty).status == "rejected"
