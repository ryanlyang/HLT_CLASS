from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.evaluation import softmax
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (
    POSTHOC_ENSEMBLE_REPORT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_ensemble import (
    ENSEMBLE_RUNGS,
    MEMBER_ORDER,
    REPORT_SCHEMA_VERSION,
    WEIGHTS,
    _validate_optional_frozen_digest,
    equal_weight_probability_logits,
    validate_ensemble_report,
)


def test_equal_weight_probability_ensemble_is_exact_probability_mean():
    members = [
        np.tile(np.arange(15, dtype=np.float64) * scale, (3, 1))
        for scale in (0.1, -0.05, 0.025)
    ]
    ensemble = equal_weight_probability_logits(members)
    expected = sum(softmax(value) for value in members) / 3.0
    assert np.allclose(softmax(ensemble), expected, rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize("bad", [[], [np.zeros((2, 15))] * 2, [np.zeros((2, 14))] * 3])
def test_equal_weight_probability_ensemble_rejects_wrong_members(bad):
    with pytest.raises(ValueError):
        equal_weight_probability_logits(bad)


def test_equal_weight_probability_ensemble_rejects_nonfinite():
    values = [np.zeros((2, 15)) for _ in range(3)]
    values[1][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        equal_weight_probability_logits(values)


def test_future_parent_logit_reference_may_omit_launch_time_report_hash():
    _validate_optional_frozen_digest(
        {"report_path": "/future/report.json", "expected_node_id": "D40F"},
        "a" * 64,
    )
    _validate_optional_frozen_digest({"report_sha256": "a" * 64}, "a" * 64)
    with pytest.raises(ValueError):
        _validate_optional_frozen_digest({"report_sha256": "b" * 64}, "a" * 64)


def _report():
    return with_content_hash({
        "contract": POSTHOC_ENSEMBLE_REPORT_CONTRACT,
        "schema_version": REPORT_SCHEMA_VERSION,
        "rung": ENSEMBLE_RUNGS[0],
        "member_order": list(MEMBER_ORDER),
        "weights": list(WEIGHTS),
        "role": "validation",
        "validation_only": True,
        "final_test_accessed": False,
    })


def test_ensemble_report_contract_freezes_validation_only_semantics():
    report = _report()
    assert validate_ensemble_report(report) == report["content_hash"]
    tampered = with_content_hash({**report, "role": "final_test"})
    with pytest.raises(ValueError):
        validate_ensemble_report(tampered)
