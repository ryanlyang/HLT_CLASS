from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.cms2jc2_response.contracts import artifact, candidates, with_content_hash
from hlt_classification.cms2jc2_response.selection import bootstrap_weights, family_finalists, select


def reports():
    return [artifact("EVALUATION", parents={"membership": "a"*64, "metric_registry": "b"*64,
                                           "response": str(i)*64 if i < 10 else "c"*64},
                     candidate_id=c["id"], budget=budget, gate=.1, role="response_select",
                     source_groups=["one", "two", "three"], point={"score": .05+i*.001},
                     standard_error=.02, qualified=True, qualification={"closure": {"state": "pass"}})
            for i, c in enumerate(candidates()["candidates"]) for budget in ("250K", "1M", "FULL")]


def sensitivities(primary):
    finalists = family_finalists(primary)
    return [with_content_hash(dict(r, gate=g)) for r in primary
            if r["budget"] == "FULL" and r["candidate_id"] in finalists.values() for g in (.05, .2)]


def test_paired_source_bootstrap_never_counts_replicas_as_jets():
    weights = bootstrap_weights(["a", "b", "c"])
    assert weights.shape == (200, 3)
    assert np.all(weights.sum(axis=1) == 3)
    np.testing.assert_array_equal(weights, bootstrap_weights(["a", "b", "c"]))
    with pytest.raises(ValueError):
        bootstrap_weights(["b", "a"])


def test_selection_requires_every_row_and_follows_simplicity():
    rows = reports()
    sensitivity = sensitivities(rows)
    result = select(rows, sensitivity, registry_hash=candidates()["content_hash"])
    assert result["selected_candidate"] == "A_L"
    assert result["qualified"] and result["association_robust"]
    assert not result["classifier_metrics_used"]
    with pytest.raises(ValueError, match="27"):
        select(rows[1:], sensitivity, registry_hash=candidates()["content_hash"])
    with pytest.raises(ValueError, match="Six"):
        select(rows, sensitivity[:-1], registry_hash=candidates()["content_hash"])


def test_poor_scientific_results_complete_without_qualified_response():
    rows = [with_content_hash(dict(r, qualified=False, qualification={"closure": {"state": "fail"}})) for r in reports()]
    result = select(rows, sensitivities(rows), registry_hash=candidates()["content_hash"])
    assert result["selected_candidate"] == "A_L"
    assert not result["qualified"]
    assert result["scientific_status"] == "unqualified"


def test_uncertain_closure_is_inconclusive_not_a_job_failure():
    rows = [with_content_hash(dict(r, qualified=False, qualification={"closure": {"state": "inconclusive"}})) for r in reports()]
    result = select(rows, sensitivities(rows), registry_hash=candidates()["content_hash"])
    assert result["scientific_status"] == "inconclusive" and not result["qualified"]


def test_selection_blocks_cross_role_or_metric_reuse():
    rows = reports(); sensitivity = sensitivities(rows)
    rows[-1] = with_content_hash(dict(rows[-1], role="response_confirm"))
    with pytest.raises(ValueError, match="frozen selection"):
        select(rows, sensitivity, registry_hash=candidates()["content_hash"])
    rows = reports()
    sensitivity[0] = with_content_hash(dict(sensitivity[0], parents={
        **sensitivity[0]["parents"], "metric_registry": "f"*64}))
    with pytest.raises(ValueError, match="lineage"):
        select(rows, sensitivity, registry_hash=candidates()["content_hash"])
