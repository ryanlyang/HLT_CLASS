from __future__ import annotations

import numpy as np
import pytest

from hlt_classification.data.identity import FileRecord, JetIdentity
from hlt_classification.prad.audit import (
    PRAD_DATA_AUDIT_CONTRACT,
    render_data_audit_markdown,
    summarize_prad_sample,
)


def _sample():
    files = tuple(
        FileRecord(f"class_{label}/sample.root", label, 20 + label)
        for label in range(10)
    )
    identities = (
        JetIdentity(files[0].file, 0, 0),
        JetIdentity(files[1].file, 1, 1),
    )
    tokens = np.zeros((2, 4, 14), dtype=np.float32)
    mask = np.asarray(
        [[True, True, False, False], [True, False, False, False]],
        dtype=np.bool_,
    )
    tokens[0, 0, :5] = (10.0, 0.0, 0.0, 10.1, 1.0)
    tokens[0, 0, 5] = 1.0
    tokens[0, 1, :5] = (5.0, 0.2, 0.2, 5.1, 0.0)
    tokens[0, 1, 7] = 1.0
    tokens[1, 0, :5] = (8.0, -0.1, -0.1, 8.1, -1.0)
    tokens[1, 0, 8] = 1.0
    return files, identities, tokens, mask


def test_prad_data_audit_reports_pairing_and_source_limits() -> None:
    files, identities, tokens, mask = _sample()
    report = summarize_prad_sample(
        files=files,
        identities=identities,
        offline_tokens=tokens,
        offline_mask=mask,
        labels=np.asarray([0, 1], dtype=np.int64),
        hlt_tokens=tokens.copy(),
        hlt_mask=mask.copy(),
        split_manifest_sha256="a" * 64,
    )
    assert report["contract"] == PRAD_DATA_AUDIT_CONTRACT
    assert report["inventory"]["total_paired_jets"] == 245
    assert report["sample"]["paired_labels_agree"]
    assert report["matching"]["particle_coverage"] == 1.0
    assert report["matching"]["matched_pt_coverage"] == 1.0
    assert report["matching"]["construction_indices_used"] is False
    assert report["physical_event_overlap"].startswith("unavailable")
    assert "Total paired jet rows: 245" in render_data_audit_markdown(report)


def test_prad_data_audit_fails_closed_on_label_disagreement() -> None:
    files, identities, tokens, mask = _sample()
    with pytest.raises(ValueError, match="labels differ"):
        summarize_prad_sample(
            files=files,
            identities=identities,
            offline_tokens=tokens,
            offline_mask=mask,
            labels=np.asarray([1, 0], dtype=np.int64),
            hlt_tokens=tokens.copy(),
            hlt_mask=mask.copy(),
            split_manifest_sha256="a" * 64,
        )
