from __future__ import annotations

import numpy as np

from hlt_classification.scouting.hcwdl_paired_bootstrap import (
    DEFAULT_METRICS,
    bootstrap_index_sha256,
    iter_stratified_indices,
    paired_classification_bootstrap,
    publish_paired_bootstrap_envelope,
)


def test_stratified_indices_preserve_all_fifteen_class_counts() -> None:
    labels = np.repeat(np.arange(15, dtype=np.int64), [2 + i % 3 for i in range(15)])
    counts = np.bincount(labels, minlength=15)
    rows = list(iter_stratified_indices(labels, replicates=8, seed=8041))
    assert len(rows) == 8
    assert all(np.array_equal(np.bincount(labels[row], minlength=15), counts) for row in rows)
    assert bootstrap_index_sha256(labels, replicates=8) == bootstrap_index_sha256(
        labels, replicates=8
    )


def test_paired_bootstrap_reuses_indices_and_records_arrays() -> None:
    labels = np.repeat(np.arange(15, dtype=np.int64), 2)
    left = np.zeros((len(labels), 15), np.float32)
    right = np.zeros_like(left)
    left[np.arange(len(labels)), labels] = 1

    def metric(logits, target):
        return {"score": float(logits[np.arange(len(target)), target].mean())}

    identities = np.zeros((len(labels), 32), dtype=np.uint8)
    identities[:, -4:] = np.arange(len(labels), dtype=">u4").view(np.uint8).reshape(-1, 4)
    sidecar, arrays = paired_classification_bootstrap(
        left_logits=left,
        right_logits=right,
        labels=labels,
        identity_digests=identities,
        left_id="left",
        right_id="right",
        comparison_id="left-minus-right",
        parent_hashes={"lock": "a" * 64},
        metrics=("score",),
        replicates=16,
        metric_function=metric,
    )
    assert sidecar["intervals"]["score"]["difference"]["point"] == 1.0
    assert sidecar["intervals"]["score"]["difference"]["bootstrap_median"] == 1.0
    assert np.all(arrays["replicate_deltas"] == 1.0)
    assert np.all(arrays["replicate_left"] == 1.0)
    assert np.all(arrays["replicate_right"] == 0.0)
    assert sidecar["scientific_authorization"] is False
    assert sidecar["replicate_index_sha256"] == bootstrap_index_sha256(
        labels, replicates=16
    )


def test_full_metric_registry_records_raw_zero_qcd_pass_and_linear_intervals() -> None:
    labels = np.repeat(np.arange(15, dtype=np.int64), 2)
    left = np.full((len(labels), 15), -8.0, np.float32)
    right = np.zeros_like(left)
    left[np.arange(len(labels)), labels] = 8.0
    identities = np.zeros((len(labels), 32), np.uint8)
    identities[:, -4:] = np.arange(len(labels), dtype=">u4").view(np.uint8).reshape(-1, 4)
    sidecar, arrays = paired_classification_bootstrap(
        left_logits=left,
        right_logits=right,
        labels=labels,
        identity_digests=identities,
        left_id="left",
        right_id="right",
        comparison_id="left-minus-right",
        parent_hashes={"metric_join": "b" * 64},
        replicates=2,
    )
    assert tuple(sidecar["metric_order"]) == DEFAULT_METRICS
    assert sidecar["quantile_method"] == "linear"
    assert sidecar["zero_qcd_pass_policy"]["raw_qcd_pass_stored"] is True
    assert "per_class.Xbb.qcd_pass_50pct" in sidecar["intervals"]
    assert sidecar["intervals"]["per_class.Xbb.qcd_pass_50pct"]["left"]["point"] == 0
    assert arrays["replicate_left"].shape == (2, len(DEFAULT_METRICS))


def test_bootstrap_binary_is_published_only_as_a_committed_envelope(tmp_path) -> None:
    labels = np.repeat(np.arange(15, dtype=np.int64), 2)
    left = np.zeros((len(labels), 15), np.float32)
    right = np.ones_like(left)
    identities = np.zeros((len(labels), 32), np.uint8)
    identities[:, -4:] = np.arange(len(labels), dtype=">u4").view(np.uint8).reshape(-1, 4)

    def metric(logits, target):
        del target
        return {"score": float(logits.mean())}

    report, arrays = paired_classification_bootstrap(
        left_logits=left,
        right_logits=right,
        labels=labels,
        identity_digests=identities,
        left_id="left",
        right_id="right",
        comparison_id="left-minus-right",
        parent_hashes={"metric_join": "c" * 64},
        metrics=("score",),
        replicates=2,
        metric_function=metric,
    )
    envelope = publish_paired_bootstrap_envelope(
        tmp_path,
        bootstrap_report=report,
        arrays=arrays,
        producer_task_id="paired_bootstrap",
        registered_output_row={"comparison_id": "left-minus-right"},
        campaign_or_recovery_owner={"campaign": "pilot"},
    )
    assert envelope.directory == tmp_path / "committed" / envelope.envelope_id
    assert (envelope.directory / "bootstrap_arrays.npz").is_file()
    assert envelope.sidecar["payload"]["source_bootstrap_report_sha256"] == report["content_hash"]
