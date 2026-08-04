"""Bounded real-data audit for the PRAD campaign."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.hlt_v3 import build_hlt_v3_view
from hlt_classification.data.identity import FileRecord, JetIdentity
from hlt_classification.data.root_reader import load_offline_view
from hlt_classification.data.schema import CLASS_LABELS, RAW_TOKEN_FEATURES

from .contracts import CURRENT_SOURCE_CAPABILITIES
from .matching import CHARGED_CATEGORIES, match_hlt_to_offline

PRAD_DATA_AUDIT_CONTRACT = "hlt_classification_prad_data_audit_v1"
PRAD_DATA_AUDIT_SCHEMA_VERSION = 1


def _category(tokens: np.ndarray) -> np.ndarray:
    flags = tokens[:, 5:10]
    result = np.full(len(tokens), 5, dtype=np.int8)
    known = flags.sum(axis=1) == 1
    result[known] = flags[known].argmax(axis=1).astype(np.int8)
    return result


def _bin(value: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    return labels[int(np.searchsorted(np.asarray(edges), value, side="right"))]


def summarize_prad_sample(
    *,
    files: Sequence[FileRecord],
    identities: Sequence[JetIdentity],
    offline_tokens: np.ndarray,
    offline_mask: np.ndarray,
    labels: np.ndarray,
    hlt_tokens: np.ndarray,
    hlt_mask: np.ndarray,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Summarize a bounded paired sample without using construction indices."""

    count = len(identities)
    expected_shapes = {
        "offline_tokens": (count, offline_tokens.shape[1], 14),
        "offline_mask": (count, offline_tokens.shape[1]),
        "labels": (count,),
        "hlt_tokens": (count, offline_tokens.shape[1], 14),
        "hlt_mask": (count, offline_tokens.shape[1]),
    }
    actual = {
        "offline_tokens": offline_tokens.shape,
        "offline_mask": offline_mask.shape,
        "labels": labels.shape,
        "hlt_tokens": hlt_tokens.shape,
        "hlt_mask": hlt_mask.shape,
    }
    for name, shape in expected_shapes.items():
        if actual[name] != shape:
            raise ValueError(f"PRAD audit {name} shape differs")
    if offline_tokens.dtype != np.float32 or hlt_tokens.dtype != np.float32:
        raise ValueError("PRAD audit tokens must be float32")
    if offline_mask.dtype != np.bool_ or hlt_mask.dtype != np.bool_:
        raise ValueError("PRAD audit masks must be boolean")
    expected_labels = np.asarray([item.label for item in identities], np.int64)
    if not np.array_equal(labels, expected_labels):
        raise ValueError("HLT/offline paired jet labels differ from identities")

    inventory_counts = Counter()
    for record in files:
        inventory_counts[record.label] += record.num_entries
    particle_counts = offline_mask.sum(axis=1)
    hlt_counts = hlt_mask.sum(axis=1)
    missing_rates: dict[str, float] = {}
    hlt_missing_rates: dict[str, float] = {}
    for index, name in enumerate(RAW_TOKEN_FEATURES):
        values = offline_tokens[:, :, index][offline_mask]
        missing_rates[name] = (
            float(np.mean(~np.isfinite(values))) if len(values) else 0.0
        )
        hlt_values = hlt_tokens[:, :, index][hlt_mask]
        hlt_missing_rates[name] = (
            float(np.mean(~np.isfinite(hlt_values))) if len(hlt_values) else 0.0
        )

    class_accumulator: dict[str, list[float]] = defaultdict(list)
    type_totals = Counter()
    type_matches = Counter()
    type_pt = Counter()
    type_matched_pt = Counter()
    total_hlt_pt = 0.0
    total_matched_pt = 0.0
    strata: dict[str, dict[str, list[float]]] = {
        "jet_pt": defaultdict(list),
        "hlt_multiplicity": defaultdict(list),
    }
    per_jet: list[dict[str, Any]] = []
    for index, identity in enumerate(identities):
        result = match_hlt_to_offline(
            hlt_tokens[index],
            hlt_mask[index],
            offline_tokens[index],
            offline_mask[index],
        )
        matched = result.hlt_to_offline >= 0
        total_hlt_pt += float(hlt_tokens[index, hlt_mask[index], 0].sum())
        total_matched_pt += float(hlt_tokens[index, matched, 0].sum())
        categories = _category(hlt_tokens[index])
        for particle_index in np.flatnonzero(hlt_mask[index]):
            group = (
                "charged"
                if int(categories[particle_index]) in CHARGED_CATEGORIES
                else "neutral"
            )
            type_totals[group] += 1
            type_pt[group] += float(hlt_tokens[index, particle_index, 0])
            if matched[particle_index]:
                type_matches[group] += 1
                type_matched_pt[group] += float(
                    hlt_tokens[index, particle_index, 0]
                )
        fraction = float(result.diagnostics["matched_particle_fraction"])
        class_accumulator[CLASS_LABELS[identity.label]].append(fraction)
        valid_tokens = hlt_tokens[index, hlt_mask[index]]
        jet_pt = float(
            np.hypot(
                np.sum(valid_tokens[:, 0] * np.cos(valid_tokens[:, 2])),
                np.sum(valid_tokens[:, 0] * np.sin(valid_tokens[:, 2])),
            )
        )
        pt_bin = _bin(
            jet_pt,
            (500.0, 750.0, 1000.0),
            ("<=500", "500-750", "750-1000", ">1000"),
        )
        multiplicity_bin = _bin(
            int(hlt_mask[index].sum()),
            (20, 40, 80),
            ("<20", "20-40", "40-80", ">=80"),
        )
        strata["jet_pt"][pt_bin].append(fraction)
        strata["hlt_multiplicity"][multiplicity_bin].append(fraction)
        per_jet.append(result.diagnostics)

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    total_hlt = sum(item["hlt_particles"] for item in per_jet)
    total_matched = sum(item["matched_particles"] for item in per_jet)
    total_hlt_pairs = sum(
        item["hlt_particles"] * max(item["hlt_particles"] - 1, 0)
        for item in per_jet
    )
    total_matched_pairs = sum(
        item["matched_particles"] * max(item["matched_particles"] - 1, 0)
        for item in per_jet
    )
    payload = {
        "contract": PRAD_DATA_AUDIT_CONTRACT,
        "schema_version": PRAD_DATA_AUDIT_SCHEMA_VERSION,
        "split_manifest_sha256": split_manifest_sha256,
        "sample_identity_order_sha256": canonical_sha256(
            [item.key for item in identities]
        ),
        "inventory": {
            "total_paired_jets": int(sum(inventory_counts.values())),
            "class_counts": {
                CLASS_LABELS[label]: int(inventory_counts[label])
                for label in range(len(CLASS_LABELS))
            },
        },
        "sample": {
            "jets": count,
            "class_counts": {
                CLASS_LABELS[label]: int(np.sum(labels == label))
                for label in range(len(CLASS_LABELS))
            },
            "offline_particles_per_jet": {
                "minimum": int(particle_counts.min()) if count else 0,
                "median": float(np.median(particle_counts)) if count else 0.0,
                "p95": float(np.quantile(particle_counts, 0.95)) if count else 0.0,
                "maximum": int(particle_counts.max()) if count else 0,
            },
            "hlt_particles_per_jet": {
                "minimum": int(hlt_counts.min()) if count else 0,
                "median": float(np.median(hlt_counts)) if count else 0.0,
                "p95": float(np.quantile(hlt_counts, 0.95)) if count else 0.0,
                "maximum": int(hlt_counts.max()) if count else 0,
            },
            "offline_nonfinite_rates": missing_rates,
            "hlt_nonfinite_rates": hlt_missing_rates,
            "paired_labels_agree": True,
        },
        "matching": {
            "particle_coverage": total_matched / total_hlt if total_hlt else 0.0,
            "pair_coverage": (
                total_matched_pairs / total_hlt_pairs if total_hlt_pairs else 0.0
            ),
            "matched_pt_coverage": total_matched_pt / total_hlt_pt if total_hlt_pt else 0.0,
            "by_class": {
                name: {"jets": len(values), "mean_particle_coverage": mean(values)}
                for name, values in class_accumulator.items()
            },
            "by_particle_type": {
                group: {
                    "particles": int(type_totals[group]),
                    "particle_coverage": (
                        type_matches[group] / type_totals[group]
                        if type_totals[group]
                        else 0.0
                    ),
                    "matched_pt_coverage": (
                        type_matched_pt[group] / type_pt[group]
                        if type_pt[group]
                        else 0.0
                    ),
                }
                for group in ("charged", "neutral")
            },
            "by_jet_pt": {
                key: {"jets": len(values), "mean_particle_coverage": mean(values)}
                for key, values in strata["jet_pt"].items()
            },
            "by_hlt_multiplicity": {
                key: {"jets": len(values), "mean_particle_coverage": mean(values)}
                for key, values in strata["hlt_multiplicity"].items()
            },
            "construction_indices_used": False,
        },
        "capabilities": dict(CURRENT_SOURCE_CAPABILITIES),
        "physical_event_overlap": "unavailable_no_physical_event_id",
        "offline_vertex_assignments": "unavailable",
    }
    return with_content_hash(payload)


def render_data_audit_markdown(report: Mapping[str, Any]) -> str:
    inventory = report["inventory"]
    sample = report["sample"]
    matching = report["matching"]
    lines = [
        "# PRAD Data Audit",
        "",
        f"Contract: `{report['contract']}`",
        f"Content SHA-256: `{report['content_hash']}`",
        f"Split manifest SHA-256: `{report['split_manifest_sha256']}`",
        "",
        "## Inventory",
        "",
        f"Total paired jet rows: {inventory['total_paired_jets']:,}.",
        "",
        "| Class | Rows | Sample rows |",
        "|---|---:|---:|",
    ]
    for name in CLASS_LABELS:
        lines.append(
            f"| {name} | {inventory['class_counts'][name]:,} | "
            f"{sample['class_counts'][name]:,} |"
        )
    lines.extend(
        [
            "",
            "## Bounded paired-sample diagnostics",
            "",
            f"Jets audited: {sample['jets']:,}.",
            f"Paired labels agree: `{sample['paired_labels_agree']}`.",
            f"Particle match coverage: {matching['particle_coverage']:.6f}.",
            f"Ordered non-self pair coverage: {matching['pair_coverage']:.6f}.",
            f"Matched transverse-momentum coverage: {matching['matched_pt_coverage']:.6f}.",
            "Offline vertex assignments: unavailable in the authenticated source.",
            "Physical event IDs: unavailable, so cross-jet physical-event overlap cannot be tested.",
            "Direct particle identities: unavailable; the registered charged/neutral Hungarian fallback was used.",
            "Construction indices used for matching: `False`.",
            "",
            "Raw nonfinite-value rates are recorded in the adjacent authenticated JSON report. "
            "The ROOT reader fails closed on any nonfinite value, so a successful audit should report zero.",
            "",
        ]
    )
    return "\n".join(lines)


def run_prad_data_audit(
    *,
    files: Sequence[FileRecord],
    identities: Sequence[JetIdentity],
    data_root: str | Path,
    split_manifest_sha256: str,
    output_markdown: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    """Load a bounded real sample, construct HLT-v3, and publish the audit."""

    selected = tuple(identities)
    if not selected:
        raise ValueError("PRAD data audit sample must not be empty")
    offline = load_offline_view(selected, data_root=data_root)
    hlt_tokens, hlt_mask, _, _ = build_hlt_v3_view(
        offline.tokens,
        offline.mask,
        canonical_identities=[item.key for item in selected],
        logical_role="model_train",
        replica_id=0,
        realization_policy="R_FIXED",
    )
    report = summarize_prad_sample(
        files=files,
        identities=selected,
        offline_tokens=offline.tokens,
        offline_mask=offline.mask,
        labels=offline.labels,
        hlt_tokens=hlt_tokens,
        hlt_mask=hlt_mask,
        split_manifest_sha256=split_manifest_sha256,
    )
    write_immutable_json(output_json, report)
    atomic_publish_bytes(
        output_markdown,
        (render_data_audit_markdown(report) + "\n").encode("utf-8"),
    )
    return report


__all__ = [
    "PRAD_DATA_AUDIT_CONTRACT",
    "render_data_audit_markdown",
    "run_prad_data_audit",
    "summarize_prad_sample",
]
