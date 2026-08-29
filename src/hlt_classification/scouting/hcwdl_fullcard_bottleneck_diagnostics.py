"""Bounded mergeable diagnostics for forced full-cardinality pairings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .hcwdl_fullcard_bottleneck_contracts import DR_QUANTUM
from .hcwdl_fullcard_bottleneck_matcher import PairingResult
from .highcov_data import Particles
from .highcov_features import edge_matrices, kinematics


DR_BIN_WIDTH = 1.0e-4
DR_MAX = 2.0
DR_BINS = int(DR_MAX / DR_BIN_WIDTH) + 2
THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999)
RANKS = (1, 2, 3, 5, 10)


def _bin_indexes(dr: np.ndarray) -> np.ndarray:
    return np.minimum(
        np.floor(np.asarray(dr, np.float64) / DR_BIN_WIDTH).astype(np.int64),
        DR_BINS - 1,
    )


def _histogram(dr: np.ndarray) -> np.ndarray:
    if not len(dr):
        return np.zeros(DR_BINS, np.int64)
    return np.bincount(_bin_indexes(dr), minlength=DR_BINS).astype(np.int64)


def _sparse(histogram: np.ndarray) -> list[list[int]]:
    return [[int(index), int(histogram[index])] for index in np.flatnonzero(histogram)]


def _dense(value: list[list[int]], *, bins: int = DR_BINS) -> np.ndarray:
    result = np.zeros(bins, np.int64)
    for index, count in value:
        if not 0 <= int(index) < bins or int(count) <= 0:
            raise ValueError("diagnostic sparse histogram differs")
        result[int(index)] += int(count)
    return result


def _summary(histogram: np.ndarray, *, total: int, sum_value: float, sum_sq: float,
             maximum: float, include_merge_state: bool = True) -> dict[str, Any]:
    cumulative = np.cumsum(histogram)
    quantiles: dict[str, float | None] = {}
    for probability in QUANTILES:
        if total == 0:
            quantiles[f"q{probability:g}"] = None
            continue
        target = max(1, int(np.ceil(probability * total)))
        index = int(np.searchsorted(cumulative, target, side="left"))
        quantiles[f"q{probability:g}"] = min(index * DR_BIN_WIDTH, DR_MAX)
    mean = sum_value / total if total else None
    variance = max(0.0, sum_sq / total - mean * mean) if total else None
    result = {
        "count": total,
        "mean": mean,
        "standard_deviation": None if variance is None else variance ** .5,
        "quantiles_histogram_resolution": DR_BIN_WIDTH,
        "quantiles": quantiles,
        "maximum": maximum if total else None,
        "threshold_counts": {
            f"dr_gt_{threshold:g}": int(histogram[int(threshold / DR_BIN_WIDTH) + 1:].sum())
            for threshold in THRESHOLDS
        },
        "threshold_fractions": {
            f"dr_gt_{threshold:g}": (
                int(histogram[int(threshold / DR_BIN_WIDTH) + 1:].sum()) / total
                if total else None
            )
            for threshold in THRESHOLDS
        },
    }
    if include_merge_state:
        result["merge_state"] = {
            "sparse_histogram": _sparse(histogram),
            "sum": float(sum_value), "sum_sq": float(sum_sq),
        }
    return result


@dataclass
class PairingDiagnosticsAccumulator:
    selected_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(DR_BINS, np.int64)
    )
    per_jet_max_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(DR_BINS, np.int64)
    )
    rank_histograms: dict[int, np.ndarray] = field(
        default_factory=lambda: {rank: np.zeros(DR_BINS, np.int64) for rank in RANKS}
    )
    rank_sums: dict[int, float] = field(
        default_factory=lambda: {rank: 0.0 for rank in RANKS}
    )
    rank_sum_squares: dict[int, float] = field(
        default_factory=lambda: {rank: 0.0 for rank in RANKS}
    )
    rank_maxima: dict[int, float] = field(
        default_factory=lambda: {rank: 0.0 for rank in RANKS}
    )
    selected_count: int = 0
    selected_sum: float = 0.0
    selected_sum_sq: float = 0.0
    selected_max: float = 0.0
    jet_count: int = 0
    nonempty_jet_count: int = 0
    per_jet_max_sum: float = 0.0
    per_jet_max_sum_sq: float = 0.0
    per_jet_max: float = 0.0
    visible_hlt: int = 0
    visible_offline: int = 0
    unpaired_hlt: int = 0
    unused_offline: int = 0
    identical_old_pairs: int = 0
    old_accepted_pairs: int = 0
    old_pairs_retained: int = 0
    newly_forced_pairs: int = 0
    slice_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    slice_delta_r: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    newly_forced_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(DR_BINS, np.int64)
    )
    newly_forced_sum: float = 0.0
    newly_forced_sum_sq: float = 0.0
    newly_forced_max: float = 0.0
    common_pair_delta_count: int = 0
    common_pair_delta_sum: float = 0.0
    common_pair_delta_sum_sq: float = 0.0
    per_jet_worst_delta_count: int = 0
    per_jet_worst_delta_sum: float = 0.0
    per_jet_worst_delta_sum_sq: float = 0.0
    retained_old_confidence_sum: float = 0.0
    retained_old_confidence_count: int = 0
    displaced_old_confidence_sum: float = 0.0
    displaced_old_confidence_count: int = 0

    def _slice(
        self, family: str, key: str, *, dr: np.ndarray, hlt: int, offline: int,
    ) -> None:
        selected = len(dr)
        row = self.slice_counts.setdefault(family, {}).setdefault(
            key, {"jets": 0, "selected_pairs": 0, "hlt_tokens": 0, "offline_tokens": 0},
        )
        row["jets"] += 1
        row["selected_pairs"] += selected
        row["hlt_tokens"] += hlt
        row["offline_tokens"] += offline
        state = self.slice_delta_r.setdefault(family, {}).setdefault(key, {
            "histogram": np.zeros(DR_BINS, np.int64), "sum": 0.0,
            "sum_sq": 0.0, "maximum": 0.0,
        })
        if selected:
            state["histogram"] += _histogram(dr)
            state["sum"] += float(np.sum(dr))
            state["sum_sq"] += float(np.sum(np.square(dr)))
            state["maximum"] = max(float(state["maximum"]), float(np.max(dr)))

    def add(
        self,
        *,
        result: PairingResult,
        hlt: Particles,
        offline: Particles,
        jet_class: int,
        old_native_mapping: np.ndarray | None = None,
        old_confidence: np.ndarray | None = None,
    ) -> None:
        nh, no = len(hlt.p4), len(offline.p4)
        accepted = np.asarray(result.pairing_validity, bool)
        selected_qdr = np.asarray(result.selected_qdr, np.int64)[accepted]
        dr = selected_qdr.astype(np.float64) * DR_QUANTUM
        self.jet_count += 1
        self.visible_hlt += nh
        self.visible_offline += no
        self.unpaired_hlt += max(nh - no, 0)
        self.unused_offline += max(no - nh, 0)
        self.selected_count += len(dr)
        self.selected_sum += float(dr.sum())
        self.selected_sum_sq += float(np.square(dr).sum())
        if len(dr):
            self.nonempty_jet_count += 1
            maximum = float(np.max(dr))
            self.selected_max = max(self.selected_max, maximum)
            self.per_jet_max = max(self.per_jet_max, maximum)
            self.per_jet_max_sum += maximum
            self.per_jet_max_sum_sq += maximum * maximum
            self.selected_histogram += _histogram(dr)
            self.per_jet_max_histogram += _histogram(np.asarray([maximum]))
            descending = np.sort(dr)[::-1]
            for rank in RANKS:
                if len(descending) >= rank:
                    self.rank_histograms[rank] += _histogram(
                        np.asarray([descending[rank - 1]])
                    )
                    value = float(descending[rank - 1])
                    self.rank_sums[rank] += value
                    self.rank_sum_squares[rank] += value * value
                    self.rank_maxima[rank] = max(self.rank_maxima[rank], value)

        _, h_eta, _, _ = kinematics(hlt.p4) if nh else (
            np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        )
        jet_p4 = np.sum(hlt.p4, axis=0) if nh else np.zeros(4)
        jet_pt = float(np.hypot(jet_p4[0], jet_p4[1]))
        jet_eta = float(np.arcsinh(jet_p4[2] / max(jet_pt, 1e-12)))
        multiplicity_bin = f"h{(nh // 10) * 10:03d}_o{(no // 10) * 10:03d}"
        self._slice("jet_class", str(int(jet_class)), dr=dr, hlt=nh, offline=no)
        self._slice("jet_pt", str(int(jet_pt // 100) * 100), dr=dr, hlt=nh, offline=no)
        self._slice("abs_jet_eta", f"{int(abs(jet_eta) / .5) * .5:.1f}", dr=dr, hlt=nh, offline=no)
        self._slice("multiplicity", multiplicity_bin, dr=dr, hlt=nh, offline=no)
        self._slice("hlt_multiplicity", str((nh // 10) * 10), dr=dr, hlt=nh, offline=no)
        self._slice(
            "offline_multiplicity", str((no // 10) * 10), dr=dr,
            hlt=nh, offline=no,
        )
        self._slice("count_imbalance", str(nh - no), dr=dr, hlt=nh, offline=no)
        del h_eta
        for category in np.unique(hlt.category):
            mask = (hlt.category == category) & accepted
            self._slice(
                "hlt_category", str(int(category)),
                dr=np.asarray(result.selected_qdr)[mask].astype(np.float64) * DR_QUANTUM,
                hlt=int(np.count_nonzero(hlt.category == category)), offline=0,
            )
        valid_charge = np.isin(hlt.charge, (-1, 0, 1))
        for state_name, state_mask in (("valid", valid_charge), ("invalid", ~valid_charge)):
            mask = state_mask & accepted
            self._slice(
                "hlt_charge_validity", state_name,
                dr=np.asarray(result.selected_qdr)[mask].astype(np.float64) * DR_QUANTUM,
                hlt=int(np.count_nonzero(state_mask)), offline=0,
            )
        if nh:
            hlt_pt = np.hypot(hlt.p4[:, 0], hlt.p4[:, 1])
            order = np.argsort(-hlt_pt, kind="stable")
            ranks = np.empty(nh, np.int64); ranks[order] = np.arange(nh)
            for index in np.flatnonzero(accepted):
                rank = int(ranks[index]) + 1
                quantile = min(9, int(10 * (rank - 1) / max(1, nh)))
                token_dr = np.asarray([result.selected_qdr[index] * DR_QUANTUM])
                self._slice(
                    "hlt_pt_rank", f"rank_{min(rank, 10):02d}" if rank <= 10 else "rank_gt10",
                    dr=token_dr, hlt=1, offline=0,
                )
                self._slice(
                    "hlt_pt_rank_quantile", f"decile_{quantile}",
                    dr=token_dr, hlt=1, offline=0,
                )

        if old_native_mapping is not None:
            old = np.asarray(old_native_mapping, np.int64)
            new = np.asarray(result.native_offline_index, np.int64)
            if old.shape != new.shape:
                raise ValueError("old/new assignment HLT skeleton differs")
            old_selected = old >= 0
            self.old_accepted_pairs += int(np.count_nonzero(old_selected))
            self.old_pairs_retained += int(np.count_nonzero(old_selected & (old == new)))
            self.identical_old_pairs += int(np.count_nonzero((new >= 0) & (old == new)))
            self.newly_forced_pairs += int(np.count_nonzero((new >= 0) & ~old_selected))
            matrices = edge_matrices(hlt, offline)
            new_selected = new >= 0
            new_local = np.asarray(result.concatenated_offline_index, np.int64)
            if new_local.shape != new.shape:
                raise ValueError("new native/concatenated assignment skeleton differs")
            native = (
                np.asarray(offline.native_index, np.int64)
                if offline.native_index is not None
                else np.arange(no, dtype=np.int64)
            )
            native_to_local = {int(value): index for index, value in enumerate(native)}
            old_local = np.full(old.shape, -1, np.int64)
            for index in np.flatnonzero(old_selected):
                if int(old[index]) not in native_to_local:
                    raise ValueError("established assignment native index is absent")
                old_local[index] = native_to_local[int(old[index])]
            forced = new_selected & ~old_selected
            forced_dr = matrices.dr[np.flatnonzero(forced), new_local[forced]]
            if len(forced_dr):
                self.newly_forced_histogram += _histogram(forced_dr)
                self.newly_forced_sum += float(forced_dr.sum())
                self.newly_forced_sum_sq += float(np.square(forced_dr).sum())
                self.newly_forced_max = max(self.newly_forced_max, float(forced_dr.max()))
            common = old_selected & new_selected
            if np.any(common):
                indexes = np.flatnonzero(common)
                changes = (
                    matrices.dr[indexes, new_local[common]]
                    - matrices.dr[indexes, old_local[common]]
                )
                self.common_pair_delta_count += len(changes)
                self.common_pair_delta_sum += float(changes.sum())
                self.common_pair_delta_sum_sq += float(np.square(changes).sum())
            if np.any(old_selected) and np.any(new_selected):
                old_worst = float(np.max(
                    matrices.dr[np.flatnonzero(old_selected), old_local[old_selected]]
                ))
                new_worst = float(np.max(
                    matrices.dr[np.flatnonzero(new_selected), new_local[new_selected]]
                ))
                self.per_jet_worst_delta_count += 1
                worst_delta = new_worst - old_worst
                self.per_jet_worst_delta_sum += worst_delta
                self.per_jet_worst_delta_sum_sq += worst_delta * worst_delta
            if old_confidence is not None:
                confidence = np.asarray(old_confidence, np.float64)
                if confidence.shape != old.shape or not np.isfinite(confidence).all():
                    raise ValueError("old confidence comparison shape differs")
                retained = old_selected & (old == new)
                displaced = old_selected & (old != new)
                self.retained_old_confidence_sum += float(confidence[retained].sum())
                self.retained_old_confidence_count += int(np.count_nonzero(retained))
                self.displaced_old_confidence_sum += float(confidence[displaced].sum())
                self.displaced_old_confidence_count += int(np.count_nonzero(displaced))
                confidence_bins = (
                    ("lt_0p25", confidence < .25),
                    ("0p25_to_0p50", (confidence >= .25) & (confidence < .50)),
                    ("0p50_to_0p75", (confidence >= .50) & (confidence < .75)),
                    ("ge_0p75", confidence >= .75),
                )
                for name, confidence_mask in confidence_bins:
                    selected_mask = confidence_mask & new_selected
                    confidence_dr = matrices.dr[
                        np.flatnonzero(selected_mask), new_local[selected_mask]
                    ]
                    self._slice(
                        "established_confidence", name, dr=confidence_dr,
                        hlt=int(np.count_nonzero(confidence_mask)), offline=0,
                    )

    def payload(self) -> dict[str, Any]:
        selected = _summary(
            self.selected_histogram, total=self.selected_count,
            sum_value=self.selected_sum, sum_sq=self.selected_sum_sq,
            maximum=self.selected_max,
        )
        jet_max = _summary(
            self.per_jet_max_histogram, total=self.nonempty_jet_count,
            sum_value=self.per_jet_max_sum, sum_sq=self.per_jet_max_sum_sq,
            maximum=self.per_jet_max,
        )
        return {
            "jets": self.jet_count,
            "nonempty_jets": self.nonempty_jet_count,
            "visible_hlt_tokens": self.visible_hlt,
            "visible_offline_tokens": self.visible_offline,
            "selected_pairs": self.selected_count,
            "unavoidable_unpaired_hlt_tokens": self.unpaired_hlt,
            "unused_offline_tokens": self.unused_offline,
            "smaller_side_coverage": 1.0 if self.jet_count else None,
            "selected_delta_r": selected,
            "per_jet_max_delta_r": jet_max,
            "rank_profiles": {
                str(rank): _summary(
                    histogram, total=int(histogram.sum()),
                    sum_value=self.rank_sums[rank],
                    sum_sq=self.rank_sum_squares[rank],
                    maximum=self.rank_maxima[rank],
                )
                for rank, histogram in self.rank_histograms.items()
            },
            "old_matcher_comparison": {
                "old_accepted_pairs": self.old_accepted_pairs,
                "old_accepted_pairs_retained": self.old_pairs_retained,
                "identical_selected_pairs": self.identical_old_pairs,
                "newly_forced_pairs": self.newly_forced_pairs,
            },
            "slices": {
                family: {
                    key: {
                        **self.slice_counts[family][key],
                        "selected_delta_r": _summary(
                            state["histogram"],
                            total=self.slice_counts[family][key]["selected_pairs"],
                            sum_value=state["sum"], sum_sq=state["sum_sq"],
                            maximum=state["maximum"],
                        ),
                        "hlt_coverage": (
                            self.slice_counts[family][key]["selected_pairs"]
                            / self.slice_counts[family][key]["hlt_tokens"]
                            if self.slice_counts[family][key]["hlt_tokens"] else None
                        ),
                        "offline_coverage": (
                            self.slice_counts[family][key]["selected_pairs"]
                            / self.slice_counts[family][key]["offline_tokens"]
                            if self.slice_counts[family][key]["offline_tokens"] else None
                        ),
                    }
                    for key, state in rows.items()
                }
                for family, rows in self.slice_delta_r.items()
            },
            "forced_pair_delta_r": _summary(
                self.newly_forced_histogram, total=self.newly_forced_pairs,
                sum_value=self.newly_forced_sum, sum_sq=self.newly_forced_sum_sq,
                maximum=self.newly_forced_max,
            ),
            "common_pair_delta_r_change": {
                "count": self.common_pair_delta_count,
                "mean": (
                    self.common_pair_delta_sum / self.common_pair_delta_count
                    if self.common_pair_delta_count else None
                ),
                "standard_deviation": (
                    max(0.0, self.common_pair_delta_sum_sq / self.common_pair_delta_count
                        - (self.common_pair_delta_sum / self.common_pair_delta_count) ** 2) ** .5
                    if self.common_pair_delta_count else None
                ),
                "merge_state": {
                    "sum": self.common_pair_delta_sum,
                    "sum_sq": self.common_pair_delta_sum_sq,
                },
            },
            "per_jet_worst_delta_r_change": {
                "count": self.per_jet_worst_delta_count,
                "mean": (
                    self.per_jet_worst_delta_sum / self.per_jet_worst_delta_count
                    if self.per_jet_worst_delta_count else None
                ),
                "standard_deviation": (
                    max(
                        0.0,
                        self.per_jet_worst_delta_sum_sq
                        / self.per_jet_worst_delta_count
                        - (self.per_jet_worst_delta_sum
                           / self.per_jet_worst_delta_count) ** 2,
                    ) ** .5
                    if self.per_jet_worst_delta_count else None
                ),
                "merge_state": {
                    "sum": self.per_jet_worst_delta_sum,
                    "sum_sq": self.per_jet_worst_delta_sum_sq,
                },
            },
            "established_confidence_comparison_only": {
                "retained_count": self.retained_old_confidence_count,
                "retained_mean": (
                    self.retained_old_confidence_sum / self.retained_old_confidence_count
                    if self.retained_old_confidence_count else None
                ),
                "displaced_count": self.displaced_old_confidence_count,
                "displaced_mean": (
                    self.displaced_old_confidence_sum / self.displaced_old_confidence_count
                    if self.displaced_old_confidence_count else None
                ),
                "new_confidence_calibrated": False,
                "merge_state": {
                    "retained_sum": self.retained_old_confidence_sum,
                    "displaced_sum": self.displaced_old_confidence_sum,
                },
            },
            "bounded_histogram": {
                "bin_width": DR_BIN_WIDTH,
                "maximum_finite_bin": DR_MAX,
                "overflow_bin": True,
            },
        }


def merge_diagnostic_payloads(values: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Exactly merge bounded source histograms and integer coverage counters."""

    if not values:
        raise ValueError("diagnostic merge requires at least one source")
    totals = {
        name: sum(int(value[name]) for value in values)
        for name in (
            "jets", "nonempty_jets", "visible_hlt_tokens",
            "visible_offline_tokens", "selected_pairs",
            "unavoidable_unpaired_hlt_tokens", "unused_offline_tokens",
        )
    }
    if totals["selected_pairs"] != totals["visible_hlt_tokens"] - totals[
        "unavoidable_unpaired_hlt_tokens"
    ]:
        raise ValueError("merged diagnostic HLT cardinality differs")
    if totals["selected_pairs"] != totals["visible_offline_tokens"] - totals[
        "unused_offline_tokens"
    ]:
        raise ValueError("merged diagnostic offline cardinality differs")
    def merged_summary(path: tuple[str, ...]) -> dict[str, Any]:
        rows = []
        for value in values:
            row: Mapping[str, Any] = value
            for name in path:
                row = row[name]
            rows.append(row)
        histogram = sum(
            (_dense(row["merge_state"]["sparse_histogram"]) for row in rows),
            start=np.zeros(DR_BINS, np.int64),
        )
        return _summary(
            histogram, total=sum(int(row["count"]) for row in rows),
            sum_value=sum(float(row["merge_state"]["sum"]) for row in rows),
            sum_sq=sum(float(row["merge_state"]["sum_sq"]) for row in rows),
            maximum=max((float(row["maximum"] or 0.0) for row in rows), default=0.0),
        )

    families = sorted(set().union(*(value.get("slices", {}) for value in values)))
    slices = {}
    for family in families:
        keys = sorted(set().union(*(
            value.get("slices", {}).get(family, {}) for value in values
        )))
        slices[family] = {}
        for key in keys:
            rows = [
                value["slices"][family][key]
                for value in values if key in value.get("slices", {}).get(family, {})
            ]
            histogram = sum(
                (_dense(row["selected_delta_r"]["merge_state"]["sparse_histogram"])
                 for row in rows), start=np.zeros(DR_BINS, np.int64),
            )
            selected = sum(int(row["selected_pairs"]) for row in rows)
            hlt = sum(int(row["hlt_tokens"]) for row in rows)
            slices[family][key] = {
                name: sum(int(row[name]) for row in rows)
                for name in ("jets", "selected_pairs", "hlt_tokens", "offline_tokens")
            }
            slices[family][key]["hlt_coverage"] = selected / hlt if hlt else None
            offline = sum(int(row["offline_tokens"]) for row in rows)
            slices[family][key]["offline_coverage"] = (
                selected / offline if offline else None
            )
            slices[family][key]["selected_delta_r"] = _summary(
                histogram, total=selected,
                sum_value=sum(float(row["selected_delta_r"]["merge_state"]["sum"]) for row in rows),
                sum_sq=sum(float(row["selected_delta_r"]["merge_state"]["sum_sq"]) for row in rows),
                maximum=max(float(row["selected_delta_r"]["maximum"] or 0.0) for row in rows),
            )
    old = {
        name: sum(int(value["old_matcher_comparison"][name]) for value in values)
        for name in (
            "old_accepted_pairs", "old_accepted_pairs_retained",
            "identical_selected_pairs", "newly_forced_pairs",
        )
    }
    def merge_scalar_moments(
        name: str, *, include_standard_deviation: bool = True,
    ) -> dict[str, Any]:
        rows = [value[name] for value in values]
        count = sum(int(row["count"]) for row in rows)
        sum_value = sum(float(row["merge_state"]["sum"]) for row in rows)
        sum_sq = sum(float(row["merge_state"]["sum_sq"]) for row in rows)
        result: dict[str, Any] = {
            "count": count,
            "mean": sum_value / count if count else None,
            "merge_state": {"sum": sum_value, "sum_sq": sum_sq},
        }
        if include_standard_deviation:
            result["standard_deviation"] = (
                max(0.0, sum_sq / count - (sum_value / count) ** 2) ** .5
                if count else None
            )
        return result

    confidence_rows = [
        value["established_confidence_comparison_only"] for value in values
    ]
    retained_count = sum(int(row["retained_count"]) for row in confidence_rows)
    displaced_count = sum(int(row["displaced_count"]) for row in confidence_rows)
    retained_sum = sum(
        float(row["merge_state"]["retained_sum"]) for row in confidence_rows
    )
    displaced_sum = sum(
        float(row["merge_state"]["displaced_sum"]) for row in confidence_rows
    )
    confidence = {
        "retained_count": retained_count,
        "retained_mean": retained_sum / retained_count if retained_count else None,
        "displaced_count": displaced_count,
        "displaced_mean": displaced_sum / displaced_count if displaced_count else None,
        "new_confidence_calibrated": False,
        "merge_state": {
            "retained_sum": retained_sum, "displaced_sum": displaced_sum,
        },
    }
    old["old_hlt_coverage"] = (
        old["old_accepted_pairs"] / totals["visible_hlt_tokens"]
        if totals["visible_hlt_tokens"] else None
    )
    old["new_hlt_coverage"] = (
        totals["selected_pairs"] / totals["visible_hlt_tokens"]
        if totals["visible_hlt_tokens"] else None
    )
    old["old_offline_coverage"] = (
        old["old_accepted_pairs"] / totals["visible_offline_tokens"]
        if totals["visible_offline_tokens"] else None
    )
    old["new_offline_coverage"] = (
        totals["selected_pairs"] / totals["visible_offline_tokens"]
        if totals["visible_offline_tokens"] else None
    )
    old["identical_selected_pair_fraction"] = (
        old["identical_selected_pairs"] / totals["selected_pairs"]
        if totals["selected_pairs"] else None
    )
    old["old_accepted_pair_retention_fraction"] = (
        old["old_accepted_pairs_retained"] / old["old_accepted_pairs"]
        if old["old_accepted_pairs"] else None
    )
    return {
        **totals,
        "smaller_side_coverage": 1.0,
        "source_count": len(values),
        "selected_delta_r": merged_summary(("selected_delta_r",)),
        "per_jet_max_delta_r": merged_summary(("per_jet_max_delta_r",)),
        "rank_profiles": {
            str(rank): merged_summary(("rank_profiles", str(rank))) for rank in RANKS
        },
        "forced_pair_delta_r": merged_summary(("forced_pair_delta_r",)),
        "common_pair_delta_r_change": merge_scalar_moments(
            "common_pair_delta_r_change"
        ),
        "per_jet_worst_delta_r_change": merge_scalar_moments(
            "per_jet_worst_delta_r_change"
        ),
        "established_confidence_comparison_only": confidence,
        "old_matcher_comparison": old,
        "slices": slices,
        "bounded_histogram": values[0]["bounded_histogram"],
    }


__all__ = [
    "DR_BIN_WIDTH", "DR_MAX", "PairingDiagnosticsAccumulator",
    "merge_diagnostic_payloads",
]
