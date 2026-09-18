"""Bounded, non-selecting distribution and two-sample diagnostics.

Classification labels are deliberately absent from this interface. All thresholds
are fit-role quantities, and none of these reports feeds the selection score.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import numpy as np

from .contracts import artifact
from .metrics import JET_JOINT, jet_summary

QUANTILES = (.001, .01, .5, .99, .999)


def scalar_thresholds(samples):
    result = {}
    for name, values in sorted(samples.items()):
        a = np.asarray(values)
        if a.ndim == 1 and len(a) and np.issubdtype(a.dtype, np.number):
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite fit-only diagnostic threshold")
            result[name] = np.quantile(a, QUANTILES).tolist()
    return result


def joint_thresholds(samples):
    """Coordinatewise tails of already registered joint observables, in raw units."""
    result = {}
    for name, values in sorted(samples.items()):
        if "joint" not in name or not len(values):
            continue
        a = np.asarray(values, dtype=float)
        if a.ndim != 2 or not np.isfinite(a).all():
            raise ValueError("Invalid fit-only joint tail threshold")
        result[name] = np.quantile(a, [.001, .999], axis=0).tolist()
    return result


def discriminator_vector(particles):
    row = jet_summary(particles)
    # Explicit undefined indicators, not fictitious measured zeroes.
    return [float(row.get(n, 0.)) for n in JET_JOINT] + [
        float(n in row) for n in JET_JOINT] + [
        float(row[f"count_{i}"]) for i in range(6)] + [
        float(particles.valid[:, i].sum()) for i in range(4)]


class TwoSample:
    """Bottom-hash paired jets per file; domain pairs never cross the split."""

    def __init__(self, cap_per_file=4096):
        if not 32 <= cap_per_file <= 4096:
            raise ValueError("Two-sample diagnostic capacity differs")
        self.cap, self.heaps, self.counts = cap_per_file, {}, {}

    def add(self, pair, proxy):
        group, jet = pair.source_group, pair.identity
        self.counts[group] = self.counts.get(group, 0)+1
        priority = int.from_bytes(hashlib.sha256(("CMS2JC2_2SAMPLE/v1:"+jet).encode()).digest(), "big")
        heap = self.heaps.setdefault(group, [])
        if len(heap) < self.cap or priority < -heap[0][0]:
            item = (-priority, jet, discriminator_vector(pair.hlt), discriminator_vector(proxy))
            if len(heap) == self.cap:
                heapq.heapreplace(heap, item)
            else:
                heapq.heappush(heap, item)

    def report(self, role):
        groups = sorted(self.heaps, key=lambda g: hashlib.sha256(("CMS2JC2_2SPLIT/v1:"+g).encode()).digest())
        common = dict(nonselecting=True, replica=0, real_source_groups=len(groups),
                      source_population=self.counts, cap_per_file=self.cap,
                      input_fields=[*JET_JOINT, "defined_indicators", "category_counts", "valid_counts"],
                      classifier=dict(kind="DecisionTreeClassifier", max_depth=3, max_leaf_nodes=8,
                                      min_samples_leaf=100, random_state=20260917),
                      split_rule="hash_files_into_halves_minimum_two_files_per_half",
                      domain_weights="equal_domains_inverse_per_file_sampling_fraction")
        if role != "response_select" or len(groups) < 4:
            return dict(**common, available=False, auc=None,
                        reason="selection_only" if role != "response_select" else "fewer_than_four_independent_files")
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import roc_auc_score
        train, test = groups[:len(groups)//2], groups[len(groups)//2:]

        def arrays(files):
            x, y, weights = [], [], []
            for g in files:
                rows = sorted(self.heaps[g], key=lambda r: (-r[0], r[1]))
                for _, _, a, b in rows:
                    x.extend((a, b)); y.extend((0, 1))
                    weights.extend([self.counts[g]/len(rows)]*2)
            return np.asarray(x), np.asarray(y), np.asarray(weights)

        x, y, w = arrays(train)
        fit = DecisionTreeClassifier(**{k: v for k, v in common["classifier"].items() if k != "kind"})
        fit.fit(x, y, sample_weight=w)
        x, y, w = arrays(test)
        return dict(**common, available=True, auc=float(roc_auc_score(y, fit.predict_proba(x)[:, 1], sample_weight=w)),
                    train_files=train, test_files=test, train_paired_jets=sum(len(self.heaps[g]) for g in train),
                    test_paired_jets=len(y)//2, fitted_classifier_exported=False)


class Diagnostics:
    def __init__(self, metric_lock, *, replicas, role, cap=16384):
        from .evaluation import ObservationSample
        self.thresholds = metric_lock["diagnostic_thresholds"]
        self.joint_thresholds = metric_lock["joint_diagnostic_thresholds"]
        self.role, self.replicas = role, list(replicas)
        self.samples = {s: ObservationSample(cap) for s in ("real", *map(str, replicas))}
        self.counters = {s: dict(jets=0, category_particles=[0]*6, category_jets=[0]*6,
                                empty_jets=0, tail_exceedances={}, joint_tail_exceedances={}) for s in self.samples}
        self.two_sample = TwoSample()

    def add(self, side, identity, particles, obs):
        counter = self.counters[side]
        counter["jets"] += 1; counter["empty_jets"] += not len(particles)
        for i in range(6):
            n = int(np.count_nonzero(particles.category == i))
            counter["category_particles"][i] += n
            counter["category_jets"][i] += int(n > 0)
        for name, thresholds in self.thresholds.items():
            values = np.asarray(obs.get(name, []), dtype=float)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Diagnostic scalar registry differs")
            if not len(values):
                continue
            self.samples[side].add(name, identity, values.tolist())
            row = counter["tail_exceedances"].setdefault(name, dict(observations=0, lower=0, upper=0, both=0))
            row["observations"] += len(values)
            row["lower"] += int(np.count_nonzero(values < thresholds[0]))
            row["upper"] += int(np.count_nonzero(values > thresholds[-1]))
            row["both"] = row["lower"]+row["upper"]
        for name, (lower, upper) in self.joint_thresholds.items():
            values = np.asarray(obs.get(name, []), dtype=float).reshape(-1, len(lower))
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite joint diagnostic values")
            row = counter["joint_tail_exceedances"].setdefault(name, dict(
                observations=0, any_coordinate=0, at_least_two_coordinates=0, all_coordinates=0))
            exceed = np.sum((values < lower) | (values > upper), axis=1)
            row["observations"] += len(values)
            row["any_coordinate"] += int(np.count_nonzero(exceed))
            row["at_least_two_coordinates"] += int(np.count_nonzero(exceed >= 2))
            row["all_coordinates"] += int(np.count_nonzero(exceed == len(lower)))

    def report(self):
        quantiles = {}
        for side, sample in self.samples.items():
            quantiles[side] = {}
            for name in self.thresholds:
                values = sample.values(name)
                n = len(values)
                quantiles[side][name] = dict(
                    values=np.quantile(values, QUANTILES).tolist() if n else None,
                    sample_records=n, population_records=sample.counts.get(name, 0),
                    # This bounds empirical-CDF sampling error, not independent-jet
                    # uncertainty. Exact tail counters above use every observation.
                    sampling_cdf_95_bound=min(1., math.sqrt(math.log(40)/(2*n))) if n else None,
                    independent_jet_confidence_interval=False)
        return artifact("NONSELECTING_DIAGNOSTICS", nonselecting=True,
                        quantile_probabilities=list(QUANTILES), quantiles=quantiles,
                        raw_observable_units=True, fit_thresholds=self.thresholds,
                        fit_joint_thresholds=self.joint_thresholds,
                        population_counters=self.counters,
                        tails_use_all_observations=True, quantiles_are_bounded_hash_sample_estimates=True,
                        two_sample=self.two_sample.report(self.role),
                        raw_records_exported=False, quality_does_not_gate_execution=True)


class BridgeAudit:
    """Fit-only transformed field audit; distributions do not certify units."""

    def __init__(self):
        from .evaluation import ObservationSample
        self.sample = ObservationSample(16384)
        self.counts = dict(jets=0, offline_particles=0, hlt_particles=0, raw_offline_charged=0,
                           raw_offline_neutral=0, excluded_lost_tracks=0, raw_hlt_particles=0,
                           rows_with_native_count_audit=0)
        self.validity = {s: np.zeros((6, 4), np.int64) for s in ("offline", "hlt")}
        self.categories = {s: np.zeros(6, np.int64) for s in ("offline", "hlt")}

    def add(self, pair):
        self.counts["jets"] += 1
        if pair.ingestion_audit is not None:
            self.counts["rows_with_native_count_audit"] += 1
            for k, value in pair.ingestion_audit.items():
                if k not in self.counts or type(value) is not int or value < 0:
                    raise ValueError("Native bridge ingestion count differs")
                self.counts[k] += value
        for side, particles in (("offline", pair.offline), ("hlt", pair.hlt)):
            self.counts[side+"_particles"] += len(particles)
            values = dict(pt_gev=particles.pt, energy_gev=particles.p4[:, 3], eta=particles.eta)
            for j, name in enumerate(("d0_mm", "dz_mm", "d0err_mm", "dzerr_mm")):
                values[name] = particles.tracking[particles.valid[:, j], j]
            for k, v in values.items():
                self.sample.add(side+":"+k, pair.identity, v.tolist())
            for i in range(6):
                keep = particles.category == i
                self.categories[side][i] += keep.sum()
                self.validity[side][i] += particles.valid[keep].sum(axis=0)

    def report(self):
        return dict(counts=self.counts, quantile_probabilities=list(QUANTILES),
                    post_bridge_quantiles={k: np.quantile(self.sample.values(k), QUANTILES).tolist()
                                           if self.sample.values(k) else None for k in self.sample.counts},
                    category_counts={s: v.tolist() for s, v in self.categories.items()},
                    valid_tracking_counts_by_category={s: v.tolist() for s, v in self.validity.items()},
                    raw_unit_evidence=False, producer_definitions_still_required=True,
                    native_counts_include_lost_tracks_but_fitted_offline_excludes_them=True,
                    sample_cap_per_coordinate=16384)
