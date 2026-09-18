"""Streaming, paired evaluation with fit-locked offline conditioning cohorts.

This module accepts bounded role readers, not filenames or a final-test route.
It never fits a generator on selection observations or persists proxy particles.
"""
from __future__ import annotations

import hashlib
import heapq
import numpy as np

from .contracts import artifact, validate
from .features import features
from .metrics import BLOCKS, Summary, fit_registry, observations
from .response import Generator
from .selection import compare_groups
from .conditioning import CONDITIONS, eligible_cells


class ObservationSample:
    """Common bottom-hash scalar/vector samples; bounded RAM, no donor export."""

    def __init__(self, cap=32768):
        if type(cap) is not int or not 32 <= cap <= 100_000:
            raise ValueError("Metric calibration capacity differs")
        self.cap, self.heaps, self.counts = cap, {}, {}

    def add(self, name, jet, values):
        heap = self.heaps.setdefault(name, [])
        self.counts[name] = self.counts.get(name, 0)+len(values)
        for i, value in enumerate(values):
            digest = int.from_bytes(hashlib.sha256(f"CMS2JC2_METRIC_RECORD/v1\0{name}\0{jet}\0{i}".encode()).digest(), "big")
            item = (-digest, jet, i, value)
            if len(heap) < self.cap:
                heapq.heappush(heap, item)
            elif digest < -heap[0][0]:
                heapq.heapreplace(heap, item)

    def values(self, name):
        return [r[3] for r in sorted(self.heaps.get(name, []), key=lambda r: (-r[0], r[1], r[2]))]


def cohort_coordinates(offline):
    x = features(offline)
    return np.median(x[:, [2, 3, 17]], axis=0).tolist() if len(x) else None


def fit_metric_lock(pairs, rules, *, membership_hash: str, cap: int = 32768, bins: int = 512) -> dict:
    """Production caller must provide the authenticated fit_location reader."""
    sample = ObservationSample(cap)
    from .diagnostics import BridgeAudit
    bridge_audit = BridgeAudit()
    jets, resolved, files = 0, 0, set()
    for pair in pairs:
        if pair.hlt is None:
            raise PermissionError("Real CMS HLT is required for fit-role metric scales")
        bridge_audit.add(pair)
        obs, audit = observations(pair.offline, pair.hlt, rules)
        for name, values in obs.items():
            if name in {n for names in BLOCKS.values() for n in names}:
                sample.add(name, pair.identity, values)
        coords = features(pair.offline)[:, [2, 3, 17]]
        sample.add("cohort_coordinates", pair.identity, coords.tolist())
        jets += 1; resolved += audit["resolved"]; files.add(pair.source_group)
        if jets % 1000 == 0:
            print(f"CMS2JC2 phase=metric_lock jets={jets}", flush=True)
    if not jets:
        raise ValueError("Metric calibration requires real fit-role jets")
    registry = fit_registry({k: sample.values(k) for k in sample.heaps if k != "cohort_coordinates"},
                            membership_hash=membership_hash, bins=bins)
    values = np.asarray(sample.values("cohort_coordinates")).reshape(-1, 3)
    edges = {name: (np.quantile(values[:, j], q).tolist() if len(values) else [0.]*len(q))
             for j, name, q in ((0, "pt", [.25, .5, .75]), (1, "eta", [.25, .5, .75]),
                                (2, "crowding", [1/3, 2/3]))}
    from .diagnostics import scalar_thresholds, joint_thresholds
    thresholds = scalar_thresholds({n: sample.values(n) for n in sample.heaps if n != "cohort_coordinates"})
    return artifact("METRIC_LOCK", parents={"membership": membership_hash,
                    "association": rules["content_hash"]}, registry=registry,
                    condition_order=list(CONDITIONS), cohort_edges=edges,
                    cohort_policy="particle_source_coordinates_jet_median_coordinates_v1",
                    unmatched_tracking="overall_only_no_invented_offline_conditioning",
                    population_policy="offline_particles_and_all_permitted_source_groups_not_actual_matches",
                    cohort_overlap="category_presence_intentionally_overlaps; numeric axes separate",
                    sparse_policy="under_1000_real_jets_backoff_to_all_once",
                    edge_ties="searchsorted_right_retaining_empty_cells", jets=jets,
                    resolved_jets=resolved, source_groups=sorted(files),
                    diagnostic_thresholds=thresholds,
                    joint_diagnostic_thresholds=joint_thresholds({n: sample.values(n) for n in sample.heaps}),
                    fit_only_bridge_audit=bridge_audit.report(),
                    sample={name: dict(population=count, retained=len(sample.heaps[name]),
                            inclusion_probability=len(sample.heaps[name])/count if count else 0.)
                            for name, count in sample.counts.items()}, raw_donor_library=False)


def cohorts(offline, lock):
    keys = eligible_cells(features(offline), lock["cohort_edges"])
    if not set(keys) <= set(lock["condition_order"]):
        raise ValueError("Conditional metric registry differs")
    return keys


def evaluate(pairs, response, lock, *, membership_hash: str, role: str,
             progress_every: int = 1000) -> dict:
    validate(lock, "METRIC_LOCK")
    if lock["condition_order"] != list(CONDITIONS):
        raise ValueError("Conditional metric cells were changed after freezing")
    generator = Generator(response)
    replicas = range(5) if role == "response_confirm" else range(3)
    from .diagnostics import Diagnostics
    diagnostics = Diagnostics(lock, replicas=replicas, role=role)
    real, proxies, flags, identities = {}, {r: {} for r in replicas}, {}, hashlib.sha256()
    classes, class_proxies = {}, {r: {} for r in replicas}
    count = 0

    def add(group, key, obs, audit):
        if key not in group:
            group[key] = Summary(lock["registry"])
        group[key].add(obs, audit)

    for pair in pairs:
        if pair.hlt is None:
            raise PermissionError("CMS closure cannot use unpaired JetClass2")
        group = real.setdefault(pair.source_group, {})
        obs, audit, conditional = observations(pair.offline, pair.hlt, response["rules"],
                                               condition_edges=lock["cohort_edges"])
        diagnostics.add("real", pair.identity, pair.hlt, obs)
        conditions = set(conditional)
        for key, row in conditional.items():
            add(group, key, row, audit)
        if pair.diagnostic_class is not None:
            add(classes, str(pair.diagnostic_class), obs, audit)
        for replica in replicas:
            proxy, info = generator(pair.offline, jet=pair.identity, replica=replica)
            obs, audit, conditional = observations(pair.offline, proxy, response["rules"],
                                                   condition_edges=lock["cohort_edges"])
            diagnostics.add(str(replica), pair.identity, proxy, obs)
            if replica == 0:
                diagnostics.two_sample.add(pair, proxy)
            if set(conditional) != conditions:
                raise ValueError("Proxy-dependent conditional membership")
            group = proxies[replica].setdefault(pair.source_group, {})
            for key, row in conditional.items():
                add(group, key, row, audit)
            if pair.diagnostic_class is not None:
                add(class_proxies[replica], str(pair.diagnostic_class), obs, audit)
            for key, value in info["flags"].items():
                flags[key] = flags.get(key, 0)+int(bool(value))
        encoded = pair.identity.encode()
        identities.update(len(encoded).to_bytes(8, "big")+encoded)
        count += 1
        if count % progress_every == 0:
            print(f"CMS2JC2 phase=evaluate role={role} jets={count}", flush=True)
    calibration_coverage = min(v["counts"]["resolved"]/v["counts"]["jets"]
                               for v in response["calibration"].values())
    result = compare_groups(real, proxies, calibration_coverage=calibration_coverage,
                            membership_hash=membership_hash, fitted_hash=response["content_hash"],
                            candidate_id=response["candidate_id"], budget=response["budget"],
                            gate=response["rules"]["gate"], role=role)
    return artifact("EVALUATION_BUNDLE", parents={"response": response["content_hash"],
                    "metric_lock": lock["content_hash"], "comparison": result["content_hash"]},
                    comparison=result, physical_status=response["physical_status"],
                    ordered_identity_sha256=identities.hexdigest(), generation_flags=flags,
                    generated_replicas_per_jet=len(replicas), durable_proxy_arrays=False,
                    absent_cohorts=[c for c in CONDITIONS if not any(c in v for v in real.values())],
                    nonselecting_class_closure={c: [summary.compare(class_proxies[r][c]) for r in replicas]
                                               for c, summary in sorted(classes.items())},
                    labelled_diagnostic_jets=sum(s.jets for s in classes.values()),
                    nonselecting_diagnostics=diagnostics.report(),
                    nonselecting_bin_occupancy={"real": occupancy(real),
                        **{str(r): occupancy(proxies[r]) for r in replicas}},
                    generation_flag_denominator="real_jets_times_replicas",
                    nonselecting_diagnostics_complete=True,
                    representative_visuals_are_separate_locked_response_task=True)


def occupancy(groups):
    """Compact exact histogram occupancy, including explicit under/overflow bins."""
    registry = next(iter(groups.values()))["all"].registry
    total = Summary(registry)
    for group in groups.values():
        total.merge(group["all"])
    result = {}
    for name, distribution in total.distributions.items():
        if distribution.definition["kind"] == "categorical":
            result[name] = dict(distinct_states=len(distribution.categories),
                                population=sum(distribution.categories.values()))
        else:
            counts = distribution.counts
            result[name] = dict(nonempty_bins_per_projection=np.count_nonzero(counts, axis=1).tolist(),
                                underflow_per_projection=counts[:, 0].tolist(),
                                overflow_per_projection=counts[:, -1].tolist(),
                                observations_per_projection=counts.sum(axis=1).tolist())
    return result
