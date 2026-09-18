"""Frozen-response, offline-only transfer diagnostics; no native JC2 HLT route."""
from __future__ import annotations

import numpy as np

from .contracts import artifact, validate
from .features import BASIC_NAMES, NEIGHBOUR_NAMES
from .metrics import jet_summary
from .response import Generator
from .support import diagnose


def evaluate_transfer(pairs, response, *, selection: dict, claim: dict,
                      profile_hash: str, role: str, progress_every: int = 1000):
    validate(selection, "SELECTION")
    validate(claim, "TRANSFER_CLAIM", parents={"selection": selection["content_hash"],
             "response": response["content_hash"], "profile": profile_hash})
    if response["content_hash"] != selection["response_hash"] or role not in {"train", "validation"}:
        raise PermissionError("Transfer response/role is not the locked selection")
    if claim.get("role") != role or claim.get("separately_authorized") is not True:
        raise PermissionError("Transfer claim does not authorize this role")
    generator = Generator(response)
    counters = dict(jets=0, particles=0, coordinate_clamp_jets=0, unseen_state_jets=0,
                    sparse_cell_jets=0, generated_empty_jets=0)
    coordinates = np.zeros(28, np.int64)
    distances = []
    means = {s: {k: 0. for k in ("multiplicity", "pt", "mass")} for s in ("offline", "proxy")}
    groups, flags = set(), {}
    for pair in pairs:
        if pair.hlt is not None:
            raise PermissionError("Transfer may not receive native JC2 HLT particles")
        proxy, trace = generator(pair.offline, jet=pair.identity, replica=0)
        support = response.get("support_grid")
        status = (diagnose(pair.offline, support) if support is not None else
                  dict(particles=len(pair.offline), coordinate_clamps=[0]*28,
                       unseen_state_particles=len(pair.offline), sparse_cell_particles=len(pair.offline),
                       nearest_centroid_distance=None))
        coordinates += status["coordinate_clamps"]
        counters["jets"] += 1; counters["particles"] += status["particles"]
        counters["coordinate_clamp_jets"] += any(status["coordinate_clamps"])
        counters["unseen_state_jets"] += bool(status["unseen_state_particles"] or trace["flags"]["unseen_state"])
        counters["sparse_cell_jets"] += bool(status["sparse_cell_particles"])
        counters["generated_empty_jets"] += not len(proxy)
        if status["nearest_centroid_distance"] is not None:
            # Retain only streaming sufficient statistics, not per-jet vectors.
            if not distances:
                distances = [0., 0., 0.]
            distances[0] += status["nearest_centroid_distance"]
            distances[1] += 1
            distances[2] = max(distances[2], status["nearest_centroid_distance"])
        for side, particles in (("offline", pair.offline), ("proxy", proxy)):
            summary = jet_summary(particles)
            means[side]["multiplicity"] += len(particles)
            means[side]["pt"] += particles.pt.sum()
            means[side]["mass"] += summary.get("mass", 0.)
        groups.add(pair.source_group)
        for key, value in trace["flags"].items():
            flags[key] = flags.get(key, 0)+int(bool(value))
        if counters["jets"] % progress_every == 0:
            print(f"CMS2JC2 phase=transfer role={role} jets={counters['jets']}", flush=True)
    if not counters["jets"]:
        raise ValueError("Empty transfer population")
    outside = (counters["coordinate_clamp_jets"]/counters["jets"] > .05 or
               counters["unseen_state_jets"]/counters["jets"] > .01)
    return artifact("TRANSFER_REPORT", parents={"response": response["content_hash"],
                    "selection": selection["content_hash"], "claim": claim["content_hash"],
                    "profile": profile_hash}, role=role, replica=0, counts=counters,
                    source_groups=sorted(groups), per_coordinate_clamps=dict(zip(BASIC_NAMES+NEIGHBOUR_NAMES,
                                                                                coordinates.tolist())),
                    nearest_supported_centroid=dict(mean=distances[0]/distances[1], maximum=distances[2])
                                               if distances else None,
                    mean_jet_summary={s: {k: float(v/counters["jets"]) for k, v in row.items()}
                                      for s, row in means.items()}, generation_flags=flags,
                    support_status="out_of_domain" if outside else "within_registered_engineering_thresholds",
                    physically_qualified=False, physical_status=response["physical_status"],
                    detector_truth_validation=False, native_jc2_hlt_accessed=False,
                    scientific_completion=True, durable_proxy_arrays=False)
