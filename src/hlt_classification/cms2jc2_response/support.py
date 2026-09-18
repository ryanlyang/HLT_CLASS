"""Fitted aggregate support cells, never a library of CMS donor particles."""
from __future__ import annotations

import numpy as np

from .contracts import artifact
from .features import fit_preprocessing, features
from .residuals import weighted_quantiles


def fit_support(x, weights, jet_ids, *, membership_hash: str):
    x, w, ids = np.asarray(x), np.asarray(weights), np.asarray(jet_ids)
    base = fit_preprocessing(x, membership_hash=membership_hash)
    axes = [(2, [.25, .5, .75]), (3, [.25, .5, .75]), (17, [1/3, 2/3])]
    edges = [weighted_quantiles(x[:, col], w, q).tolist() for col, q in axes]
    groups = {}
    for i, row in enumerate(x):
        key = (int(row[0]), *[int(np.searchsorted(edge, row[col], side="right"))
                               for edge, (col, _) in zip(edges, axes)])
        groups.setdefault(key, []).append(i)
    cells = []
    for key, indexes in sorted(groups.items()):
        independent = len(set(ids[indexes]))
        cells.append(dict(key=list(key), sampled_independent_jets=independent,
                          estimated_records=float(w[indexes].sum()), supported=independent >= 1000,
                          centroid=np.average(x[indexes, :28], axis=0, weights=w[indexes]).tolist()
                                   if independent >= 1000 else None))
    states = sorted({(int(row[0]), int(row[1]), *(int(v) for v in row[9:13])) for row in x})
    return artifact("SUPPORT", parents={"fit_location_membership": membership_hash},
                    preprocessing=base, edges=edges, cells=cells, states=[list(s) for s in states],
                    prototype_policy="supported_cell_weighted_aggregate_centroids_not_donors_v1")


def diagnose(offline, support):
    x = features(offline)
    if not len(x):
        return dict(particles=0, coordinate_clamps=[0]*28, unseen_state_particles=0,
                    sparse_cell_particles=0, nearest_centroid_distance=None)
    base = support["preprocessing"]
    lo, hi = np.asarray(base["support_low"]), np.asarray(base["support_high"])
    states = {tuple(s) for s in support["states"]}
    cells = {tuple(c["key"]): c for c in support["cells"]}
    unknown, sparse = 0, 0
    for row in x:
        state = (int(row[0]), int(row[1]), *(int(v) for v in row[9:13]))
        unknown += state not in states
        key = (int(row[0]), *[int(np.searchsorted(edge, row[col], side="right"))
                               for edge, col in zip(support["edges"], (2, 3, 17))])
        sparse += key not in cells or not cells[key]["supported"]
    centroids = [c["centroid"] for c in support["cells"] if c["supported"]]
    nearest = None
    if centroids:
        scale = np.asarray(base["scale"][:28])
        # Bounded (particles x at most 288 cells), not full donor retrieval.
        distances = np.linalg.norm((x[:, None, :28]-np.asarray(centroids)[None])/scale, axis=2)
        nearest = float(distances.min(axis=1).mean())
    return dict(particles=len(x), coordinate_clamps=(((x < lo) | (x > hi))[:, :28].sum(axis=0)).tolist(),
                unseen_state_particles=int(unknown), sparse_cell_particles=int(sparse),
                nearest_centroid_distance=nearest)
