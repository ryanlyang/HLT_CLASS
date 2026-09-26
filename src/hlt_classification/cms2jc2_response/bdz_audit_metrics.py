"""Additive diagnostics only: no response fit, clipping or candidate selection."""
from copy import deepcopy
import hashlib
import math

import numpy as np

from . import bdz_metrics as old, bdz_maps as maps

TOP = 5
NAMES = old.NAMES
SIGNED_EDGES = np.linspace(-12., 12., 49).tolist()
ERROR_EDGES = np.linspace(-18., 8., 53).tolist()
MAG_EDGES = [-8., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4.]
ERR_EDGES = [-8., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3.]


def moments(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite audit observation")
    with np.errstate(over="raise", invalid="raise"):
        return dict(count=len(x), jets=int(len(x) > 0), sum=float(x.sum()),
                    sumsq=float(np.dot(x, x)), minimum=float(x.min()) if len(x) else None,
                    maximum=float(x.max()) if len(x) else None)


def summary(m):
    n = m["count"]
    mean = m["sum"]/n if n else None
    return dict(count=n, jets=m["jets"], mean=mean,
                sd=math.sqrt(max(0., m["sumsq"]/n-mean**2)) if n else None,
                minimum=m["minimum"], maximum=m["maximum"])


def indices(x, edges, *, log=False):
    if log:
        # Zero has its own underflow; positive sub-range values also use it.
        y = np.full(len(x), -np.inf)
        positive = x > 0
        y[positive] = np.log10(x[positive])
    else:
        y = x
    return np.searchsorted(edges, y, side="right")


def variable(x, j):
    result = moments(x)
    transformed = np.log(x) if j in (2, 3) else np.arcsinh(x)
    edges = ERROR_EDGES if j in (2, 3) else SIGNED_EDGES
    result["bins"] = np.bincount(indices(transformed, edges), minlength=len(edges)+1).tolist()
    result["exceed"] = [int((abs(x) > t).sum()) for t in old.THRESHOLDS[j]]
    result["exceed_sumsq"] = [float(np.dot(x[abs(x) > t], x[abs(x) > t])) for t in old.THRESHOLDS[j]]
    return result


def correlation(x, y):
    return dict(count=len(x), x=float(x.sum()), y=float(y.sum()),
                xx=float(x@x), yy=float(y@y), xy=float(x@y))


def pearson(r):
    n = r["count"]
    if n < 2:
        return None
    a, b = r["xx"]-r["x"]**2/n, r["yy"]-r["y"]**2/n
    return float(np.clip((r["xy"]-r["x"]*r["y"]/n)/math.sqrt(a*b), -1, 1)) if min(a, b) > 0 else None


def route(mapping, pid, j, before, valid, candidate):
    if not valid:
        return "invalid"
    if mapping is None:
        return "observed"
    if candidate == "B_DZ":
        return "control_identity"
    cell = mapping["cells"][f"{pid}/{j}"]
    scope = "own"
    if cell["status"] != "fitted":
        cell, scope = mapping["cells"][f"all/{j}"], "pooled"
    if cell["status"] != "fitted":
        return "identity_fallback"
    if j < 2 and before == 0:
        return scope+"_zero_preserved"
    z = float(maps.transform(np.array([before]), j)[0])
    return scope+("_endpoint" if z < cell["x"][0] or z > cell["x"][-1] else "_interpolation")


def top(rows):
    return sorted(rows, key=lambda r: (-abs(r["significance"]), r["jet_hash"], r["particle_hash"],
                                      r.get("source_group", ""), r.get("side", "")))[:TOP]


def particles(p, *, jet, base=None, mapping=None, candidate=None, source_group="", side=""):
    """One jet/side; replicas must be passed and retained separately by caller."""
    result = {}
    base = p if base is None else base
    maps.check_invariants(base, p)
    for pid in ("all", *map(str, range(6))):
        mask = np.ones(len(p), bool) if pid == "all" else p.category == int(pid)
        v, tr = p.valid[mask], p.tracking[mask]
        xs = [tr[v[:, j], j] for j in range(4)]
        sigs = [tr[v[:, j] & v[:, j+2], j]/tr[v[:, j] & v[:, j+2], j+2] for j in range(2)]
        cell = dict(jets=1, particles=int(mask.sum()), variables={
            name: variable(x, j) for j, (name, x) in enumerate(zip(NAMES, [*xs, *sigs]))}, pairs={})
        for j, name in enumerate(("d0", "dz")):
            both = v[:, j] & v[:, j+2]
            value, error, sig = tr[both, j], tr[both, j+2], sigs[j]
            ix, iy = indices(abs(value), MAG_EDGES, log=True), indices(error, ERR_EDGES, log=True)
            grid = np.zeros((len(MAG_EDGES)+1, len(ERR_EDGES)+1), dtype=np.int64)
            np.add.at(grid, (ix, iy), 1)
            rows = []
            original = np.flatnonzero(mask)[both]
            # Pick per-jet top first; global top cannot include any other entry.
            chosen = sorted(range(len(sig)), key=lambda k: (-abs(sig[k]),
                hashlib.sha256(p.keys[original[k]].encode()).hexdigest()))[:TOP]
            for k in chosen:
                i = original[k]
                rows.append(dict(jet_hash=hashlib.sha256(jet.encode()).hexdigest(),
                    source_group=source_group, side=side,
                    particle_hash=hashlib.sha256(p.keys[i].encode()).hexdigest(), pid=int(p.category[i]),
                    pt_gev=float(p.pt[i]), tracking_mm=p.tracking[i].tolist(), valid=p.valid[i].tolist(),
                    before_mm=base.tracking[i].tolist(), significance=float(sig[k]),
                    value_route=route(mapping, int(p.category[i]), j, base.tracking[i, j], True, candidate),
                    error_route=route(mapping, int(p.category[i]), j+2, base.tracking[i, j+2], True, candidate)))
            cell["pairs"][name] = dict(
                validity=np.bincount(v[:, j].astype(int)*2+v[:, j+2], minlength=4).tolist(),
                count=len(sig), jets=int(len(sig) > 0), zero_values=int((value == 0).sum()),
                joint_bins=grid.tolist(), signed=correlation(np.arcsinh(value), np.log(error)),
                magnitude=correlation(np.log1p(abs(value)), np.log(error)),
                by_error=[moments(sig[iy == b]) for b in range(len(ERR_EDGES)+1)], examples=top(rows))
        result[pid] = cell
    return result


def merge(a, b):
    """Associative bounded accumulators; only physical floats need roundoff tolerance."""
    if a is None:
        return deepcopy(b)
    for key, value in b.items():
        if key not in a:
            a[key] = deepcopy(value)
        elif key == "examples":
            a[key] = top(a[key]+value)
        elif key in ("minimum", "maximum"):
            values = [v for v in (a[key], value) if v is not None]
            a[key] = (min(values) if key == "minimum" else max(values)) if values else None
        elif isinstance(value, dict):
            merge(a[key], value)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                if len(a[key]) != len(value):
                    raise ValueError("Audit bin registry differs")
                for x, y in zip(a[key], value):
                    merge(x, y)
            else:
                if np.shape(a[key]) != np.shape(value):
                    raise ValueError("Audit histogram shape differs")
                a[key] = (np.asarray(a[key])+np.asarray(value)).tolist()
        else:
            a[key] += value
    return a


def pooled(payload):
    total = {}
    for source in sorted(payload):
        merge(total, payload[source])
    return total


def tv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.sum() or not b.sum():
        return None  # unavailable, never disguised as perfect agreement
    return float(abs(a/a.sum()-b/b.sum()).sum()/2)


def diagnostics(payload):
    total = pooled(payload)
    out = {}
    for side, cells in total.items():
        out[side] = {}
        for pid, cell in cells.items():
            fields = {}
            for name, m in cell["variables"].items():
                fields[name] = dict(**summary(m),
                    tail_rates=[n/m["count"] if m["count"] else None for n in m["exceed"]],
                    tail_square_fractions=[n/m["sumsq"] if m["sumsq"] else None for n in m["exceed_sumsq"]],
                    pid_square_fraction=m["sumsq"]/cells["all"]["variables"][name]["sumsq"]
                        if cells["all"]["variables"][name]["sumsq"] else None,
                    histogram_tv=tv(total["real"][pid]["variables"][name]["bins"], m["bins"]))
            out[side][pid] = dict(variables=fields, pairs={name: dict(
                signed_correlation=pearson(r["signed"]), magnitude_correlation=pearson(r["magnitude"]),
                complete_particles=r["count"], complete_jets=r["jets"], validity=r["validity"],
                zero_values=r["zero_values"], by_error=[summary(m) for m in r["by_error"]],
                examples=r["examples"], top_square_fraction=sum(x["significance"]**2 for x in r["examples"])
                    /cell["variables"][name+"_significance"]["sumsq"]
                    if cell["variables"][name+"_significance"]["sumsq"] else None)
                for name, r in cell["pairs"].items()})
    return out


def pooled_proxy_diagnostics(payload):
    """Particle moments over dependent replicas, never independent-jet evidence."""
    total = pooled(payload)
    combined = {"real": total["real"]}
    for name in maps.CANDIDATES:
        merged = {}
        for replica in range(3):
            merge(merged, total[f"{name}/proxy{replica}"])
        combined[name] = merged
    result = diagnostics({"pooled": combined})
    for name in maps.CANDIDATES:
        for cell in result[name].values():
            for row in cell["variables"].values():
                row["contributing_jet_replica_exposures"] = row.pop("jets")
            for pair in cell["pairs"].values():
                pair["complete_jet_replica_exposures"] = pair.pop("complete_jets")
                for row in pair["by_error"]:
                    row["contributing_jet_replica_exposures"] = row.pop("jets")
    return result


def render(report):
    lines = [f"Frozen significance replay: {report['jets']} development jets",
             f"Historical choice retained: {report['historical_choice']} (no new selection)",
             "side / PID / field                         count      mean        SD       TV"]
    for side, cells in report["pooled_proxy_diagnostics"].items():
        for pid in ("all", "0", "4"):
            for name in ("d0_significance", "dz_significance"):
                r = cells[pid]["variables"][name]
                label = f"{side} / {pid} / {name}"
                fmt = lambda x: "n/a" if x is None else f"{x:.5g}"
                lines.append(f"{label:<42} {r['count']:7} {fmt(r['mean']):>9} {fmt(r['sd']):>9} {fmt(r['histogram_tv']):>8}")
    return "\n".join(lines+["PID 0=charged hadron, 4=muon. All six PIDs are in the JSON report.",
        "Proxy moments pool dependent replicas, not extra independent jets. SD is width, not uncertainty.",
        "Diagnostic only: no refit; confirmation/test untouched; not production qualified."])
