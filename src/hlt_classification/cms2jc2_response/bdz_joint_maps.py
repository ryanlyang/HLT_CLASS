"""New PID-safe and error-conditional significance maps; old maps stay frozen."""
from copy import deepcopy

import numpy as np

from . import bdz_maps as old
from .bridge import Particles
from .contracts import artifact, validate

CANDIDATES = (*old.CANDIDATES, "PID_SAFE", "JOINT")
MIN_PAIR_JETS = 1000
MIN_BIN_JETS = 500
QUARTILES = (.25, .5, .75)
MAX_BYTES = old.MAX_BYTES


def observations(p):
    result = {}
    for pid in range(6):
        for j in range(2):
            mask = (p.category == pid) & p.valid[:, j] & p.valid[:, j+2]
            if mask.any():
                result[f"{pid}/{j}"] = np.column_stack((
                    np.log(p.tracking[mask, j+2]),
                    np.arcsinh(p.tracking[mask, j]/p.tracking[mask, j+2])))
    return result


def fit_cell(real, proxy, minimum):
    # Columns: transformed coordinate, exact integer jet ordinal stored in float64.
    for a in (real, proxy):
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
            raise ValueError("Nonfinite or malformed joint calibration data")
        if np.any(a[:, 1] < 0) or np.any(a[:, 1] != np.floor(a[:, 1])):
            raise ValueError("Joint calibration jet ordinals must be exact nonnegative integers")
    rj, pj = (len(np.unique(a[:, 1])) for a in (real, proxy))
    row = dict(real_jets=rj, proxy_jets=pj, real_particles=len(real),
               proxy_particles=len(proxy), x=[], y=[])
    if min(rj, pj) < minimum:
        return dict(row, status="insufficient_unique_jets")
    x, y = (np.quantile(a[:, 0], old.QUANTILES) for a in (proxy, real))
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        return dict(row, status="degenerate_source")
    return dict(row, status="fitted", x=unique.tolist(),
                y=(np.bincount(inverse, weights=y)/counts).tolist())


def fit(arrays, *, model_hash, samples_hash):
    cells = {}
    for pid in range(6):
        for j in range(2):
            key = f"{pid}/{j}"
            # log(error), asinh(significance), jet ordinal. No rows are durable.
            real, proxy = (np.concatenate(arrays.get(s+"/"+key, [np.empty((0, 3))]))
                           for s in ("real", "proxy"))
            for a in (real, proxy):
                if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
                    raise ValueError("Nonfinite or malformed joint pair observations")
            error = fit_cell(real[:, [0, 2]], proxy[:, [0, 2]], MIN_PAIR_JETS)
            edges = {s: np.quantile(a[:, 0], QUARTILES).tolist() if len(a) else []
                     for s, a in (("real", real), ("proxy", proxy))}
            bins = []
            for b in range(4):
                parts = []
                for s, a in (("real", real), ("proxy", proxy)):
                    mask = (np.searchsorted(edges[s], a[:, 0], side="right") == b) & (a[:, 1] != 0)
                    parts.append(a[mask][:, [1, 2]])
                bins.append(fit_cell(*parts, MIN_BIN_JETS))
            cells[key] = dict(error=error, edges=edges, significance=bins)
    value = artifact("BDZ_JOINT_MAP", parents={"model": model_hash, "samples": samples_hash},
        cells=cells, quantiles=list(old.QUANTILES), quartiles=list(QUARTILES),
        minimum_pair_jets=MIN_PAIR_JETS, minimum_bin_jets=MIN_BIN_JETS,
        transform="log_mm_error_asinh_dimensionless_significance", bin_ties="right",
        fallback="identity_pair_no_pooling", extrapolation="constant_endpoint",
        preserve_zero_values=True)
    validate_map(value)
    return value


def validate_cell(cell, minimum):
    for side in ("real", "proxy"):
        n, jets = cell[side+"_particles"], cell[side+"_jets"]
        if type(n) is not int or type(jets) is not int or not 0 <= jets <= n:
            raise ValueError("Invalid joint support")
    x, y = np.asarray(cell["x"]), np.asarray(cell["y"])
    supported = min(cell["real_jets"], cell["proxy_jets"]) >= minimum
    if cell["status"] == "fitted":
        if (not supported or x.ndim != 1 or x.shape != y.shape or len(x) < 2
                or not np.isfinite([x, y]).all() or (np.diff(x) <= 0).any() or (np.diff(y) < 0).any()):
            raise ValueError("Invalid monotone joint map")
    elif (cell["status"] not in ("insufficient_unique_jets", "degenerate_source") or x.size or y.size
          or (cell["status"] == "degenerate_source") != supported):
        raise ValueError("Invalid joint fallback status")


def validate_map(value):
    validate(value, "BDZ_JOINT_MAP")
    expected = dict(quantiles=list(old.QUANTILES), quartiles=list(QUARTILES),
        minimum_pair_jets=MIN_PAIR_JETS, minimum_bin_jets=MIN_BIN_JETS,
        transform="log_mm_error_asinh_dimensionless_significance", bin_ties="right",
        fallback="identity_pair_no_pooling", extrapolation="constant_endpoint", preserve_zero_values=True)
    if (any(value.get(k) != v for k, v in expected.items())
            or set(value["cells"]) != {f"{p}/{j}" for p in range(6) for j in range(2)}):
        raise ValueError("Joint map interface differs")
    for cell in value["cells"].values():
        validate_cell(cell["error"], MIN_PAIR_JETS)
        if len(cell["significance"]) != 4 or set(cell["edges"]) != {"real", "proxy"}:
            raise ValueError("Joint rank bins differ")
        for side, edges in cell["edges"].items():
            if (len(edges) != (3 if cell["error"][side+"_particles"] else 0)
                    or not np.isfinite(edges).all() or (np.diff(edges) < 0).any()):
                raise ValueError("Invalid joint error edges")
        for b in cell["significance"]:
            validate_cell(b, MIN_BIN_JETS)
            for side in ("real", "proxy"):
                if any(b[side+"_"+k] > cell["error"][side+"_"+k] for k in ("jets", "particles")):
                    raise ValueError("Conditional support exceeds PID support")
        for side in ("real", "proxy"):
            if sum(b[side+"_particles"] for b in cell["significance"]) > cell["error"][side+"_particles"]:
                raise ValueError("Conditional particles exceed PID support")


def safe_mapping(mapping):
    """Disable pooled routes in a temporary view; never mutate a donor artifact."""
    from .contracts import with_content_hash
    result = deepcopy(mapping)
    for j in range(4):
        result["cells"][f"all/{j}"] = dict(real_jets=0, proxy_jets=0,
            real_particles=0, proxy_particles=0, x=[], y=[], status="insufficient_unique_jets")
    return with_content_hash(result)


def route(mapping, pid, j, tracking, valid):
    if not valid[j]:
        return "invalid"
    coordinate = j % 2
    if not (valid[coordinate] and valid[coordinate+2]):
        return "identity_incomplete_pair"
    cell = mapping["cells"][f"{pid}/{coordinate}"]
    if cell["error"]["status"] != "fitted":
        return "identity_pid_pair"
    e = np.log(tracking[coordinate+2])
    ec = cell["error"]
    er = "endpoint" if e < ec["x"][0] or e > ec["x"][-1] else "interpolation"
    if tracking[coordinate] == 0:
        return "zero_preserved" if j < 2 else "own_error_"+er
    b = int(np.searchsorted(cell["edges"]["proxy"], e, side="right"))
    sc = cell["significance"][b]
    if sc["status"] != "fitted":
        return f"identity_rank_bin_{b}"
    sig = np.arcsinh(tracking[coordinate]/tracking[coordinate+2])
    sr = "endpoint" if sig < sc["x"][0] or sig > sc["x"][-1] else "interpolation"
    return "own_error_"+er if j >= 2 else f"joint_bin_{b}_{sr}"


def apply(base, historical, joint, candidate):
    if candidate in old.CANDIDATES:
        return old.apply(base, historical, candidate)
    if candidate == "PID_SAFE":
        out, counts = old.apply(base, safe_mapping(historical), "TRACK_FULL")
        # 'pooled_fallback' in old counters means attempted lookup, not application.
        for row in counts.values():
            row["unsupported_pid_identity"] = row.pop("pooled_fallback")
        return out, counts
    if candidate != "JOINT":
        raise ValueError("Unregistered joint repair candidate")
    track, counters = base.tracking.copy(), {}
    for pid in range(6):
        for j in range(2):
            mask = (base.category == pid) & base.valid[:, j] & base.valid[:, j+2]
            if not mask.any():
                continue
            cell = joint["cells"][f"{pid}/{j}"]
            counts = dict(complete=int(mask.sum()), mapped=0, identity=0, zero_preserved=0,
                          error_endpoint=0, significance_endpoint=0)
            counters[f"{pid}/{j}"] = counts
            if cell["error"]["status"] != "fitted":
                counts["identity"] = counts["complete"]
                continue
            ix = np.flatnonzero(mask)
            errs = np.log(base.tracking[ix, j+2])
            sigs = np.arcsinh(base.tracking[ix, j]/base.tracking[ix, j+2])
            bins = np.searchsorted(cell["edges"]["proxy"], errs, side="right")
            zeros = sigs == 0
            ec = cell["error"]
            for b in range(4):
                sc = cell["significance"][b]
                active = (bins == b) & (zeros | (sc["status"] == "fitted"))
                if not active.any():
                    continue
                indices, e, s = ix[active], errs[active], sigs[active].copy()
                nonzero = s != 0
                if nonzero.any():
                    counts["significance_endpoint"] += int(((s[nonzero] < sc["x"][0]) | (s[nonzero] > sc["x"][-1])).sum())
                    s[nonzero] = np.interp(s[nonzero], sc["x"], sc["y"])
                counts["error_endpoint"] += int(((e < ec["x"][0]) | (e > ec["x"][-1])).sum())
                with np.errstate(over="raise", invalid="raise", under="raise"):
                    error = np.exp(np.interp(e, ec["x"], ec["y"]))
                    track[indices, j+2] = error
                    track[indices, j] = error*np.sinh(s)
                counts["mapped"] += len(indices)
                counts["zero_preserved"] += int((~nonzero).sum())
            counts["identity"] = counts["complete"]-counts["mapped"]
    output = Particles(base.p4, base.charge, base.category, track, base.valid, base.keys)
    old.check_invariants(base, output)
    return output, counters
