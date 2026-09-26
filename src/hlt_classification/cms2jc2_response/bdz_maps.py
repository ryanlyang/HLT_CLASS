"""Frozen tracking-only quantile calibration; never a new categorical fit."""
import numpy as np

from .bridge import Particles, TRACKING
from .contracts import artifact, validate

CANDIDATES = ("B_DZ", "TRACK_HALF", "TRACK_FULL")
STRENGTHS = dict(zip(CANDIDATES, (0., .5, 1.)))
QUANTILES = (0., .0001, .001, .005, .01, .025, .05, .1, .25, .5,
             .75, .9, .95, .975, .99, .995, .999, .9999, 1.)
MIN_JETS = 1000
MAX_BYTES = 2 * 1024**3


def transform(values, coordinate):
    return np.arcsinh(values) if coordinate < 2 else np.log(values)


def observations(particles):
    """Temporary RAM arrays; no pairwise correspondence is assumed."""
    result = {}
    for pid in ("all", *map(str, range(6))):
        mask = np.ones(len(particles), bool) if pid == "all" else particles.category == int(pid)
        for j in range(4):
            values = particles.tracking[mask & particles.valid[:, j], j]
            if len(values):
                result[f"{pid}/{j}"] = transform(values, j)
    return result


def fit_cell(real, proxy, real_jets, proxy_jets):
    row = dict(real_jets=real_jets, proxy_jets=proxy_jets,
               real_particles=len(real), proxy_particles=len(proxy), x=[], y=[])
    if min(real_jets, proxy_jets) < MIN_JETS:
        return dict(row, status="insufficient_unique_jets")
    if not np.isfinite(real).all() or not np.isfinite(proxy).all():
        raise ValueError("Nonfinite calibration observations")
    x, y = np.quantile(proxy, QUANTILES), np.quantile(real, QUANTILES)
    unique, indices, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        return dict(row, status="degenerate_source")
    targets = np.bincount(indices, weights=y) / counts
    return dict(row, status="fitted", x=unique.tolist(), y=targets.tolist())


def fit(samples, supports, *, model_hash, samples_hash):
    cells = {}
    for pid in ("all", *map(str, range(6))):
        for j in range(4):
            key = f"{pid}/{j}"
            a, b = (np.concatenate(samples.get(side+"/"+key, [np.empty(0)])) for side in ("real", "proxy"))
            cells[key] = fit_cell(a, b, supports.get("real/"+key, 0), supports.get("proxy/"+key, 0))
    value = artifact("BDZ_MAP", parents={"model": model_hash, "samples": samples_hash},
                     cells=cells, quantiles=list(QUANTILES), minimum_unique_jets=MIN_JETS,
                     transform="asinh_mm_values_log_mm_errors", preserve_zero_values=True,
                     extrapolation="constant_endpoint", ties="mean_target_at_identical_source_knots")
    validate_map(value)
    return value


def validate_map(value):
    validate(value, "BDZ_MAP")
    if (set(value["cells"]) != {f"{p}/{j}" for p in ("all", *map(str, range(6))) for j in range(4)}
            or value["quantiles"] != list(QUANTILES) or value["minimum_unique_jets"] != MIN_JETS
            or value["transform"] != "asinh_mm_values_log_mm_errors" or value["preserve_zero_values"] is not True
            or value["extrapolation"] != "constant_endpoint"
            or value["ties"] != "mean_target_at_identical_source_knots"):
        raise ValueError("Tracking map interface differs")
    for cell in value["cells"].values():
        if any(type(cell[k]) is not int or cell[k] < 0 for k in ("real_jets", "proxy_jets", "real_particles", "proxy_particles")):
            raise ValueError("Invalid map support counts")
        if any(cell[s+"jets"] > cell[s+"particles"] for s in ("real_", "proxy_")):
            raise ValueError("Map support exceeds observations")
        x, y = np.asarray(cell["x"]), np.asarray(cell["y"])
        if cell["status"] == "fitted":
            if (min(cell["real_jets"], cell["proxy_jets"]) < MIN_JETS or x.ndim != 1 or x.shape != y.shape
                    or len(x) < 2 or not np.isfinite([x, y]).all()
                    or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0)):
                raise ValueError("Invalid monotone map")
        elif cell["status"] not in ("insufficient_unique_jets", "degenerate_source") or len(x) or len(y):
            raise ValueError("Invalid identity fallback")
        elif ((cell["status"] == "insufficient_unique_jets") != (min(cell["real_jets"], cell["proxy_jets"]) < MIN_JETS)):
            raise ValueError("Map fallback disagrees with support")


def apply(base, mapping, candidate):
    strength = STRENGTHS[candidate]
    if not strength:
        return base, {}
    track, counters = base.tracking.copy(), {}
    for pid in range(6):
        for j, name in enumerate(TRACKING):
            mask = (base.category == pid) & base.valid[:, j]
            key = f"{pid}/{name}"
            count = int(mask.sum())
            if not count:
                continue
            cell = mapping["cells"][f"{pid}/{j}"]
            fallback = cell["status"] != "fitted"
            if fallback:
                cell = mapping["cells"][f"all/{j}"]
            zeros = mask & (track[:, j] == 0) if j < 2 else np.zeros(len(base), bool)
            active = mask & ~zeros
            row = dict(valid=count, pooled_fallback=count if fallback else 0,
                       identity_fallback=count if cell["status"] != "fitted" else 0,
                       zero_preserved=int(zeros.sum()), outside=0, mapped=0)
            if cell["status"] == "fitted" and active.any():
                x = transform(track[active, j], j)
                row["outside"] = int(((x < cell["x"][0]) | (x > cell["x"][-1])).sum())
                y = np.interp(x, cell["x"], cell["y"])
                z = (1-strength)*x + strength*y
                with np.errstate(over="raise", invalid="raise", under="raise"):
                    track[active, j] = np.sinh(z) if j < 2 else np.exp(z)
                row["mapped"] = int(active.sum())
            counters[key] = row
    output = Particles(base.p4, base.charge, base.category, track, base.valid, base.keys)
    check_invariants(base, output)
    return output, counters


def check_invariants(base, output):
    if base.keys != output.keys:
        raise ValueError("Tracking calibration changed particle identities/order")
    for field in ("p4", "charge", "category", "valid"):
        np.testing.assert_array_equal(getattr(base, field), getattr(output, field))
