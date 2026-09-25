"""Frozen development scores, explicit missingness and paired file uncertainty."""
import numpy as np

from .dev_diagnostics import CONDITIONS, PID, correlation_report, merge_payloads

BLOCKS = {
    "counts_state": ["jet_multiplicity", *[f"jet_count_{i}" for i in range(6)],
                     *["particle_valid_"+v for v in ("d0", "dz", "d0err", "dzerr")]],
    "particle_kinematics": ["particle_"+v for v in ("pt", "energy", "eta", "radius")]
                           + [p+"_pt" for p in PID],
    "tracking": ["particle_"+v for v in ("d0", "dz", "d0err", "dzerr", "d0_significance", "dz_significance")],
    "jet_shape": ["jet_"+v for v in ("scalar_pt", "vector_pt", "energy", "mass", "eta",
        "charged_fraction", "leading_fraction", "subleading_fraction", "width", "e2_1", "e2_2",
        *[f"radial_{i}" for i in range(7)], "pt_ratio", "axis_dr")],
}
BIAS_NAMES = ("jet_multiplicity", "jet_scalar_pt", "jet_mass")
DZ_NAMES = ("particle_dz", "particle_dz_significance")


def tv(a, b):
    if a is None and b is None:
        return 0., "absent_both"
    if a is None or b is None:
        return 1., "absent_one"
    return float(.5*np.abs(np.asarray(a["bins"])/a["count"]-np.asarray(b["bins"])/b["count"]).sum()), "available"


def histogram_block(payload, names):
    cells = payload["cells"]
    cohorts = [cohort for cohort in CONDITIONS if "real/"+cohort in cells
               and (cohort == "all" or cells["real/"+cohort]["jets"] >= 1000)]
    values, missing = [], {}
    for cohort in cohorts:
        real = cells["real/"+cohort]
        for replica in range(3):
            other = cells[f"proxy{replica}/"+cohort]
            if real["jets"] != other["jets"]:
                raise ValueError("Bounded score populations differ")
            for name in names:
                value, status = tv(real["variables"].get(name), other["variables"].get(name))
                values.append(value)
                if status != "available":
                    missing[f"{cohort}/{replica}/{name}"] = status
    if not values or "all" not in cohorts:
        raise ValueError("Empty bounded score population")
    return dict(score=float(np.mean(values)), missing=missing, eligible_cohorts=cohorts)


def score(payload):
    blocks = {name: histogram_block(payload, names) for name, names in BLOCKS.items()}
    correlations = correlation_report(payload)
    real = correlations.get("real")
    if real is None:
        raise ValueError("No real correlation population")
    r = real["pearson_correlation"]
    values, missing = [], {}
    for replica in range(3):
        candidate = correlations.get(f"proxy{replica}")
        p = candidate["pearson_correlation"] if candidate else [[None]*len(r) for _ in r]
        for i in range(len(r)):
            for j in range(i):
                if r[i][j] is None or p[i][j] is None:
                    both = r[i][j] is None and p[i][j] is None
                    values.append(0. if both else 1.)
                    missing[f"{replica}/{i}/{j}"] = "absent_both" if both else "absent_one"
                else:
                    values.append(abs(r[i][j]-p[i][j])/2)
    blocks["joint_jet"] = dict(score=float(np.mean(values)), missing=missing, eligible_cohorts=["all"])
    biases = {}
    for name in BIAS_NAMES:
        real = payload["cells"]["real/all"]["variables"][name]
        rmean = real["sum"]/real["count"]
        means = [payload["cells"][f"proxy{i}/all"]["variables"][name] for i in range(3)]
        # Absolute bias per replica, then average (opposite biases must not cancel).
        biases[name] = float(np.mean([abs(v["sum"]/v["count"]-rmean)/max(abs(rmean), 1e-12) for v in means]))
    result = dict(score=float(np.mean([v["score"] for v in blocks.values()])), blocks=blocks, mean_bias=biases)
    if not np.isfinite([result["score"], *biases.values(), *[v["score"] for v in blocks.values()]]).all():
        raise ValueError("Nonfinite bounded decision statistic")
    return result


def guards(base, candidate):
    return (["block:"+key for key in base["blocks"]
             if candidate["blocks"][key]["score"] > base["blocks"][key]["score"]+.02]
            + ["bias:"+key for key in BIAS_NAMES if candidate["mean_bias"][key] > base["mean_bias"][key]+.02])


def choose(scores):
    from .bounded_models import CANDIDATES
    if set(scores) != set(CANDIDATES):
        raise ValueError("Selection needs all three registered candidates")
    rejected = {name: guards(scores["B"], scores[name]) for name in CANDIDATES}
    winner = min((n for n in CANDIDATES if not rejected[n]), key=lambda n: (scores[n]["score"], CANDIDATES.index(n)))
    return winner, rejected


def uncertainty(by_file, candidate):
    """Source files, not particles or replicas, are the independent units."""
    files = sorted(by_file)
    rng = np.random.default_rng(20260925)
    improvements, cache = [], {}
    for _ in range(200):
        draw = rng.choice(files, size=len(files), replace=True)
        # Repeated draws of the same file-count vector have identical evidence.
        # Canonicalize the addition order and cache, especially with few files.
        key = tuple(int(np.count_nonzero(draw == f)) for f in files)
        if key not in cache:
            if candidate == "B":
                cache[key] = 0.
            else:
                canonical = [f for f, n in zip(files, key) for _ in range(n)]
                values = {name: score(merge_payloads([by_file[f][name] for f in canonical]))["score"]
                          for name in ("B", candidate)}
                cache[key] = values["B"]-values[candidate]
        improvements.append(cache[key])
    return dict(source_files=len(files), draws=200, seed=20260925,
        sufficient_groups=len(files) >= 4, improvement_95pct=np.quantile(improvements, [.025, .975]).tolist(),
        unit="source_file; replicas kept together", model_selection_bias_corrected=False)


def confirmation_status(base, selected, interval, candidate):
    if candidate == "B":
        return "control_retained"
    gain = base["score"]-selected["score"]
    if (gain >= .005 and gain >= .1*base["score"] and interval["sufficient_groups"]
            and interval["improvement_95pct"][0] > 0 and not guards(base, selected)):
        return "bounded_confirmation_improvement"
    return "no_clear_improvement"
