"""Tail-aware development metrics with explicit complete-case denominators."""
from copy import deepcopy
import numpy as np

from .bounded_metrics import BLOCKS, histogram_block
from .bdz_maps import CANDIDATES

NAMES = ("d0", "dz", "d0err", "dzerr", "d0_significance", "dz_significance")
THRESHOLDS = ((.1, 1., 10., 100.),)*2 + ((.01, .1, 1., 10.),)*2 + ((1., 3., 10., 30., 100.),)*2


def tracking_row(p):
    values = [p.tracking[p.valid[:, j], j] for j in range(4)]
    for j in range(2):
        mask = p.valid[:, j] & p.valid[:, j+2]
        values.append(p.tracking[mask, j]/p.tracking[mask, j+2])
    tails = {n: dict(count=len(a), exceed=[int((abs(a) > t).sum()) for t in ts])
             for n, a, ts in zip(NAMES, values, THRESHOLDS)}
    full = p.tracking[p.valid.all(axis=1)]
    x = np.column_stack((np.arcsinh(full[:, :2]), np.log(full[:, 2:]),
                         np.arcsinh(full[:, :2]/full[:, 2:])))
    return dict(tails=tails, correlation=dict(count=len(x), sum=x.sum(axis=0).tolist(), cross=(x.T@x).tolist()))


def merge(a, b):
    if a is None:
        return deepcopy(b)
    for name in NAMES:
        a["tails"][name]["count"] += b["tails"][name]["count"]
        a["tails"][name]["exceed"] = (np.asarray(a["tails"][name]["exceed"])+b["tails"][name]["exceed"]).tolist()
    a["correlation"]["count"] += b["correlation"]["count"]
    for key in ("sum", "cross"):
        a["correlation"][key] = (np.asarray(a["correlation"][key])+b["correlation"][key]).tolist()
    return a


def correlation(row):
    if not row["count"]:
        return [[None]*6 for _ in range(6)]
    avg = np.asarray(row["sum"])/row["count"]
    cov = np.asarray(row["cross"])/row["count"]-np.outer(avg, avg)
    scale = np.sqrt(np.maximum(0., np.diag(cov)))
    return [[float(np.clip(cov[i, j]/(scale[i]*scale[j]), -1, 1)) if scale[i]*scale[j] > 1e-14 else None
             for j in range(6)] for i in range(6)]


def score(payload, rows):
    per_tv = {n: histogram_block(payload, ["particle_"+n])["score"] for n in NAMES}
    core = histogram_block(payload, BLOCKS["tracking"])
    real, per_tail, missing, joint = rows["real"], {}, {}, []
    rc = correlation(real["correlation"])
    for n in NAMES:
        r = real["tails"][n]
        scores = []
        for replica in range(3):
            p = rows[f"proxy{replica}"]["tails"][n]
            if not r["count"] or not p["count"]:
                scores.append(float(bool(r["count"]) != bool(p["count"])))
                missing[f"tail/{n}/{replica}"] = "both_missing" if not (r["count"] or p["count"]) else "one_missing"
            else:
                a, b = np.asarray(r["exceed"])/r["count"], np.asarray(p["exceed"])/p["count"]
                scores.append(float(np.mean(abs(a-b)/(a+b+1/r["count"]))))
        per_tail[n] = float(np.mean(scores))
    for replica in range(3):
        pc = correlation(rows[f"proxy{replica}"]["correlation"])
        for i in range(6):
            for j in range(i):
                a, b = rc[i][j], pc[i][j]
                if a is None or b is None:
                    joint.append(float((a is None) != (b is None)))
                    missing[f"joint/{replica}/{i}/{j}"] = "both_missing" if a is b is None else "one_missing"
                else:
                    joint.append(abs(a-b)/2)
    blocks = dict(core_tv=core["score"], tail_balanced=float(np.mean(list(per_tail.values()))), joint=float(np.mean(joint)))
    return dict(score=float(np.mean(list(blocks.values()))), blocks=blocks, variable_tv=per_tv,
                variable_tail=per_tail, missing={**core["missing"], **missing}, eligible_cohorts=core["eligible_cohorts"])


def choose(scores):
    if set(scores) != set(CANDIDATES):
        raise ValueError("Incomplete B_DZ tuning grid")
    base, rejected = scores["B_DZ"], {}
    for name in CANDIDATES:
        row = scores[name]
        reasons = [f"{key}/{n}" for key in ("variable_tv", "variable_tail") for n in NAMES
                   if row[key][n] > base[key][n]+.02]
        if row["blocks"]["joint"] > base["blocks"]["joint"]+.02:
            reasons.append("joint")
        rejected[name] = reasons
    winner = min((n for n in CANDIDATES if not rejected[n]), key=lambda n: (scores[n]["score"], CANDIDATES.index(n)))
    return winner, rejected


def tail_plot(tracks):
    """Exact exceedance-rate plots; zero rates are omitted, never floored."""
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    try:
        for ax, name, thresholds in zip(axes.flat, NAMES, THRESHOLDS):
            real = tracks["B_DZ"]["real"]["tails"][name]
            curves = {"CMS HLT": real}
            for candidate in CANDIDATES:
                rows = [tracks[candidate][f"proxy{i}"]["tails"][name] for i in range(3)]
                curves[candidate] = dict(count=sum(r["count"] for r in rows),
                    exceed=np.sum([r["exceed"] for r in rows], axis=0))
            for label, row in curves.items():
                if not row["count"]: continue
                rates = np.asarray(row["exceed"], float)/row["count"]
                zero = int((rates == 0).sum())
                ax.plot(thresholds, np.where(rates > 0, rates, np.nan), marker="o",
                        label=f"{label} (zero bins={zero})")
            ax.set(xscale="log", yscale="log", title=name,
                   xlabel="absolute threshold (mm; significance dimensionless)", ylabel="fraction exceeding threshold")
            ax.legend(fontsize=7)
        fig.suptitle("Development tracking tails; proxies pool three dependent replicas. Zero-rate points omitted.")
        fig.tight_layout()
        output = BytesIO()
        with matplotlib.rc_context({"svg.hashsalt": "cms2jc2_bdz_v1"}):
            fig.savefig(output, format="svg", metadata={"Date": None})
        return output.getvalue()
    finally:
        plt.close(fig)
