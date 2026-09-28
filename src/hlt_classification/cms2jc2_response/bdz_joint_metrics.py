"""PID-aware development score and diagnostics; no claims of qualification."""
import math
import numpy as np

from . import bdz_audit_metrics as audit, bdz_metrics as old
from .bdz_joint_maps import CANDIDATES

MIN_JETS = 50
GUARD = .02


def pooled_diagnostics(payload):
    total = audit.pooled(payload)
    combined = {"real": total["real"]}
    for n in CANDIDATES:
        combined[n] = {}
        for i in range(3):
            audit.merge(combined[n], total[f"{n}/proxy{i}"])
    out = audit.diagnostics({"pooled": combined})
    for n in CANDIDATES:
        for cell in out[n].values():
            for row in cell["variables"].values():
                row["contributing_jet_replica_exposures"] = row.pop("jets")
            for row in cell["pairs"].values():
                row["complete_jet_replica_exposures"] = row.pop("complete_jets")
                for part in row["by_error"]:
                    part["contributing_jet_replica_exposures"] = part.pop("jets")
    return out


def score(histogram, tracking, payload, name):
    historical = old.score(histogram, tracking)
    total = audit.pooled(payload)
    cells, excluded = {}, {}
    blocks = dict(tv=[], tails=[], joint=[])
    for pid in map(str, range(6)):
        real = total["real"][pid]
        proxies = [total[f"{name}/proxy{i}"][pid] for i in range(3)]
        for field in old.NAMES:
            r = real["variables"][field]
            ps = [p["variables"][field] for p in proxies]
            key = pid+"/"+field
            support = [r["jets"], *[p["jets"] for p in ps]]
            if min(support) < MIN_JETS:
                excluded[key] = support
                continue
            tv = float(np.mean([audit.tv(r["bins"], p["bins"]) for p in ps]))
            a = np.asarray(r["exceed"])/r["count"]
            tails = float(np.mean([np.mean(abs(a-np.asarray(p["exceed"])/p["count"])/
                (a+np.asarray(p["exceed"])/p["count"]+1/r["count"])) for p in ps]))
            cells[key+"/tv"], cells[key+"/tails"] = tv, tails
            blocks["tv"].append(tv); blocks["tails"].append(tails)
        for field in ("d0", "dz"):
            r, ps = real["pairs"][field], [p["pairs"][field] for p in proxies]
            key = pid+"/"+field+"/joint"
            support = [r["jets"], *[p["jets"] for p in ps]]
            if min(support) < MIN_JETS:
                excluded[key] = support
                continue
            cells[key] = float(np.mean([audit.tv(r["joint_bins"], p["joint_bins"]) for p in ps]))
            blocks["joint"].append(cells[key])
    # Average within PID first, then across PIDs: rare eligible PIDs are not
    # overwhelmed by the charged-hadron population.
    block_scores = {}
    for block in blocks:
        per_pid = []
        for pid in map(str, range(6)):
            vals = [v for k, v in cells.items() if k.startswith(pid+"/") and k.endswith("/"+block)]
            if vals:
                per_pid.append(float(np.mean(vals)))
        block_scores[block] = float(np.mean(per_pid)) if per_pid else 1.
    pid_score = float(np.mean(list(block_scores.values())))
    return dict(score=(historical["score"]+pid_score)/2, historical=historical,
        pid_score=pid_score, pid_blocks=block_scores, pid_cells=cells,
        excluded_support=excluded, unavailable_blocks=[k for k, v in blocks.items() if not v])


def choose(scores):
    if set(scores) != set(CANDIDATES):
        raise ValueError("Incomplete joint repair candidate registry")
    base, rejected = scores["B_DZ"], {}
    for n in CANDIDATES:
        row, reasons = scores[n], []
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("Invalid joint development score")
        if (row["pid_cells"].keys() != base["pid_cells"].keys()
                or row["excluded_support"] != base["excluded_support"]):
            raise ValueError("Tracking calibration changed diagnostic eligibility")
        for block in ("variable_tv", "variable_tail"):
            for field in old.NAMES:
                if row["historical"][block][field] > base["historical"][block][field]+GUARD:
                    reasons.append(block+"/"+field)
        if row["historical"]["blocks"]["joint"] > base["historical"]["blocks"]["joint"]+GUARD:
            reasons.append("historical_joint")
        reasons += ["pid/"+k for k, v in row["pid_cells"].items() if v > base["pid_cells"][k]+GUARD]
        rejected[n] = sorted(reasons)
    choice = min((n for n in CANDIDATES if not rejected[n]), key=lambda n: (scores[n]["score"], CANDIDATES.index(n)))
    return choice, rejected


def figures(payload):
    """Normalised joint maps; real/proxy sample sizes cannot change colour scale."""
    import io
    import matplotlib
    from matplotlib.figure import Figure
    total = audit.pooled(payload)
    for pid in ("0", "4"):
        sides = ["real", *[f"{n}/proxy0" for n in CANDIDATES]]
        fig = Figure(figsize=(18, 7))
        for j, field in enumerate(("d0", "dz")):
            for i, side in enumerate(sides):
                ax = fig.add_subplot(2, len(sides), j*len(sides)+i+1)
                grid = np.asarray(total[side][pid]["pairs"][field]["joint_bins"], float)
                grid = grid/grid.sum() if grid.sum() else grid
                shown = np.ma.masked_less_equal(grid, 0)
                from matplotlib.colors import LogNorm
                ax.imshow(shown, origin="lower", aspect="auto", norm=LogNorm(vmin=1e-5, vmax=1))
                ax.set(title=side+" / "+field, xlabel="log10 error bin (incl. flow)",
                       ylabel="log10 |value| bin (incl. zero/flow)")
        fig.suptitle(f"PID {pid}: probability per bin, fixed colour scale 1e-5..1; zero bins masked; replica 0")
        fig.tight_layout()
        buf = io.BytesIO()
        with matplotlib.rc_context({"svg.hashsalt": "cms2jc2_joint_v1"}):
            fig.savefig(buf, format="svg", metadata={"Date": None})
        yield f"joint_pid_{pid}.svg", buf.getvalue()
