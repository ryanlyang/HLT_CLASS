"""All-jet development plots without expensive proxy rematching or quality cuts.

These collection diagnostics deliberately do not masquerade as the production
six-block association/transition score. Arrays are transient; histograms persist.
"""
from __future__ import annotations

from io import BytesIO
import math
import numpy as np

from .association import _axis
from .bridge import wrap_phi
from .contracts import artifact, validate
from .evaluation import ObservationSample
from .features import features
from .metrics import jet_summary, JET_JOINT

PID = ("charged_hadron", "neutral_hadron", "photon", "electron", "muon", "unknown")
JET_NAMES = ("multiplicity", *[f"count_{i}" for i in range(6)], "scalar_pt", "vector_pt",
             "energy", "mass", "eta", "charged_fraction", "leading_fraction", "subleading_fraction",
             "width", "e2_1", "e2_2", *[f"radial_{i}" for i in range(7)], "pt_ratio", "axis_dr")
PARTICLE_NAMES = ("pt", "energy", "eta", "phi", "radius", "d0", "dz", "d0err", "dzerr",
                  "d0_significance", "dz_significance", "valid_d0", "valid_dz", "valid_d0err", "valid_dzerr")
NAMES = tuple("jet_"+n for n in JET_NAMES) + tuple("particle_"+n for n in PARTICLE_NAMES) + tuple(
    f"{pid}_{field}" for pid in PID for field in ("pt", "d0", "dz"))
CONDITIONS = ("all", "count_lt50", "count_50_99", "count_ge100", "pt_lt500", "pt_ge500",
              "crowding_lt3", "crowding_ge3", "crowding_empty", "d0_lt0p1", "d0_ge0p1", "d0_missing")


def transform_name(name):
    if name.endswith(("_d0", "_dz", "_d0_significance", "_dz_significance")) and "valid_" not in name:
        return "asinh (tracking values/errors in mm; significance dimensionless)"
    if name.endswith(("_pt", "_energy", "_mass", "_d0err", "_dzerr")) and "valid_" not in name:
        return "log1p (GeV for momentum/energy/mass; mm for errors)"
    return "identity"


def transformed(name, values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite development diagnostic")
    transform = transform_name(name)
    if transform.startswith("asinh"):
        return np.arcsinh(a)
    if transform.startswith("log1p"):
        if np.any(a < 0):
            raise ValueError("Negative positive-scale diagnostic")
        return np.log1p(a)
    return a


def observables(offline, output):
    summary = jet_summary(output)
    summary["scalar_pt"] = float(output.pt.sum())
    summary["energy"] = float(output.p4[:, 3].sum())
    summary["mass"] = summary.get("mass", 0.)
    axis = _axis(output.p4.sum(axis=0))
    origin = _axis(offline.p4.sum(axis=0))
    summary["vector_pt"] = float(np.hypot(*output.p4[:, :2].sum(axis=0)))
    if axis is not None:
        summary["eta"] = axis[1]
        if origin is not None:
            summary["pt_ratio"] = axis[0]/origin[0]
            summary["axis_dr"] = float(np.hypot(axis[1]-origin[1], wrap_phi(axis[2]-origin[2])))
    values = {"jet_"+k: [float(v)] for k, v in summary.items() if k in JET_NAMES}
    values.update(particle_pt=output.pt.tolist(), particle_energy=output.p4[:, 3].tolist(),
                  particle_eta=output.eta.tolist(), particle_phi=output.phi.tolist())
    if axis is not None:
        values["particle_radius"] = np.hypot(output.eta-axis[1], wrap_phi(output.phi-axis[2])).tolist()
    for j, name in enumerate(("d0", "dz", "d0err", "dzerr")):
        values["particle_"+name] = output.tracking[output.valid[:, j], j].tolist()
        values["particle_valid_"+name] = output.valid[:, j].astype(float).tolist()
    for j, name in enumerate(("d0_significance", "dz_significance")):
        mask = output.valid[:, j] & output.valid[:, j+2]
        values["particle_"+name] = (output.tracking[mask, j]/output.tracking[mask, j+2]).tolist()
    for cat, pid in enumerate(PID):
        mask = output.category == cat
        values[pid+"_pt"] = output.pt[mask].tolist()
        for j, name in enumerate(("d0", "dz")):
            values[pid+"_"+name] = output.tracking[mask & output.valid[:, j], j].tolist()
    return values


def conditions(offline):
    result = ["all", "count_lt50" if len(offline) < 50 else "count_50_99" if len(offline) < 100 else "count_ge100",
              "pt_lt500" if offline.pt.sum() < 500 else "pt_ge500"]
    x = features(offline)
    result.append("crowding_empty" if not len(x) else "crowding_lt3" if np.median(x[:, 17]) < 3 else "crowding_ge3")
    d0 = abs(offline.tracking[offline.valid[:, 0], 0])
    result.append("d0_missing" if not len(d0) else "d0_lt0p1" if max(d0) < .1 else "d0_ge0p1")
    return result


def fit_ranges(pairs, sample_hash):
    sample = ObservationSample(cap=4096)
    jets = 0
    for pair in pairs:
        for name, vals in observables(pair.offline, pair.hlt).items():
            sample.add(name, pair.identity, transformed(name, vals).tolist())
        jets += 1
        if jets % 1000 == 0:
            print(f"CMS2JC2-DEV phase=diagnostic_ranges jets={jets}", flush=True)
    definitions = {}
    for name in NAMES:
        values = sample.values(name)
        lo, hi = (min(values), max(values)) if values else (0., 1.)
        if lo == hi:
            lo -= .5; hi += .5
        definitions[name] = dict(edges=np.linspace(lo, hi, 129).tolist(), transform=transform_name(name),
                                 fit_observations=sample.counts.get(name, 0), sampled_observations=len(values))
    return artifact("DEV_RANGES", parents={"samples": sample_hash}, definitions=definitions,
                    fitting_jets=jets, conditions=list(CONDITIONS), replicas=[0, 1, 2],
                    selection_accessed=False, confirmation_accessed=False,
                    underflow_overflow="explicit counts; plots disclose both; no sample rejection")


class Histograms:
    def __init__(self, ranges):
        validate(ranges, "DEV_RANGES")
        self.ranges = ranges
        self.cells = {}
        self.correlations = {}

    def add(self, side, cohorts, offline, output):
        values = observables(offline, output)
        for cohort in cohorts:
            cell = self.cells.setdefault(side+"/"+cohort, dict(jets=0, empty_jets=0, variables={}))
            cell["jets"] += 1
            cell["empty_jets"] += int(len(output) == 0)
            for name, raw in values.items():
                if not raw:
                    continue
                a = np.asarray(raw, np.float64)
                edges = self.ranges["definitions"][name]["edges"]
                v = transformed(name, a)
                # searchsorted includes explicit under/overflow; equality to last edge is overflow.
                counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges)+1)
                row = cell["variables"].setdefault(name, dict(count=0, covered_jets=0, sum=0., sumsq=0.,
                    minimum=float(a.min()), maximum=float(a.max()), bins=np.zeros(len(edges)+1, np.int64)))
                row["count"] += len(a); row["covered_jets"] += 1
                row["sum"] += float(a.sum()); row["sumsq"] += float(a@a)
                row["minimum"] = min(row["minimum"], float(a.min()))
                row["maximum"] = max(row["maximum"], float(a.max()))
                row["bins"] += counts
        jet = jet_summary(output)
        if all(n in jet for n in JET_JOINT):
            x = np.asarray([jet[n] for n in JET_JOINT])
            row = self.correlations.setdefault(side, dict(count=0, sum=np.zeros(len(x)), cross=np.zeros((len(x), len(x)))))
            row["count"] += 1; row["sum"] += x; row["cross"] += np.outer(x, x)

    def payload(self):
        cells = {key: {**cell, "variables": {name: {**r, "bins": r["bins"].tolist()}
                 for name, r in cell["variables"].items()}} for key, cell in self.cells.items()}
        corr = {key: dict(count=r["count"], sum=r["sum"].tolist(), cross=r["cross"].tolist())
                for key, r in self.correlations.items()}
        return dict(cells=cells, correlations=corr)


def merge_payloads(payloads):
    import copy
    result = dict(cells={}, correlations={})
    for payload in payloads:
        for key, cell in payload["cells"].items():
            if key not in result["cells"]:
                result["cells"][key] = copy.deepcopy(cell); continue
            target = result["cells"][key]
            target["jets"] += cell["jets"]; target["empty_jets"] += cell["empty_jets"]
            for name, row in cell["variables"].items():
                if name not in target["variables"]:
                    target["variables"][name] = copy.deepcopy(row); continue
                old = target["variables"][name]
                for n in ("count", "covered_jets", "sum", "sumsq"):
                    old[n] += row[n]
                old["minimum"] = min(old["minimum"], row["minimum"])
                old["maximum"] = max(old["maximum"], row["maximum"])
                old["bins"] = (np.asarray(old["bins"])+row["bins"]).tolist()
        for key, row in payload["correlations"].items():
            if key not in result["correlations"]:
                result["correlations"][key] = copy.deepcopy(row); continue
            target = result["correlations"][key]
            target["count"] += row["count"]
            for n in ("sum", "cross"):
                target[n] = (np.asarray(target[n])+row[n]).tolist()
    return result


def comparisons(payload):
    result = {}
    for cohort in CONDITIONS:
        real = payload["cells"].get("real/"+cohort)
        if real is None:
            continue
        for replica in range(3):
            proxy = payload["cells"].get(f"proxy{replica}/"+cohort)
            if proxy is None or proxy["jets"] != real["jets"]:
                raise ValueError("Development proxy population differs")
            rows = {}
            for name in NAMES:
                a, b = real["variables"].get(name), proxy["variables"].get(name)
                if a is None or b is None:
                    rows[name] = dict(status="unavailable", real_observations=0 if a is None else a["count"],
                                      proxy_observations=0 if b is None else b["count"])
                    continue
                av, bv = a["sum"]/a["count"], b["sum"]/b["count"]
                rows[name] = dict(status="available", real_mean=av, proxy_mean=bv, mean_shift=bv-av,
                    relative_mean_shift=(bv-av)/abs(av) if av else None,
                    histogram_tv=float(.5*abs(np.asarray(a["bins"])/a["count"]-np.asarray(b["bins"])/b["count"]).sum()),
                    real_observations=a["count"], proxy_observations=b["count"],
                    real_covered_jets=a["covered_jets"], proxy_covered_jets=b["covered_jets"],
                    real_under_over=[a["bins"][0], a["bins"][-1]], proxy_under_over=[b["bins"][0], b["bins"][-1]])
            result[f"proxy{replica}/"+cohort] = rows
    return result


def correlation_report(payload):
    result = {}
    for side, row in payload["correlations"].items():
        n = row["count"]
        mean = np.asarray(row["sum"])/n
        cov = np.asarray(row["cross"])/n - np.outer(mean, mean)
        scale = np.sqrt(np.maximum(np.diag(cov), 0))
        corr = [[float(np.clip(cov[i, j]/(scale[i]*scale[j]), -1, 1)) if scale[i]*scale[j] > 0 else None
                 for j in range(len(scale))] for i in range(len(scale))]
        result[side] = dict(jets=n, pearson_correlation=corr, coordinates=list(JET_JOINT),
                            undefined_correlations=None, independent_replica_jets=False)
    return result


def plot_pages(payload, ranges, candidate, *, cohort="all"):
    """Yield bounded SVG pages; publication/storage accounting belongs to the caller."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    names = [n for n in NAMES if any(n in payload["cells"].get(s+"/"+cohort, {}).get("variables", {})
                                     for s in ("offline", "real", "proxy0"))]
    colors = dict(offline="#999999", real="#111111", proxy0="#0072B2", proxy1="#D55E00", proxy2="#009E73")
    for start in range(0, len(names), 9):
        fig, axes = plt.subplots(3, 3, figsize=(14, 11))
        try:
            for ax, name in zip(axes.flat, names[start:start+9]):
                edges = np.asarray(ranges["definitions"][name]["edges"])
                for side, color in colors.items():
                    row = payload["cells"].get(side+"/"+cohort, {}).get("variables", {}).get(name)
                    if row is None:
                        continue
                    bins = np.asarray(row["bins"])/row["count"]
                    ax.stairs(bins[1:-1], edges, color=color,
                              label=f"{side} (outside={bins[0]+bins[-1]:.1%})", linewidth=1.2)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(transform_name(name), fontsize=6)
                ax.set_ylabel("fraction of all observations/bin", fontsize=7)
                ax.legend(fontsize=6)
            for ax in list(axes.flat)[len(names[start:start+9]):]:
                ax.set_visible(False)
            fig.suptitle(f"{candidate}: {cohort} — development only; replicas are not independent data")
            fig.tight_layout(rect=(0, 0, 1, .96))
            output = BytesIO()
            with matplotlib.rc_context({"svg.hashsalt": "cms2jc2_dev_v1"}):
                fig.savefig(output, format="svg", metadata={"Date": None})
            yield f"{cohort}_{start//9:02d}.svg", output.getvalue()
        finally:
            plt.close(fig)
