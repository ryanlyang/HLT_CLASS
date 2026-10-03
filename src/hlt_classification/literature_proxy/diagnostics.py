"""Fixed-bin descriptive diagnostics. No fitted targets or quality gates."""
from __future__ import annotations

from collections import defaultdict
import csv
import io

import numpy as np

from hlt_classification.cms2jc2_response.bridge import CATEGORIES, TRACKING, wrap_phi
from .contracts import atomic_publish_bytes
from .kernel import geometry


def edges(name):
    field = name.split("/")[-1]
    if "significance" in field:
        positive = np.geomspace(.01, 1e4, 101)
        return np.r_[-positive[::-1], 0., positive]
    if field in ("d0", "dz"):
        positive = np.geomspace(.0001, 1e3, 101)
        return np.r_[-positive[::-1], 0., positive]
    if field in ("d0err", "dzerr"):
        return np.r_[0., np.geomspace(.0001, 1e3, 151)]
    if field in ("pt", "energy", "scalar_pt", "mass"):
        return np.r_[0., np.geomspace(.001, 1e5, 151)]
    if field in ("eta", "axis_eta"):
        return np.linspace(-6, 6, 121)
    if field in ("width", "radius"):
        return np.linspace(0, 2, 101)
    if field == "charged_fraction" or field.startswith("valid_"):
        return np.linspace(0, 1, 101)
    if field == "pt_ratio":
        return np.linspace(0, 2, 101)
    if field == "count_delta":
        return np.arange(-2.5, 3.5)
    return np.arange(-.5, 1000.5)  # counts


def summarize(values, name):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite diagnostic: {name}")
    bins = edges(name)
    return dict(count=len(a), sum=float(a.sum()), sumsq=float(np.dot(a, a)),
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None,
                bins=bins.tolist(), histogram=np.histogram(a, bins=bins)[0].tolist(),
                underflow=int((a < bins[0]).sum()), overflow=int((a > bins[-1]).sum()),
                abs_tail_counts={str(t): int((np.abs(a) > t).sum()) for t in (.1, 1., 3., 10., 50., 100., 1000.)})


def moments(row):
    if not row["count"]:
        return None, None
    mean = row["sum"] / row["count"]
    return mean, float(np.sqrt(max(0., row["sumsq"] / row["count"] - mean**2)))


def quantile(row, q):
    """Bin midpoint approximation; explicitly indicate censored histogram tails."""
    if not row["count"]:
        return None
    target = q * row["count"]
    if target <= row["underflow"]:
        return f"<{row['bins'][0]}"
    cumulative = row["underflow"] + np.cumsum(row["histogram"])
    i = int(np.searchsorted(cumulative, target))
    if i >= len(row["histogram"]):
        return f">{row['bins'][-1]}"
    return (row["bins"][i] + row["bins"][i + 1]) / 2


def merge(reports):
    result = {}
    for report in reports:
        for name, row in report.items():
            if name not in result:
                import copy
                result[name] = copy.deepcopy(row)
                continue
            out = result[name]
            if out["bins"] != row["bins"]:
                raise ValueError("Histogram bins differ")
            for f in ("count", "sum", "sumsq", "underflow", "overflow"):
                out[f] += row[f]
            for f, fn in (("minimum", min), ("maximum", max)):
                vals = [v for v in (out[f], row[f]) if v is not None]
                out[f] = fn(vals) if vals else None
            out["histogram"] = (np.array(out["histogram"]) + row["histogram"]).tolist()
            for t, count in row["abs_tail_counts"].items():
                out["abs_tail_counts"][t] += count
    return result


def jet_values(p):
    total = p.p4.sum(axis=0)
    pt = float(np.hypot(*total[:2]))
    scalar = float(p.pt.sum())
    eta = float(np.arcsinh(total[2] / pt)) if pt else 0.
    phi = float(np.arctan2(total[1], total[0]))
    radius = np.hypot(p.eta - eta, wrap_phi(p.phi - phi))
    return dict(multiplicity=len(p), pt=pt, scalar_pt=scalar, energy=float(total[3]),
                mass=float(np.sqrt(max(0., total[3]**2 - np.dot(total[:3], total[:3])))),
                axis_eta=eta, width=float(np.dot(p.pt, radius) / scalar) if scalar else 0.,
                charged_fraction=float(p.pt[p.charge != 0].sum() / scalar) if scalar else 0.,
                **{f"count_{c}": int((p.category == i).sum()) for i, c in enumerate(CATEGORIES)}), radius


class Collector:
    """Bounded per-file arrays; histogram once per observable, not once per jet."""
    def __init__(self):
        self.values = defaultdict(list)

    def add(self, name, values):
        self.values[name].append(np.asarray(values).reshape(-1))

    def particle(self, prefix, p):
        for field in ("pt", "eta"):
            self.add(f"{prefix}/{field}", getattr(p, field))
        self.add(f"{prefix}/energy", p.p4[:, 3])
        for i, field in enumerate(TRACKING):
            self.add(f"{prefix}/{field}", p.tracking[p.valid[:, i], i])
            self.add(f"{prefix}/valid_{field}", p.valid[:, i])
        for i, field in enumerate(("d0", "dz")):
            ok = p.valid[:, i] & p.valid[:, i + 2]
            self.add(f"{prefix}/{field}_significance", p.tracking[ok, i] / p.tracking[ok, i + 2])

    def observe(self, side, p):
        jet, radius = jet_values(p)
        for field, value in jet.items():
            self.add(f"{side}/jet/{field}", [value])
        self.add(f"{side}/particle/radius", radius)
        self.particle(f"{side}/particle", p)
        for i, category in enumerate(CATEGORIES):
            self.particle(f"{side}/pid_{category}", p.take(np.flatnonzero(p.category == i)))

    def paired(self, side, original, response):
        self.add(f"{side}/paired/count_delta", [len(response.particles) - len(original)])
        a, _ = jet_values(original)
        b, _ = jet_values(response.particles)
        if a["pt"]:
            self.add(f"{side}/paired/pt_ratio", [b["pt"] / a["pt"]])
        # Condition on source particles; exclude merged descendants from 1:1 tracking summaries.
        source = {k: i for i, k in enumerate(original.keys)}
        out = [(i, source[anc[0]]) for i, anc in enumerate(response.ancestry) if len(anc) == 1]
        crowding, _ = geometry(original)
        for axis, bins, coord in (("pt", (2., 10.), original.pt),
                                  ("abs_eta", (.8, 1.6), np.abs(original.eta)),
                                  ("crowding", (.25, .75), crowding)):
            groups = np.searchsorted(bins, coord, side="right")
            for group in range(3):
                indices = [i for i, j in out if groups[j] == group]
                p = response.particles.take(indices)
                self.particle(f"{side}/source_{axis}_bin{group}", p)
                self.add(f"{side}/source_{axis}_bin{group}/neutral_count", [int((p.charge == 0).sum())])

    def finish(self):
        return {name: summarize(np.concatenate(values), name) for name, values in sorted(self.values.items())}


def exports(root, rows):
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(("observable", "count", "mean", "sd", "q50_bin_approx", "q90_bin_approx", "q99_bin_approx", "min", "max", "underflow", "overflow"))
    for name, row in sorted(rows.items()):
        w.writerow((name, row["count"], *moments(row), *(quantile(row, q) for q in (.5, .9, .99)),
                    row["minimum"], row["maximum"], row["underflow"], row["overflow"]))
    atomic_publish_bytes(root / "statistics.csv", buf.getvalue().encode())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pdf = io.BytesIO()
    with PdfPages(pdf, metadata={"CreationDate": None, "ModDate": None}) as book:
        fields = [n.removeprefix("OFFLINE/") for n in rows if n.startswith("OFFLINE/jet/") or n.startswith("OFFLINE/particle/")]
        for start in range(0, len(fields), 6):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax, field in zip(axes.flat, fields[start:start + 6]):
                occupied = []
                for side in ("OFFLINE", "MILD", "NOMINAL", "STRONG"):
                    row = rows[f"{side}/{field}"]
                    hist = np.array(row["histogram"]) / max(1, row["count"])
                    ax.stairs(hist, row["bins"], label=side)
                    nonzero = np.flatnonzero(hist)
                    if len(nonzero):
                        occupied.extend((row["bins"][nonzero[0]], row["bins"][nonzero[-1] + 1]))
                ax.set_title(field, fontsize=9)
                ax.set_ylabel("probability per fixed bin")
                if field.split("/")[-1] in ("d0", "dz", "d0_significance", "dz_significance"):
                    ax.set_xscale("symlog", linthresh=.01)
                elif field.split("/")[-1] in ("pt", "energy", "scalar_pt", "mass", "d0err", "dzerr"):
                    ax.set_xscale("symlog", linthresh=.001)
                elif occupied:
                    low, high = min(occupied), max(occupied)
                    padding = max((high-low)*.05, .005)
                    ax.set_xlim(low-padding, high+padding)
            for ax in list(axes.flat)[len(fields[start:start + 6]):]:
                ax.set_visible(False)
            axes.flat[0].legend(fontsize=7)
            fig.suptitle("Controlled proxy, training only; see CSV for overflow and masks")
            fig.tight_layout()
            book.savefig(fig)
            plt.close(fig)
    atomic_publish_bytes(root / "overlays.pdf", pdf.getvalue())
