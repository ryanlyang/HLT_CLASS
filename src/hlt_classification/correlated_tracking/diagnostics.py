"""Marginals and jet-weighted pair moments; no performance-based gating."""
import io

import numpy as np

from hlt_classification.literature_proxy import diagnostics as old
from .contracts import atomic_publish_bytes
from .kernel import SIDES, VARIANTS


def pair_moments(residual, projected, variance, correlated):
    """Average distinct ordered pairs without constructing an NxN covariance."""
    n = len(residual)
    if n < 2:
        return None
    product = (residual.sum()**2-np.dot(residual, residual))/(n*(n-1))
    cross = ((projected.sum(axis=0)**2).sum()-(projected**2).sum())/(n*(n-1)) if correlated else 0.
    return dict(pair_product=float(product), expected_pair_product=float(cross),
        pair_difference_sq=float(max(0., 2*np.mean(residual**2)-2*product)),
        expected_pair_difference_sq=float(max(0., 2*np.mean(variance)-2*cross)))


def edges(name):
    if "/response/" not in name:
        return old.edges(name)
    if name.endswith("pull"):
        return np.linspace(-8, 8, 161)
    positive = np.geomspace(1e-8, 1e3, 121)
    return np.r_[-positive[::-1], 0., positive]


class Collector(old.Collector):
    def response(self, side, original, output, g):
        mode, strength = VARIANTS[side]
        groups = [("all", np.ones(len(original), bool))]
        for axis, bins, coord in (("pt", (2., 10.), original.pt),
                                  ("density", (.25, .75), g.density),
                                  ("abs_eta", (.8, 1.6), np.abs(original.eta))):
            index = np.searchsorted(bins, coord, side="right")
            groups.extend((f"{axis}{i}", index == i) for i in range(3))
        groups.extend((f"pid{i}", original.category == i) for i in range(6))
        for k, field in enumerate(("d0", "dz")):
            delta = output.tracking[:, k]-original.tracking[:, k]
            variance = strength**2*g.variance[:, k]
            for group, mask in groups:
                ok = mask & g.eligible[:, k]
                prefix = f"{side}/response/{group}/{field}"
                self.add(prefix+"/residual", delta[ok])
                self.add(prefix+"/pull", delta[ok]/np.sqrt(variance[ok]))
            ok = g.eligible[:, k]
            moments = pair_moments(delta[ok], strength*g.projected_sd[ok, k],
                                   variance[ok], mode == "CORR")
            for key in ("pair_product", "expected_pair_product", "pair_difference_sq", "expected_pair_difference_sq"):
                self.add(f"{side}/response/pairs/{field}/{key}", [] if moments is None else [moments[key]])

    def finish(self):
        return {name: old.summarize(np.concatenate(values), name, bins=edges(name))
                for name, values in sorted(self.values.items())}


def exports(root, metrics):
    old.exports(root, metrics, sides=SIDES)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fields = [f"response/{group}/{field}/{stat}" for field in ("d0", "dz")
              for group, stats in (("all", ("residual", "pull")),
                  ("pairs", ("pair_product", "expected_pair_product", "pair_difference_sq", "expected_pair_difference_sq")))
              for stat in stats]
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"CreationDate": None, "ModDate": None}) as book:
        for start in range(0, len(fields), 6):
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            for ax, field in zip(axes.flat, fields[start:start+6]):
                for side in VARIANTS:
                    r = metrics[f"{side}/{field}"]
                    ax.stairs(np.asarray(r["histogram"])/max(1, r["count"]), r["bins"], label=side)
                if not field.endswith("pull"):
                    ax.set_xscale("symlog", linthresh=1e-5)
                ax.set_title(field, fontsize=8)
            axes.flat[0].legend(fontsize=6)
            fig.suptitle("Synthetic tracking response; paired jets, dependent particles; tails in CSV")
            fig.tight_layout()
            book.savefig(fig)
            plt.close(fig)
    atomic_publish_bytes(root / "mechanism.pdf", buf.getvalue())
