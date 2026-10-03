"""Paired response widths distinguish noise from changes in population means."""
import io

import numpy as np

from hlt_classification.cms2jc2_response.bridge import wrap_phi
from hlt_classification.literature_proxy import diagnostics as old
from .contracts import atomic_publish_bytes

SIDES = ("OFFLINE", "COUNT38_V2", "NOISE_V3")


def edges(name):
    field = name.split("/")[-1]
    if field == "count_delta":
        return np.arange(-1000.5, 1.5)
    if field in ("delta_eta", "delta_phi"):
        return np.linspace(-.3, .3, 301)
    if field == "delta_r":
        return np.linspace(0., .3, 151)
    if field in ("d0_delta", "dz_delta"):
        return old.edges(field.removesuffix("_delta"))
    if field.endswith("err_ratio"):
        return np.r_[0., np.geomspace(.1, 100., 151)]
    return old.edges(name)


class Collector(old.Collector):
    def paired(self, side, original, response):
        super().paired(side, original, response)
        p = response.particles
        a, b = original.p4.sum(axis=0), p.p4.sum(axis=0)
        apt, bpt = np.hypot(*a[:2]), np.hypot(*b[:2])
        if apt > 0 and bpt > 0:
            de = np.arcsinh(b[2]/bpt)-np.arcsinh(a[2]/apt)
            dp = wrap_phi(np.arctan2(b[1], b[0])-np.arctan2(a[1], a[0]))
            axis = dict(delta_eta=[de], delta_phi=[dp], delta_r=[np.hypot(de, dp)])
        else:
            axis = {k: [] for k in ("delta_eta", "delta_phi", "delta_r")}
        for field, values in axis.items():
            self.add(f"{side}/paired/{field}", values)
        source = {key: j for j, key in enumerate(original.keys)}
        pairs = [(i, source[anc[0]]) for i, anc in enumerate(response.ancestry) if len(anc) == 1]
        out = p.take([i for i, _ in pairs])
        inp = original.take([j for _, j in pairs])
        prefix = f"{side}/single_parent"
        de, dp = out.eta-inp.eta, wrap_phi(out.phi-inp.phi)
        for field, values in dict(pt_ratio=out.pt/inp.pt, delta_eta=de, delta_phi=dp,
                                  delta_r=np.hypot(de, dp)).items():
            self.add(f"{prefix}/{field}", values)
        for i, field in enumerate(("d0", "dz")):
            valid = out.valid[:, i] & inp.valid[:, i]
            self.add(f"{prefix}/{field}_delta", out.tracking[valid, i]-inp.tracking[valid, i])
            ok = out.valid[:, i+2] & inp.valid[:, i+2]
            self.add(f"{prefix}/{field}err_ratio", out.tracking[ok, i+2]/inp.tracking[ok, i+2])

    def finish(self):
        return {name: old.summarize(np.concatenate(values), name, bins=edges(name))
                for name, values in sorted(self.values.items())}


def exports(root, rows):
    old.exports(root, rows, sides=SIDES)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fields = [n.removeprefix("COUNT38_V2/") for n in rows
              if n.startswith(("COUNT38_V2/paired/", "COUNT38_V2/single_parent/"))]
    pdf = io.BytesIO()
    with PdfPages(pdf, metadata={"CreationDate": None, "ModDate": None}) as book:
        for start in range(0, len(fields), 6):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax, field in zip(axes.flat, fields[start:start+6]):
                occupied = []
                for side in SIDES[1:]:
                    row = rows[f"{side}/{field}"]
                    hist = np.array(row["histogram"])/max(1, row["count"])
                    ax.stairs(hist, row["bins"], label=f"{side} (n={row['count']})")
                    nz = np.flatnonzero(hist)
                    if len(nz):
                        occupied.extend((row["bins"][nz[0]], row["bins"][nz[-1]+1]))
                if field.endswith(("d0_delta", "dz_delta")):
                    ax.set_xscale("symlog", linthresh=.01)
                elif occupied:
                    low, high = min(occupied), max(occupied)
                    pad = max((high-low)*.05, .001)
                    ax.set_xlim(low-pad, high+pad)
                ax.set_title(field, fontsize=9)
                ax.legend(fontsize=6)
                ax.set_ylabel("probability per bin")
            for ax in list(axes.flat)[len(fields[start:start+6]):]:
                ax.set_visible(False)
            fig.suptitle("Paired response; single-parent survivors only for particle residuals; tails in CSV")
            fig.tight_layout()
            book.savefig(fig)
            plt.close(fig)
    atomic_publish_bytes(root / "paired_response.pdf", pdf.getvalue())
