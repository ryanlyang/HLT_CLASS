"""Streaming stored-unit diagnostics; never infer units from empirical widths."""
import numpy as np

from .particles import FIELDS, ParticleReader


class Moments:
    def __init__(self):
        self.n = self.nonfinite = self.zero = self.negative = 0
        self.mean = self.m2 = 0.
        self.minimum, self.maximum = None, None
        self.edges = np.logspace(-8, 5, 27)
        self.histogram = np.zeros(len(self.edges) + 1, np.int64)

    def add(self, values):
        values = np.asarray(values, np.float64).ravel()
        self.nonfinite += int((~np.isfinite(values)).sum())
        v = values[np.isfinite(values)]
        if not len(v):
            return
        self.zero += int((v == 0).sum())
        self.negative += int((v < 0).sum())
        mean, count = float(v.mean()), len(v)
        delta, total = mean - self.mean, self.n + count
        self.m2 += float(((v - mean)**2).sum()) + delta**2 * self.n * count / total
        self.mean += delta * count / total
        self.n = total
        self.minimum = min(float(v.min()), self.minimum if self.minimum is not None else float(v.min()))
        self.maximum = max(float(v.max()), self.maximum if self.maximum is not None else float(v.max()))
        self.histogram += np.bincount(np.searchsorted(self.edges, np.abs(v), side="right"),
                                      minlength=len(self.histogram))

    def report(self):
        return dict(finite=self.n, nonfinite=self.nonfinite, zero=self.zero, negative=self.negative,
                    mean=self.mean if self.n else None, sd=(self.m2/self.n)**.5 if self.n else None,
                    minimum=self.minimum, maximum=self.maximum,
                    abs_bin_edges=self.edges.tolist(), abs_bin_counts=self.histogram.tolist())


def scan(container, inventory, splits):
    statistics, coverage, issues = {}, {}, {}
    for role in ("train", "validation"):
        meters = {}
        flags = dict(nonfinite_required=0, nonbinary_pid=0, multiple_pid=0, missing_pid=0,
                     invalid_charge=0, charge_pid_conflict=0, nonpositive_pt=0,
                     nonpositive_energy=0, spacelike=0, negative_error=0)
        side_flags = {side: flags.copy() for side in ("hlt", "offline")}
        jets = 0
        def add(key, values):
            meters.setdefault(key, Moments()).add(values)
        for jet in ParticleReader(container, inventory, splits, role=role, include_offline=True):
            jets += 1
            for side in ("hlt", "offline"):
                a = getattr(jet, side)
                f = side_flags[side]
                add(side + "/multiplicity", [len(a)])
                pid, charge = a[:, 5:10], a[:, 4]
                ok = np.isin(pid, (0, 1)).all(axis=1)
                sums = pid.sum(axis=1)
                category = np.where(ok & (sums == 1), pid.argmax(axis=1), 5)
                f["nonfinite_required"] += int((~np.isfinite(a)).sum())
                f["nonbinary_pid"] += int((~ok).sum())
                f["multiple_pid"] += int((ok & (sums > 1)).sum())
                f["missing_pid"] += int((ok & (sums == 0)).sum())
                f["invalid_charge"] += int((~np.isin(charge, (-1, 0, 1))).sum())
                f["charge_pid_conflict"] += int(((category != 5) & ((charge != 0) != np.isin(category, (0, 3, 4)))).sum())
                pt = np.hypot(a[:, 0], a[:, 1])
                p2 = (a[:, :3]**2).sum(axis=1)
                f["nonpositive_pt"] += int((pt <= 0).sum())
                f["nonpositive_energy"] += int((a[:, 3] <= 0).sum())
                f["spacelike"] += int((a[:, 3]**2 < p2 - 2e-6*np.maximum(p2, 1)).sum())
                f["negative_error"] += int((a[:, 12:14] < 0).sum())
                for group, mask in [("all", np.ones(len(a), bool)), ("charged", charge != 0),
                                    ("neutral", charge == 0)] + [(str(i), category == i) for i in range(6)]:
                    for col in range(10, 14):
                        add(f"{side}/{group}/{FIELDS[col]}", a[mask, col])
                for i in range(6):
                    add(f"{side}/count_pid_{i}", [(category == i).sum()])
                add(side + "/particle_pt", pt)
                add(side + "/particle_energy", a[:, 3])
            if jets % 10000 == 0:
                print(f"LUKA audit role={role} jets={jets}", flush=True)
        statistics[role] = {k: v.report() for k, v in sorted(meters.items())}
        coverage[role], issues[role] = jets, side_flags
    return dict(counts=coverage, statistics=statistics, issues=issues,
                units="stored_unconfirmed_no_conversion", moments="particle_weighted_except_jet_counts_SD_is_width",
                zero_error_meaning="unresolved_not_inferred", scientific_admission=False,
                final_test_accessed=False)
