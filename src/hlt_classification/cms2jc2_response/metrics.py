"""Frozen six-block observables and bounded, mergeable distribution summaries.

Histogram Wasserstein approximations expose a deterministic error bound based
on actual within-bin extrema. Undefined quantities have their own state mass.
Nothing in this module selects or trains on classification/KD performance.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.stats import wasserstein_distance

from .association import associate, _axis, distances, groups
from .bridge import Particles, wrap_phi
from .contracts import artifact, validate
from .features import features, group_features
from .conditioning import particle_cells, jet_cells, eligible_cells

ANNULI = (0., .02, .05, .1, .2, .4, .8, float("inf"))
MASKS = (0, 1, 2, 3, 5, 7, 10, 11, 15)
JET_JOINT = ("multiplicity", "log_sum_pt", "mass", "charged_fraction", "width", "e2_1", "e2_2")
TRACK = ("d0", "dz", "log_d0err", "log_dzerr", "d0_significance", "dz_significance")
BLOCKS = {
    "topology": ["multiplicity", *[f"count_{i}" for i in range(6)], "loss_rate", "merge_rate", "split_rate", "additional_rate"],
    "particles": ["log_pt_ratio", "log_energy_ratio", "delta_eta", "delta_phi", "leading_fraction", "subleading_fraction"],
    "identity": ["category_charge_transition", "d0_availability_transition", "dz_availability_transition",
                 "d0err_availability_transition", "dzerr_availability_transition"],
    "tracking": [*TRACK, *[f"track_corr_{a}_{b}" for a, b in combinations(TRACK, 2)]],
    "jets": ["jet_log_pt_ratio", "mass", "jet_log_mass_ratio", "jet_axis_dr", "charged_fraction",
             *[f"radial_{i}" for i in range(7)], "width", "e2_1", "e2_2"],
    "joint": ["jet_joint", *[f"particle_joint_{mask}" for mask in MASKS],
              *[f"jet_corr_{a}_{b}" for a, b in combinations(JET_JOINT, 2)]],
}
CATEGORICAL = set(BLOCKS["identity"])


def _append(result, name, value):
    if value is not None:
        a = np.asarray(value)
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite physical metric: {name}")
        result.setdefault(name, []).append(value)


def jet_summary(p: Particles) -> dict:
    result = dict(multiplicity=float(len(p)))
    result.update({f"count_{i}": float(np.count_nonzero(p.category == i)) for i in range(6)})
    if not len(p):
        return result
    pt_sum = p.pt.sum()
    z = p.pt/pt_sum
    vector = p.p4.sum(axis=0)
    mass = float(np.sqrt(max(0., vector[3]**2-(vector[:3]**2).sum())))
    result.update(log_sum_pt=float(np.log(pt_sum)), mass=mass,
                  charged_fraction=float(z[p.charge != 0].sum()),
                  leading_fraction=float(np.max(z)))
    if len(p) > 1:
        result["subleading_fraction"] = float(np.sort(z)[-2])
    dr = distances(p)
    product = z[:, None]*z[None, :]
    result.update(e2_1=float(np.sum(np.triu(product*dr, 1))),
                  e2_2=float(np.sum(np.triu(product*dr**2, 1))))
    axis = _axis(vector)
    if axis is not None:
        radius = np.hypot(p.eta-axis[1], wrap_phi(p.phi-axis[2]))
        result["width"] = float(z@radius)
        result.update({f"radial_{i}": float(z[(radius >= lo) & (radius < hi)].sum())
                       for i, (lo, hi) in enumerate(zip(ANNULI[:-1], ANNULI[1:]))})
    return result


def observations(offline: Particles, output: Particles, rules: dict, *, condition_edges: dict | None = None):
    """All-jet collection metrics plus explicitly covered association metrics."""
    result = {}
    x = features(offline) if condition_edges is not None else None
    possible = []
    if condition_edges is not None:
        possible = set(eligible_cells(x, condition_edges))
        for group in groups(offline, rules["same_side_radius"]):
            possible.update(particle_cells(group_features(offline, group, x), condition_edges))
    conditional = {key: {} for key in sorted(possible)}
    active = jet_cells(x, condition_edges) if condition_edges is not None else []

    def put(name, value):
        _append(result, name, value)
        for key in active:
            if key not in conditional:
                raise ValueError("Conditional population depends on an unregistered source group")
            _append(conditional[key], name, value)

    base, target = jet_summary(offline), jet_summary(output)
    for name, value in target.items():
        put(name, value)
    if all(k in target for k in JET_JOINT):
        put("jet_joint", [target[k] for k in JET_JOINT])
    for a, b in combinations(JET_JOINT, 2):
        if a in target and b in target:
            put(f"jet_corr_{a}_{b}", [target[a], target[b]])
    oa, ha = _axis(offline.p4.sum(axis=0)), _axis(output.p4.sum(axis=0))
    if oa is not None and ha is not None:
        put("jet_log_pt_ratio", np.log(ha[0]/oa[0]))
        put("jet_axis_dr", np.hypot(ha[1]-oa[1], wrap_phi(ha[2]-oa[2])))
    if base.get("mass", 0) > 0 and target.get("mass", 0) > 0:
        put("jet_log_mass_ratio", np.log(target["mass"]/base["mass"]))
    match = associate(offline, output, rules)
    sources = {j: tuple(h["offline"]) for h in match["hypotheses"] for j in h["hlt"]
               if h["offline"] and match["resolved"]}
    for i in range(len(output)):
        if condition_edges is not None:
            active = (particle_cells(group_features(offline, sources[i], x), condition_edges)
                      if i in sources else ["all"])
        track = {}
        for j, name in enumerate(TRACK[:4]):
            if output.valid[i, j]:
                track[name] = (float(output.tracking[i, j]) if j < 2
                               else float(np.log(output.tracking[i, j])))
        for j, name in enumerate(TRACK[4:]):
            if output.valid[i, j+2]:
                track[name] = float(output.tracking[i, j]/output.tracking[i, j+2])
        for name, value in track.items():
            put(name, value)
        for a, b in combinations(TRACK, 2):
            if a in track and b in track:
                put(f"track_corr_{a}_{b}", [track[a], track[b]])
    active = jet_cells(x, condition_edges) if condition_edges is not None else []
    if match["resolved"]:
        hs = match["hypotheses"]
        put("loss_rate", sum(len(h["offline"]) for h in hs if not h["hlt"])/max(1, len(offline)))
        put("merge_rate", sum(len(h["offline"]) > 1 for h in hs)/max(1, len(offline)))
        put("split_rate", sum(len(h["hlt"]) > 1 for h in hs)/max(1, len(offline)))
        put("additional_rate", sum(not h["offline"] for h in hs)/max(1, len(output)))
        for h in hs:
            if not h["offline"] or not h["hlt"]:
                continue
            source = offline.take(h["offline"])
            origin = _axis(source.p4.sum(axis=0))
            if origin is None:
                continue
            input_category = int(source.category[0]) if np.all(source.category == source.category[0]) else 5
            input_charge = int(np.sign(source.charge.sum()))
            source_valid = source.valid.any(axis=0)
            if condition_edges is not None:
                active = particle_cells(group_features(offline, tuple(h["offline"]), x), condition_edges)
            for j in h["hlt"]:
                response = [float(np.log(output.pt[j]/origin[0])), float(output.eta[j]-origin[1]),
                            float(wrap_phi(output.phi[j]-origin[2]))]
                for name, value in zip(("log_pt_ratio", "delta_eta", "delta_phi"), response):
                    put(name, value)
                put("log_energy_ratio", np.log(output.p4[j, 3]/source.p4[:, 3].sum()))
                put("category_charge_transition",
                        (input_category, input_charge, int(output.category[j]), int(output.charge[j])))
                for k, name in enumerate(("d0", "dz", "d0err", "dzerr")):
                    put(name+"_availability_transition",
                            (input_category, int(source_valid[k]), int(output.valid[j, k])))
                mask = sum(int(output.valid[j, k]) << k for k in range(4))
                values = [float(output.tracking[j, k]) if k < 2 else float(np.log(output.tracking[j, k]))
                          for k in range(4) if output.valid[j, k]]
                put(f"particle_joint_{mask}", [*response, *values])
    audit = dict(resolved=match["resolved"], jets=1, empty=len(output) == 0,
                        jet_moments=dict(multiplicity=float(len(output)), pt=float(output.pt.sum()),
                                         mass=target.get("mass", 0.)))
    # Membership derives only from offline inputs and allowed offline groups,
    # not which associations or emitted particles happened to be observed.
    return (result, audit, conditional) if condition_edges is not None else (result, audit)


def _transform(a, names, scales):
    a = np.asarray(a, np.float64).copy()
    if a.ndim == 1:
        a = a[:, None]
    for i, name in enumerate(names):
        if name in {"d0", "dz", "d0_significance", "dz_significance"}:
            a[:, i] = np.arcsinh(a[:, i]/scales[i])
        elif name == "mass":
            a[:, i] = np.log1p(a[:, i])
    return a


def coordinates(name):
    if name == "jet_joint":
        return list(JET_JOINT)
    if name.startswith("particle_joint_"):
        mask = int(name.rsplit("_", 1)[1])
        return ["log_pt_ratio", "delta_eta", "delta_phi", *[TRACK[k] for k in range(4) if mask & (1 << k)]]
    for prefix, names in (("jet_corr_", JET_JOINT), ("track_corr_", TRACK)):
        for a, b in combinations(names, 2):
            if name == f"{prefix}{a}_{b}":
                return [a, b]
    return [name]


def fit_registry(observed: dict, *, membership_hash: str, bins: int = 512) -> dict:
    """Input observations must come exclusively from authenticated fit_location.

    A worker supplies bounded, shared bottom-hash observations. Sampling counts
    and inclusion probabilities are bound separately by the preprocessing lock.
    """
    if bins < 32 or bins > 8192:
        raise ValueError("Unregistered metric resolution")
    rows = []
    rng = np.random.default_rng(20260919)
    for block, names in BLOCKS.items():
        for name in names:
            if name in CATEGORICAL:
                rows.append(dict(name=name, block=block, kind="categorical"))
                continue
            cols = coordinates(name)
            values = np.asarray(observed.get(name, []), np.float64).reshape(-1, len(cols))
            available = len(values) > 0
            raw_scale = np.median(abs(values), axis=0) if available else np.ones(len(cols))
            raw_scale[raw_scale <= 0] = 1.
            transformed = _transform(values, cols, raw_scale)
            q = np.quantile(transformed, [.05, .25, .5, .75, .95], axis=0) if available else np.zeros((5, len(cols)))
            scale = q[3]-q[1]
            tolerance = 64*np.finfo(np.float64).eps*np.maximum(1., abs(q).max(axis=0))
            scale = np.where(scale > tolerance, scale, q[4]-q[0])
            point_mass = scale <= tolerance
            scale[point_mass] = 1.
            joint = "joint" in name
            correlation = "_corr_" in name
            projection = rng.normal(size=(64, len(cols))) if joint else np.eye(len(cols))
            projection /= np.linalg.norm(projection, axis=1)[:, None]
            z = (transformed-q[2])/scale
            projected = z@projection.T
            bounds = (np.quantile(projected, [.001, .999], axis=0) if available
                      else np.zeros((2, len(projection))))
            extent = np.maximum(bounds[1]-bounds[0], 1.)
            rows.append(dict(name=name, block=block, kind="correlation" if correlation else "continuous",
                             coordinates=cols, raw_scale=raw_scale.tolist(), center=q[2].tolist(),
                             scale=scale.tolist(), point_mass=point_mass.tolist(), fit_available=available,
                             projections=projection.tolist(), bins=min(bins, 64) if correlation else bins,
                             numerical_degeneracy_tolerance=tolerance.tolist(),
                             low=(bounds[0]-.1*extent).tolist(), high=(bounds[1]+.1*extent).tolist()))
    return artifact("METRIC_REGISTRY", parents={"fit_location_membership": membership_hash},
                    blocks=BLOCKS, block_weight=1/6, observables=rows, slices=64, bootstrap=200,
                    approximation="shared_fixed_bins_with_observed_extrema_W1_bound_v1",
                    undefined_policy="separate_jet_coverage_mass_v1", conditional_minimum_jets=1000)


class Distribution:
    def __init__(self, definition):
        self.definition = definition
        self.jets = self.covered = 0
        self.categories = {}
        if definition["kind"] != "categorical":
            shape = (len(definition["projections"]), definition["bins"]+2)
            self.counts = np.zeros(shape, np.int64)
            self.sums = np.zeros(shape)
            self.minimum = np.full(shape, np.inf)
            self.maximum = np.full(shape, -np.inf)
            if definition["kind"] == "correlation":
                self.cross = np.zeros((shape[1], shape[1]), np.int64)

    def add(self, values):
        self.jets += 1
        self.covered += bool(len(values))
        d = self.definition
        if d["kind"] == "categorical":
            for value in values:
                key = ",".join(map(str, value))
                self.categories[key] = self.categories.get(key, 0)+1
            return
        if not len(values):
            return
        transformed = _transform(np.asarray(values).reshape(-1, len(d["coordinates"])),
                                 d["coordinates"], d["raw_scale"])
        z = (transformed-d["center"])/d["scale"]
        projected = z@np.asarray(d["projections"]).T
        indexes = []
        for j, column in enumerate(projected.T):
            bins = np.searchsorted(np.linspace(d["low"][j], d["high"][j], d["bins"]+1), column, side="right")
            indexes.append(bins)
            np.add.at(self.counts[j], bins, 1)
            np.add.at(self.sums[j], bins, column)
            np.minimum.at(self.minimum[j], bins, column)
            np.maximum.at(self.maximum[j], bins, column)
        if d["kind"] == "correlation":
            np.add.at(self.cross, (indexes[0], indexes[1]), 1)

    def merge(self, other, weight=1):
        if self.definition != other.definition or type(weight) is not int or weight < 0:
            raise ValueError("Summary definition/bootstrap weight differs")
        if not weight:
            return
        self.jets += weight*other.jets
        self.covered += weight*other.covered
        if self.definition["kind"] == "categorical":
            for key, value in other.categories.items():
                self.categories[key] = self.categories.get(key, 0)+weight*value
        else:
            self.counts += weight*other.counts
            self.sums += weight*other.sums
            self.minimum = np.minimum(self.minimum, other.minimum)
            self.maximum = np.maximum(self.maximum, other.maximum)
            if self.definition["kind"] == "correlation":
                self.cross += weight*other.cross

    def _rho(self):
        weights = self.cross
        total = weights.sum()
        if not total:
            return None
        x, y = weights.sum(axis=1), weights.sum(axis=0)
        rx, ry = (np.cumsum(x)-.5*x)/total, (np.cumsum(y)-.5*y)/total
        vx, vy = np.sum(x*(rx-.5)**2)/total, np.sum(y*(ry-.5)**2)/total
        if vx == 0 or vy == 0:
            return None
        return float(np.sum(weights*(rx[:, None]-.5)*(ry[None, :]-.5))/total/np.sqrt(vx*vy))

    def _rho_bound(self):
        # Within a rank bin, true midranks differ from its midpoint by at most
        # half its probability mass. Bound the L2 perturbation of each centered
        # rank vector, then of its unit vector; their dot product is Spearman.
        errors = []
        for j in range(2):
            counts = self.counts[j]
            if not counts.sum():
                return 0.
            p = counts/counts.sum()
            mid = np.cumsum(p)-.5*p
            sigma = np.sqrt(np.sum(p*(mid-.5)**2))
            varying = self.minimum[j] != self.maximum[j]
            delta = .5*np.sqrt(np.sum(p[varying]**3))
            if delta == 0:
                errors.append(0.)
            elif sigma <= delta:
                errors.append(2.)
            else:
                errors.append(min(2., 2*delta/sigma))
        return min(2., sum(errors))

    def compare(self, other):
        if self.definition != other.definition or self.jets != other.jets:
            raise ValueError("Comparison requires identical registry and jet coverage")
        coverage = abs(self.covered-other.covered)/max(self.jets, 1)
        kind = self.definition["kind"]
        if kind == "categorical":
            a, b = sum(self.categories.values()), sum(other.categories.values())
            tv = .5*sum(abs(self.categories.get(k, 0)/max(a, 1)-other.categories.get(k, 0)/max(b, 1))
                         for k in set(self.categories) | set(other.categories))
            return dict(discrepancy=max(coverage, float(tv)), error_bound=0., covered_jets=self.covered)
        if kind == "correlation":
            a, b = self._rho(), other._rho()
            delta = 0. if a is None and b is None else 1. if a is None or b is None else abs(a-b)
            return dict(discrepancy=max(coverage, delta),
                        error_bound=min(2., self._rho_bound()+other._rho_bound()), covered_jets=self.covered,
                        method="fixed_bin_spearman_approximation", reference_rho=a, proxy_rho=b)
        values, errors = [], []
        for j in range(len(self.counts)):
            a, b = self.counts[j] > 0, other.counts[j] > 0
            if not a.any() or not b.any():
                values.append(float(a.any() != b.any())); errors.append(0.)
                continue
            xa, xb = self.sums[j, a]/self.counts[j, a], other.sums[j, b]/other.counts[j, b]
            if len(self.definition["projections"]) == 1 and self.definition["point_mass"][0]:
                # With a degenerate fit reference, any bin whose actual range
                # excludes zero is certainly a mismatch. A straddling bin gives
                # an explicit uncertainty rather than fabricated exact equality.
                masses, uncertain = [], []
                for d, keep in ((self, a), (other, b)):
                    weights = d.counts[j, keep]/d.counts[j, keep].sum()
                    lo, hi = d.minimum[j, keep], d.maximum[j, keep]
                    measure = {}
                    for value, weight in zip(lo[lo == hi], weights[lo == hi]):
                        measure[float(value)] = measure.get(float(value), 0.)+float(weight)
                    masses.append(measure)
                    uncertain.append(float(weights[lo != hi].sum()))
                values.append(.5*sum(abs(masses[0].get(k, 0)-masses[1].get(k, 0))
                                    for k in set(masses[0]) | set(masses[1])))
                errors.append(sum(uncertain))
            else:
                values.append(float(wasserstein_distance(xa, xb, self.counts[j, a], other.counts[j, b])))
                errors.append(float(np.average(self.maximum[j, a]-self.minimum[j, a], weights=self.counts[j, a])+
                                    np.average(other.maximum[j, b]-other.minimum[j, b], weights=other.counts[j, b])))
        return dict(discrepancy=max(coverage, float(np.mean(values))), error_bound=float(np.mean(errors)),
                    covered_jets=self.covered, reference_empty=self.covered == 0)


class Summary:
    def __init__(self, registry):
        validate(registry, "METRIC_REGISTRY")
        self.registry = registry
        self.distributions = {r["name"]: Distribution(r) for r in registry["observables"]}
        self.jets = self.resolved = 0
        self.moments = dict(multiplicity=0., pt=0., mass=0.)

    def add(self, observation, audit):
        for name, distribution in self.distributions.items():
            distribution.add(observation.get(name, []))
        self.jets += 1
        self.resolved += int(audit["resolved"])
        if set(audit["jet_moments"]) != set(self.moments):
            raise ValueError("Jet-bias registry differs")
        for key, value in audit["jet_moments"].items():
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid jet moment")
            self.moments[key] += value

    def merge(self, other, weight=1):
        if self.registry != other.registry:
            raise ValueError("Summary registry differs")
        for name, distribution in self.distributions.items():
            distribution.merge(other.distributions[name], weight)
        self.jets += weight*other.jets
        self.resolved += weight*other.resolved
        for key in self.moments:
            self.moments[key] += weight*other.moments[key]

    def compare(self, proxy):
        if self.jets < 1:
            raise ValueError("An empty evaluation population cannot select a response")
        rows = {name: value.compare(proxy.distributions[name]) for name, value in self.distributions.items()}
        blocks = {block: float(np.mean([rows[n]["discrepancy"] for n in names]))
                  for block, names in BLOCKS.items()}
        return dict(primary_score=float(np.mean(list(blocks.values()))), blocks=blocks, observables=rows,
                    block_error_bounds={block: float(np.mean([rows[n]["error_bound"] for n in names]))
                                        for block, names in BLOCKS.items()},
                    reference_association_coverage=self.resolved/max(1, self.jets),
                    proxy_association_coverage=proxy.resolved/max(1, proxy.jets), jets=self.jets,
                    mean_jet_biases={k: dict(reference=self.moments[k]/self.jets,
                        proxy=proxy.moments[k]/self.jets,
                        relative=(proxy.moments[k]-self.moments[k])/self.moments[k]
                                 if self.moments[k] else (0. if not proxy.moments[k] else None))
                                     for k in self.moments})
