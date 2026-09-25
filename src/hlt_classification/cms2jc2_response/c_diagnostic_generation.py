"""Explicit frozen-model interventions; production Generator is not modified.

The traversal is a recorded copy of response.Generator at 5f8fecc8. Runtime
replay checks against that generator guard against accidental semantic drift.
Per-emission detail returned here is transient, never a durable particle trace.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .bridge import Particles
from .contracts import QUANTILES
from .association import groups
from .features import features, group_features
from .residuals import sample, _sqrt
from .response import Generator, _conditioning, _directionless
from .rng import categorical, normals, group_key
from .topology import condition_on_state, decode_emission, reference, jet_features

VARIANTS = ("FULL", "CENTRAL", "NO_ANGLE", "NO_PT", "INDEPENDENT")
COORDINATES = ("log_pt_ratio", "delta_eta", "delta_phi", "log_mass", "d0", "dz", "log_d0err", "log_dzerr")


def cell_for(backend, condition):
    bins = [int(np.searchsorted(backend["edges"][n], condition[j]))
            for n, j in (("pt", 1), ("eta", 2), ("crowding", 3))]
    cat = int(condition[0])
    keys = ([(cat, *bins)] if backend["with_crowding"] else [])
    keys += [(cat, *bins[:2]), (cat, bins[0]), (cat,)]
    lookup = {tuple(c["key"]): c for c in backend["cells"]}
    key = next((k for k in keys if k in lookup), None)
    return (lookup[key] if key is not None else None), key != keys[0]


def independent_factor(cell):
    # Preserve covariance of the actual stored factors, including rounding.
    a, b = np.asarray(cell["shared_factor"]), np.asarray(cell["independent_factor"])
    return _sqrt(a @ a.T + b @ b.T)


def residual_draw(backend, condition, *, variant, jet, replica, object_key, module, factors):
    residual, info = sample(backend, condition, jet=jet, replica=replica,
                            object_key=object_key, module=module, validate_model=False)
    cell, backoff = cell_for(backend, condition)
    detail = dict(missing_category=cell is None, coarse_bin_backoff=bool(cell is not None and backoff),
                  low_statistics=bool(cell is not None and not cell["supported"]))
    if residual is None:
        return residual, info, detail
    if variant == "INDEPENDENT":
        factor = factors.setdefault(id(cell), None)
        if factor is None:
            factor = factors[id(cell)] = independent_factor(cell)
        z = factor @ normals(jet, replica, "kinematics", module+":"+object_key, len(residual))
        u, q = ndtr(z), np.asarray(cell["quantiles"])
        residual = np.array([np.interp(u[j], backend["probabilities"], q[:, j]) for j in range(len(residual))])
        info = dict(info, tail_clamped=bool(np.any(u < QUANTILES[0]) or np.any(u > QUANTILES[-1])))
    elif variant == "CENTRAL":
        residual = np.zeros_like(residual)
    elif variant == "NO_ANGLE":
        residual = residual.copy()
        residual[1::8] = 0; residual[2::8] = 0
    elif variant == "NO_PT":
        residual = residual.copy(); residual[0::8] = 0
    return residual, info, detail


class DiagnosticGenerator(Generator):
    def __init__(self, response, variant):
        if variant not in VARIANTS:
            raise ValueError("Unknown C diagnostic intervention")
        super().__init__(response)
        self.variant, self.factors = variant, {}

    def __call__(self, offline, *, jet, replica=0):
        p = offline.take(sorted(range(len(offline)), key=lambda i: offline.keys[i]))
        x = features(p)
        consumed, output, operations, events, missing, draws = set(), [], [], [], [], {}
        flags = dict(unseen_state=0, residual_backoff=0, tail_clamped=0, support_clamps=0,
                     directionless_source=int(_directionless(p)), empty_output=False,
                     response_clamps=0, scale_clamps=0)
        support = self.response["input_support"]
        excursions = (x < support["support_low"]) | (x > support["support_high"])
        if len(x):
            flags["support_clamps"] = int(np.any(excursions, axis=1).sum())

        def draw(module, row, key, component):
            draws[module] = draws.get(module, 0)+1
            model = self.modules.get(module)
            if model is None:
                missing.append(module)
                flags["unseen_state"] += 1
                return (1,) if module == "singleton" else (0,)
            probabilities = ([1.] if model["model"] is None
                             else self.predictors[id(model["model"])](row[None])[0])
            return tuple(model["states"][categorical(probabilities, jet, replica, component, key)])

        def emit(indices, module, key):
            row = group_features(p, indices, x) if indices else jet_features(p, x)
            state = draw(module+"_state", row, key, "identity")
            fitted = self.modules.get(module+"_value")
            mechanism = ("additional" if not indices else "merged" if len(indices) > 1
                         else "split" if module == "emission2" else "singleton")
            event = dict(module=module, mechanism=mechanism, state=list(state),
                         missing_value_module=fitted is None, invalid_state=len(state) not in (4, 8))
            events.append(event)
            if len(state) not in (4, 8) or fitted is None:
                flags["unseen_state"] += 1
                if indices:
                    output.append(p.take(indices)); event["output"] = output[-1]
                operations.append(dict(operation="unseen_emission_identity_fallback", inputs=list(indices)))
                return
            conditioned = condition_on_state(row, state)
            values = self.predictors[id(fitted["mean"])](conditioned[None])[0]
            backend = next((r["backend"] for r in fitted["backends"]
                            if tuple(r["state"]) == tuple(map(int, conditioned[35:]))), None)
            event["missing_state_backend"] = backend is None
            if backend is None:
                flags["unseen_state"] += 1
            else:
                residual, info, detail = residual_draw(backend, _conditioning(conditioned[None])[0],
                    variant=self.variant, jet=jet, replica=replica, object_key=key, module=module, factors=self.factors)
                event.update(detail); event["quantile_tail"] = info["tail_clamped"]
                flags["unseen_state"] += int(info["unseen_category"] or not info["supported"])
                flags["residual_backoff"] += int(info["backoff"])
                flags["tail_clamped"] += int(info["tail_clamped"])
                if residual is not None:
                    with np.errstate(over="raise", invalid="raise"):
                        log_scale = self.predictors[id(fitted["scale"])](conditioned[None])[0]
                        limited_scale = np.clip(log_scale, *fitted["scale_limits"])
                        flags["scale_clamps"] += int(np.any(log_scale != limited_scale))
                        scale = np.exp(limited_scale)
                    event["scale"] = (log_scale, limited_scale)
                    values = values + scale*residual
            limited_values = np.clip(values, *fitted["response_limits"])
            flags["response_clamps"] += int(np.any(values != limited_values))
            event["response"] = (values, limited_values)
            origin = reference(p, indices or tuple(range(len(p))))
            output.append(decode_emission(limited_values, state, origin, key=key))
            event["output"] = output[-1]
            operations.append(dict(operation=module, input_keys=[p.keys[i] for i in indices],
                                   output_keys=list(output[-1].keys), state=list(state)))

        for indices in groups(p, self.rules["same_side_radius"]):
            if consumed.intersection(indices):
                continue
            key = group_key([p.keys[i] for i in indices])
            if draw("merge", group_features(p, indices, x), key, "topology")[0]:
                consumed.update(indices); emit(indices, "emission1", key)
        for i in sorted(set(range(len(p)))-consumed, key=lambda i: p.keys[i]):
            count = draw("singleton", x[i], p.keys[i], "survival")[0]
            if count:
                emit((i,), f"emission{count}", group_key([p.keys[i]]))
            else:
                operations.append(dict(operation="loss", input_keys=[p.keys[i]]))
        count = draw("additional_count", jet_features(p, x), "jet", "additional_count")[0]
        for j in range(count):
            emit((), "additional", f"additional:{j}")
        if output:
            result = Particles(*(np.concatenate([getattr(o, k) for o in output])
                                for k in ("p4", "charge", "category", "tracking", "valid")),
                               tuple(k for o in output for k in o.keys))
            result = result.take(sorted(range(len(result)), key=lambda i: (-float(result.pt[i]), result.keys[i])))
        else:
            result = p.take([])
        flags["empty_output"] = len(result) == 0
        return result, dict(flags=flags, operations=operations, events=events, missing_modules=missing, draws=draws,
                            support_excursions=excursions, input_categories=p.category)


def check_replay(expected, expected_info, actual, info, *, full):
    if info["operations"] != expected_info["operations"]:
        raise ValueError("Frozen C topology decisions changed")
    if set(actual.keys) != set(expected.keys):
        raise ValueError("Frozen C output identities changed")
    a = actual.take(sorted(range(len(actual)), key=lambda i: actual.keys[i]))
    b = expected.take(sorted(range(len(expected)), key=lambda i: expected.keys[i]))
    for name in (("p4", "charge", "category", "tracking", "valid") if full else ("charge", "category", "valid")):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise ValueError(f"Frozen C replay differs: {name}")
    if full and info["flags"] != expected_info["flags"]:
        raise ValueError("Frozen C replay flag accounting changed")
