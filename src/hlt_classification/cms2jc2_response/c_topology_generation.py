"""Privileged topology/state intervention; observed continuous values never predict.

Only the codec closure and observed-accounting path uses target coordinates.
Continuous prediction is a frozen copy of DiagnosticGenerator.emit at 36a0190.
"""
from dataclasses import dataclass
import numpy as np

from .bridge import Particles
from .contracts import validate
from .features import features, group_features
from .rng import group_key
from .topology import (calibration_records, condition_on_state, decode_emission,
                       emission_coordinates, emission_state, jet_features, reference)
from .response import Generator, _conditioning, _directionless
from .c_diagnostic_generation import cell_for, residual_draw

LEVELS = {1: "category", 2: "category_pt", 3: "category_pt_eta", 4: "category_pt_eta_crowding"}


@dataclass(frozen=True)
class Emission:
    # Deliberately excludes observed continuous targets and target feature rows.
    module: str
    mechanism: str
    key: str
    indices: tuple
    state: tuple


def join(outputs, empty):
    if not outputs:
        return empty.take([])
    result = Particles(*(np.concatenate([getattr(p, name) for p in outputs])
        for name in ("p4", "charge", "category", "tracking", "valid")),
        tuple(k for p in outputs for k in p.keys))
    return result.take(sorted(range(len(result)), key=lambda i: (-float(result.pt[i]), result.keys[i])))


def info_for(p):
    return dict(flags=dict(unseen_state=0, residual_backoff=0, tail_clamped=0, support_clamps=0,
        directionless_source=int(_directionless(p)), empty_output=False, response_clamps=0, scale_clamps=0),
        operations=[], events=[], missing_modules=[], draws={},
        support_excursions=np.zeros((len(p), 43), bool), input_categories=p.category)


def observed_emissions(p, hlt, association, rules):
    """Verify canonical training-target replay; return state-only oracle registry."""
    validate(association, "ASSOCIATION", parents={"policy": rules["content_hash"]})
    if (association["offline_keys"] != list(p.keys) or association["hlt_keys"] != list(hlt.keys)
            or not association["resolved"] or association["unresolved_components"]):
        raise ValueError("Observed topology requires a fully resolved, identity-aligned association")
    hypotheses = association["hypotheses"]
    for name, size in (("offline", len(p)), ("hlt", len(hlt))):
        indices = [i for h in hypotheses for i in h[name]]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(range(size)):
            raise ValueError("Observed association does not cover every particle exactly once")
    records, _ = calibration_records(p, hlt, association, rules)
    values = {r.key: r for r in records if r.module.endswith("_value")}
    if len(values) != sum(r.module.endswith("_value") for r in records):
        raise ValueError("Duplicate canonical emission record")
    x, emissions, seen = features(p), [], set()
    truth = info_for(p)
    closure = dict(p4_maximum=0., tracking_maximum=0., source_energy_projection_maximum=0.)
    additional = sorted((j for h in hypotheses if not h["offline"] for j in h["hlt"]), key=lambda j: hlt.keys[j])
    for h in hypotheses:
        indices = tuple(h["offline"])
        targets = sorted(h["hlt"], key=lambda j: (-float(hlt.pt[j]), hlt.keys[j]))
        if not targets:
            if len(indices) != 1:
                raise ValueError("Invalid observed loss")
            truth["operations"].append(dict(operation="loss"))
            continue
        if not indices and len(targets) != 1 or len(indices) > 1 and len(targets) != 1 or len(targets) > 2:
            raise ValueError("Unregistered observed emission cardinality")
        module = "additional" if not indices else f"emission{len(targets)}"
        mechanism = "additional" if not indices else "merged" if len(indices) > 1 else "split" if len(targets) == 2 else "singleton"
        raw_key = group_key([p.keys[i] for i in indices]) if indices else "additional"
        record_key = raw_key+":"+":".join(hlt.keys[j] for j in targets)
        key = raw_key if indices else f"additional:{additional.index(targets[0])}"
        out = hlt.take(targets)
        state = emission_state(out)
        row = group_features(p, indices, x) if indices else jet_features(p, x)
        origin = reference(p, indices or tuple(range(len(p))))
        coordinates = emission_coordinates(out, origin)
        record = values.get(record_key)
        if (record is None or record.module != module+"_value"
            or not np.array_equal(record.x, condition_on_state(row, state))
            or not np.array_equal(record.target, coordinates) or record_key in seen):
            raise ValueError("Oracle registry differs from canonical calibration records")
        seen.add(record_key)
        decoded = decode_emission(coordinates, state, origin, key=key)
        for name in ("category", "charge", "valid"):
            if not np.array_equal(getattr(decoded, name), getattr(out, name)):
                raise ValueError("Observed codec categorical closure failed: "+name)
        # The bridge admits source-rounding spacelike p4. Compare against its
        # declared nonnegative-mass projection, recording the raw E correction.
        expected_p4 = out.p4.copy()
        expected_p4[:, 3] = np.sqrt(np.maximum(out.p4[:, 3]**2, (out.p4[:, :3]**2).sum(axis=1)))
        closure["source_energy_projection_maximum"] = max(closure["source_energy_projection_maximum"],
            float(np.max(np.abs(expected_p4[:, 3]-out.p4[:, 3]))))
        for name, expected in (("p4", expected_p4), ("tracking", out.tracking)):
            actual = getattr(decoded, name)
            if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8):
                raise ValueError("Observed codec continuous closure failed: "+name)
            closure[name+"_maximum"] = max(closure[name+"_maximum"], float(np.max(np.abs(actual-expected))))
        emissions.append(Emission(module, mechanism, key, indices, state))
        truth["events"].append(dict(module=module, mechanism=mechanism, state=list(state), output=out))
        truth["operations"].append(dict(operation=module))
    if seen != set(values):
        raise ValueError("Observed emission registry omits canonical targets")
    truth["flags"]["empty_output"] = len(hlt) == 0
    return tuple(sorted(emissions, key=lambda e: (e.module, e.key))), truth, closure


class FixedGenerator(Generator):
    def missing(self, emissions):
        return sorted({e.module+"_value" for e in emissions if e.module+"_value" not in self.modules})

    def __call__(self, p, emissions, *, jet, replica=0, variant="FULL"):
        if variant not in ("FULL", "CENTRAL"):
            raise ValueError("Unregistered observed-topology intervention")
        missing = self.missing(emissions)
        if missing:
            raise ValueError("Fixed topology unestimable; missing continuous modules: "+", ".join(missing))
        x, info, outputs = features(p), info_for(p), []
        support = self.response["input_support"]
        info["support_excursions"] = (x < support["support_low"]) | (x > support["support_high"])
        flags = info["flags"]
        flags["support_clamps"] = int(np.any(info["support_excursions"], axis=1).sum())
        occupancy = {}
        for e in emissions:
            row = group_features(p, e.indices, x) if e.indices else jet_features(p, x)
            conditioned = condition_on_state(row, e.state)
            fitted = self.modules[e.module+"_value"]
            values = self.predictors[id(fitted["mean"])](conditioned[None])[0]
            backend = next((r["backend"] for r in fitted["backends"]
                            if tuple(r["state"]) == tuple(map(int, conditioned[35:]))), None)
            event = dict(module=e.module, mechanism=e.mechanism, state=list(e.state), missing_state_backend=backend is None)
            level = "missing_state_backend"
            if backend is None:
                flags["unseen_state"] += 1
            else:
                condition = _conditioning(conditioned[None])[0]
                cell, _ = cell_for(backend, condition)
                level = "missing_category" if cell is None else LEVELS[len(cell["key"])]
                residual, detail, extra = residual_draw(backend, condition, variant=variant,
                    jet=jet, replica=replica, object_key=e.key, module=e.module, factors={})
                event.update(extra); event["quantile_tail"] = detail["tail_clamped"]
                flags["unseen_state"] += int(detail["unseen_category"] or not detail["supported"])
                flags["residual_backoff"] += int(detail["backoff"])
                flags["tail_clamped"] += int(detail["tail_clamped"])
                if residual is not None:
                    with np.errstate(over="raise", invalid="raise"):
                        scale = self.predictors[id(fitted["scale"])](conditioned[None])[0]
                        limited_scale = np.clip(scale, *fitted["scale_limits"])
                        values = values + np.exp(limited_scale)*residual
                    flags["scale_clamps"] += int(np.any(scale != limited_scale))
                    event["scale"] = (scale, limited_scale)
            scope = f"{e.module}/{e.mechanism}/input_pid{int(row[0])}/{level}"
            occupancy[scope] = occupancy.get(scope, 0)+1
            limited = np.clip(values, *fitted["response_limits"])
            flags["response_clamps"] += int(np.any(values != limited))
            event["response"] = (values, limited)
            out = decode_emission(limited, e.state, reference(p, e.indices or tuple(range(len(p)))), key=e.key)
            outputs.append(out); event["output"] = out
            info["events"].append(event)
            info["operations"].append(dict(operation=e.module))
        result = join(outputs, p)
        flags["empty_output"] = len(result) == 0
        info["selected_residual_levels"] = occupancy
        return result, info
