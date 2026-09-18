"""Portable complete set response with no raw donor-particle library.

The only generation arguments are offline particles, frozen response, jet key
and replica. Neither CMS targets nor JetClass2 native HLT/labels are accepted.
"""
from __future__ import annotations

import numpy as np

from .association import associate, groups
from .assumptions import physical_status
from .bridge import Particles
from .contracts import artifact, canonical_sha256, validate, validate_compatibility
from .families import fit_conditional, predict, calibrate_temperature
from .features import FEATURE_NAMES, features, group_features, fit_preprocessing
from .records import Reservoir
from .residuals import fit_backend, sample
from .rng import categorical, group_key
from .support import fit_support
from .topology import (calibration_records, jet_features, reference, condition_on_state,
                       decode_emission, COORDINATES)

SCALE_FLOOR = 1e-6


def collect(pairs, rules: dict, *, cap: int = 2_000_000, progress_every: int = 1000):
    reservoir = Reservoir(cap)
    counts = dict(jets=0, resolved=0, offline=0, hlt=0)
    unresolved_reasons = {}
    groups_seen = set()
    identity_hash = __import__("hashlib").sha256()
    for pair in pairs:
        if pair.hlt is None:
            raise PermissionError("Response calibration requires CMS paired observations")
        association = associate(pair.offline, pair.hlt, rules)
        records, report = calibration_records(pair.offline, pair.hlt, association, rules)
        counts["jets"] += 1
        counts["resolved"] += int(report["resolved"])
        counts["offline"] += len(pair.offline)
        counts["hlt"] += len(pair.hlt)
        for component in association["unresolved_components"]:
            reason = component["reason"]
            unresolved_reasons[reason] = unresolved_reasons.get(reason, 0)+1
        groups_seen.add(pair.source_group)
        encoded = pair.identity.encode()
        identity_hash.update(len(encoded).to_bytes(8, "big")+encoded)
        for record in records:
            reservoir.add(pair.identity, record)
        if counts["jets"] % progress_every == 0:
            print(f"CMS2JC2 phase=records jets={counts['jets']} resolved={counts['resolved']}", flush=True)
    if not counts["jets"]:
        raise ValueError("Empty calibration jet population")
    return reservoir, artifact("CALIBRATION_RECORDS", parents={"association": rules["content_hash"]},
                               counts=counts, source_groups=sorted(groups_seen),
                               unresolved_component_reasons=unresolved_reasons,
                               ordered_identity_sha256=identity_hash.hexdigest(),
                               sampling=reservoir.report())


def _conditioning(x):
    return x[:, [0, 2, 3, 17]]


def _fit_categorical(location, residual, candidate, lh, rh):
    x, y, w, _ = location
    states = sorted(set(tuple(map(int, s)) for s in y))
    lookup = {s: i for i, s in enumerate(states)}
    labels = np.array([lookup[tuple(s)] for s in y], np.int64)
    if len(states) == 1:
        model = None
    else:
        model = fit_conditional(x, labels, w, candidate_id=candidate, task="categorical",
                                classes=len(states), membership_hash=lh)
    calibration = dict(records=0, unseen_state_records=0, calibrated=False)
    if residual is not None:
        xr, yr, wr, _ = residual
        indexes = [i for i, state in enumerate(yr) if tuple(state) in lookup]
        calibration.update(records=len(yr), unseen_state_records=len(yr)-len(indexes))
        if model is not None and indexes:
            model = calibrate_temperature(model, xr[indexes],
                    np.array([lookup[tuple(yr[i])] for i in indexes]), wr[indexes],
                    residual_membership_hash=rh)
            calibration["calibrated"] = True
    return dict(kind="categorical", states=[list(s) for s in states], model=model,
                calibration=calibration)


def _fit_continuous(location, residual, candidate, lh, rh):
    x, y, w, _ = location
    target = np.asarray(y, np.float64)
    mean = fit_conditional(x, target.copy(), w, candidate_id=candidate, task="continuous",
                           membership_hash=lh, tracking_linear=True,
                           tracking_outputs=tuple(j for j in range(target.shape[1]) if j % 8 >= 4))
    error = target-predict(x, mean)
    scale_limits = [np.full(target.shape[1], np.log(SCALE_FLOOR)),
                    np.log(np.maximum(abs(error).max(axis=0), SCALE_FLOOR))]
    scale = fit_conditional(x, np.log(np.maximum(abs(error), SCALE_FLOOR)), w,
                            candidate_id=candidate, task="continuous", membership_hash=lh)
    edges = {name: np.unique(np.quantile(x[:, j], q)).tolist()
             for name, j, q in (("pt", 2, [.25, .5, .75]), ("eta", 3, [.25, .5, .75]),
                                ("crowding", 17, [1/3, 2/3]))}
    backends = []
    if residual is not None:
        xr, yr, wr, ids = residual
        with np.errstate(over="raise", invalid="raise"):
            scales = np.exp(np.clip(predict(xr, scale), *scale_limits))
        residual_values = (np.asarray(yr)-predict(xr, mean))/scales
        state_keys = [tuple(map(int, row[35:])) for row in xr]
        state_groups = {}
        for i, state in enumerate(state_keys):
            state_groups.setdefault(state, []).append(i)
        for state, indexes in sorted(state_groups.items()):
            backend = fit_backend(residual_values[indexes], wr[indexes], ids[indexes],
                      _conditioning(xr[indexes]), location_membership_hash=lh,
                      residual_membership_hash=rh, with_crowding=not candidate.startswith("A"),
                      edges=edges, coordinates=[f"child{i//8}_{COORDINATES[i%8]}" for i in range(target.shape[1])])
            backends.append(dict(state=list(state), backend=backend))
    calibration_y = target if residual is None else np.concatenate((target, residual[1]))
    return dict(kind="continuous", mean=mean, scale=scale, backends=backends,
                scale_limits=np.asarray(scale_limits).tolist(),
                response_limits=[calibration_y.min(axis=0).tolist(), calibration_y.max(axis=0).tolist()],
                limit_policy="fit_only_coordinate_envelope_and_clamp_report_v1",
                scale_floor=SCALE_FLOOR, missing_residual_policy="location_only_and_flag_unseen_state_v1")


def fit_response(location: Reservoir, residual: Reservoir, *, location_report: dict,
                 residual_report: dict, candidate_id: str, review: dict, rules: dict,
                 budget: str, source_hash: str) -> dict:
    validate_compatibility(review)
    validate(location_report, "CALIBRATION_RECORDS")
    validate(residual_report, "CALIBRATION_RECORDS")
    if (set(location_report["source_groups"]) & set(residual_report["source_groups"])
            or location_report["ordered_identity_sha256"] == residual_report["ordered_identity_sha256"]):
        raise PermissionError("Location and residual files must be disjoint")
    if any(r["parents"] != {"association": rules["content_hash"]}
           for r in (location_report, residual_report)):
        raise ValueError("Calibration association policies differ")
    lh, rh = location_report["content_hash"], residual_report["content_hash"]
    modules = {}
    support_rows = []
    support_data = None
    for name in location.modules():
        data, held = location.arrays(name), residual.arrays(name)
        print(f"CMS2JC2 phase=fit candidate={candidate_id} module={name} rows={len(data[0])}", flush=True)
        if name.endswith("_value"):
            modules[name] = _fit_continuous(data, held, candidate_id, lh, rh)
        else:
            modules[name] = _fit_categorical(data, held, candidate_id, lh, rh)
        # All modules use the same sampled-record memberships across families.
        # This pooled input support is a bounded diagnostic, not model fitting.
        if name == "singleton":
            support_rows.append(data[0])
            support_data = data
    estimable = {"singleton", "additional_count"} <= set(modules)
    if not support_rows:
        support_rows = [np.zeros((1, len(FEATURE_NAMES)))]
    return artifact("FITTED_RESPONSE", parents={"compatibility": review["content_hash"],
                    "association": rules["content_hash"], "location": lh, "residual": rh,
                    "source": source_hash}, candidate_id=candidate_id, budget=budget, rules=rules,
                    modules=modules, physical_status=physical_status(review), estimable=estimable,
                    input_support=fit_preprocessing(np.concatenate(support_rows), membership_hash=lh),
                    support_grid=(fit_support(support_data[0], support_data[2], support_data[3], membership_hash=lh)
                                  if support_data is not None else None),
                    calibration=dict(location=location_report, residual=residual_report),
                    coordinates=list(COORDINATES), output_count_truncation=False,
                    ordering="pt_desc_then_generation_key_v1", raw_donor_library=False)


class Generator:
    """Validate once when loading, then evaluate many jets without rehashing models."""

    def __init__(self, response: dict):
        validate(response, "FITTED_RESPONSE")
        from copy import deepcopy
        self.response = deepcopy(response)
        self.modules = self.response["modules"]
        self.rules = self.response["rules"]
        from .families import Predictor
        self.predictors = {}
        for module in self.modules.values():
            for name in ("model", "mean", "scale"):
                if module.get(name) is not None:
                    validate(module[name], "CONDITIONAL_MODEL")
                    self.predictors[id(module[name])] = Predictor(module[name])
            for row in module.get("backends", []):
                validate(row["backend"], "RESIDUAL_BACKEND")

    def __call__(self, offline: Particles, *, jet: str, replica: int = 0,
                 trace: bool = False, diagnostic_reverse_ties: bool = False):
        # Canonical input ordering also fixes floating summation order.
        p = offline.take(sorted(range(len(offline)), key=lambda i: offline.keys[i]))
        x = features(p)
        consumed, output, operations = set(), [], []
        flags = dict(unseen_state=0, residual_backoff=0, tail_clamped=0, support_clamps=0,
                     directionless_source=int(_directionless(p)), empty_output=False,
                     response_clamps=0, scale_clamps=0)
        support = self.response["input_support"]
        if len(x):
            flags["support_clamps"] = int(np.any((x < support["support_low"]) |
                                                        (x > support["support_high"]), axis=1).sum())

        def draw(module, row, key, component):
            model = self.modules.get(module)
            if model is None:
                flags["unseen_state"] += 1
                # Unseen merge/additional mechanisms do not invent particles;
                # an unseen singleton remains an explicitly flagged identity.
                return (1,) if module == "singleton" else (0,)
            probabilities = ([1.] if model["model"] is None
                             else self.predictors[id(model["model"])](row[None])[0])
            return tuple(model["states"][categorical(probabilities, jet, replica, component, key)])

        def emit(indices, module, key):
            row = group_features(p, indices, x) if indices else jet_features(p, x)
            state = draw(module+"_state", row, key, "identity")
            fitted = self.modules.get(module+"_value")
            if len(state) not in (4, 8) or fitted is None:
                flags["unseen_state"] += 1
                if indices:
                    output.append(p.take(indices))
                operations.append(dict(operation="unseen_emission_identity_fallback", inputs=list(indices)))
                return
            conditioned = condition_on_state(row, state)
            values = self.predictors[id(fitted["mean"])](conditioned[None])[0]
            backend = next((r["backend"] for r in fitted["backends"]
                            if tuple(r["state"]) == tuple(map(int, conditioned[35:]))), None)
            if backend is None:
                flags["unseen_state"] += 1
            else:
                residual, info = sample(backend, _conditioning(conditioned[None])[0], jet=jet,
                                        replica=replica, object_key=key, module=module, validate_model=False)
                flags["unseen_state"] += int(info["unseen_category"] or not info["supported"])
                flags["residual_backoff"] += int(info["backoff"])
                flags["tail_clamped"] += int(info["tail_clamped"])
                if residual is not None:
                    with np.errstate(over="raise", invalid="raise"):
                        log_scale = self.predictors[id(fitted["scale"])](conditioned[None])[0]
                        limited_scale = np.clip(log_scale, *fitted["scale_limits"])
                        flags["scale_clamps"] += int(np.any(log_scale != limited_scale))
                        scale = np.exp(limited_scale)
                    values = values + scale*residual
            limited_values = np.clip(values, *fitted["response_limits"])
            flags["response_clamps"] += int(np.any(values != limited_values))
            values = limited_values
            origin = reference(p, indices or tuple(range(len(p))))
            output.append(decode_emission(values, state, origin, key=key))
            operations.append(dict(operation=module, input_keys=[p.keys[i] for i in indices],
                                   output_keys=list(output[-1].keys), state=list(state)))

        for indices in groups(p, self.rules["same_side_radius"], reverse_ties=diagnostic_reverse_ties):
            if consumed.intersection(indices):
                continue
            key = group_key([p.keys[i] for i in indices])
            if draw("merge", group_features(p, indices, x), key, "topology")[0]:
                consumed.update(indices)
                emit(indices, "emission1", key)
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
        return result, dict(response_hash=self.response["content_hash"], jet_key=jet, replica=replica,
                            flags=flags, operations=operations if trace else [],
                            diagnostic_reverse_ties=diagnostic_reverse_ties,
                            physical_status=self.response["physical_status"])


def _directionless(p):
    return not len(p) or np.hypot(*p.p4.sum(axis=0)[:2]) == 0
