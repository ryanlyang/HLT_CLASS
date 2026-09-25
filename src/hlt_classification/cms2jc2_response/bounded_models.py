"""Finite, portable response recipes; original fitted artifacts are never edited."""
from copy import deepcopy

import numpy as np

from .contracts import artifact, validate, with_content_hash
from .response import Generator

CANDIDATES = ("B", "B_DZ", "BC")
SCALES = (0., .25, .5, .75, 1.)


def check_donors(b, c):
    for model, name in ((b, "B_L"), (c, "C_L")):
        validate(model, "FITTED_RESPONSE")
        if model["candidate_id"] != name or not model["estimable"]:
            raise ValueError("Bounded comparison needs estimable original B_L/C_L")
    for key in ("parents", "rules", "calibration", "coordinates", "physical_status",
                "input_support", "support_grid", "ordering", "budget"):
        if b[key] != c[key]:
            raise ValueError("B/C calibration interface differs: " + key)
    if set(b["modules"]) != set(c["modules"]):
        raise ValueError("B/C module coverage differs")
    for name, module in b["modules"].items():
        if module["kind"] != c["modules"][name]["kind"]:
            raise ValueError("B/C module kinds differ")
        if not name.endswith("_value") and module["states"] != c["modules"][name]["states"]:
            raise ValueError("B/C fitted state registry differs")


def build(b, c, candidate, *, dz_scale=1.):
    check_donors(b, c)
    if candidate not in CANDIDATES or type(dz_scale) not in (int, float) or dz_scale not in SCALES:
        raise ValueError("Unregistered bounded model recipe")
    if candidate != "B_DZ" and dz_scale != 1.:
        raise ValueError("Only B_DZ has a residual scale")
    response = deepcopy(b)
    if candidate == "BC":
        for name in response["modules"]:
            if name.endswith("_value"):
                response["modules"][name] = deepcopy(c["modules"][name])
    elif candidate == "B_DZ" and dz_scale != 1.:
        for name, module in response["modules"].items():
            if not name.endswith("_value"):
                continue
            for row in module["backends"]:
                backend = row["backend"]
                for cell in backend["cells"]:
                    quantiles = np.asarray(cell["quantiles"], dtype=np.float64)
                    if quantiles.shape[1] not in (8, 16):
                        raise ValueError("Unexpected continuous coordinate count")
                    quantiles[:, 5::8] *= dz_scale
                    cell["quantiles"] = quantiles.tolist()
                row["backend"] = with_content_hash(backend)
    # Explicit derived runtime identity; never publish it as the original B fit.
    if candidate != "B":
        response.update(candidate_id="BOUNDED_"+candidate,
                        bounded_recipe=dict(candidate=candidate, dz_scale=dz_scale,
                            original_b=b["content_hash"], original_c=c["content_hash"]))
        response = with_content_hash(response)
    return artifact("BOUNDED_MODEL", parents={"B": b["content_hash"], "C": c["content_hash"]},
        candidate=candidate, dz_scale=dz_scale, runtime_response=response,
        topology_source="B_L", continuous_source="C_L" if candidate == "BC" else "B_L",
        original_fit_modified=False, production_qualified=False)


def validate_model(value, b, c):
    validate(value, "BOUNDED_MODEL")
    if build(b, c, value["candidate"], dz_scale=value["dz_scale"]) != value:
        raise ValueError("Frozen bounded response recipe changed")


def engines(models):
    return {name: Generator(row["runtime_response"]) for name, row in models.items()}


def check_state(base, base_info, output, info, *, tracking_only):
    """Compare by immutable generated keys, not pT-dependent output ordering."""
    if base_info["operations"] != info["operations"] or set(base.keys) != set(output.keys):
        raise ValueError("Bounded intervention changed B topology/state")
    a = base.take(sorted(range(len(base)), key=lambda i: base.keys[i]))
    b = output.take(sorted(range(len(output)), key=lambda i: output.keys[i]))
    for name in ("category", "charge", "valid"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    if tracking_only:
        np.testing.assert_array_equal(a.p4, b.p4)
        np.testing.assert_array_equal(a.tracking[:, [0, 2, 3]], b.tracking[:, [0, 2, 3]])
