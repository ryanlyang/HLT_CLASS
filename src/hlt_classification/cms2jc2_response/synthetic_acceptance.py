"""Known-mechanism coverage run inside the genuine CPU acceptance worker.

These supplemental synthetic fixtures never count as real detector closure.
"""
from __future__ import annotations

import numpy as np

from .association import policy
from .bridge import Particles, p4_from_coordinates
from .contracts import artifact
from .readers import Pair
from .response import collect, fit_response, Generator


def _particles(pt, eta, *, prefix):
    n = len(pt)
    return Particles(p4_from_coordinates(pt, eta, np.zeros(n), np.zeros(n)),
                     np.ones(n, dtype=int), np.zeros(n, dtype=int),
                     np.tile([.1, .2, .01, .02], (n, 1)), np.ones((n, 4), bool),
                     tuple(prefix+str(i) for i in range(n)))


def mechanism_pairs(group):
    scenarios = [([10.], [0.], [9.], [0.]),
                 ([10., 10.], [0., .01], [20.], [.005]),
                 ([10., 10., 10.], [0., .01, .02], [30.], [.01]),
                 ([20.], [0.], [10., 10.], [-.005, .005]),
                 ([10.], [0.], [], []),
                 ([10.], [0.], [9., 2.], [0., 1.])]
    for repeat in range(4):
        for i, (pt, eta, hp, he) in enumerate(scenarios):
            scale = 1+.03*repeat
            yield Pair(f"{group}:{repeat}:{i}", group,
                       _particles(np.asarray(pt)*scale, eta, prefix="o"),
                       _particles(np.asarray(hp)*scale, he, prefix="h"))


def exercise_modules(review, source_hash):
    rules = policy()
    loc, lr = collect(mechanism_pairs("synthetic_location"), rules, cap=7200)
    res, rr = collect(mechanism_pairs("synthetic_residual"), rules, cap=7200)
    required = {"merge", "singleton", "additional_count", "additional_state", "additional_value",
                "emission1_state", "emission1_value", "emission2_state", "emission2_value"}
    if not required <= set(loc.modules()):
        raise AssertionError("Synthetic acceptance did not expose all response modules")
    checked = {}
    for candidate in ("A_L", "B_L", "C_L"):
        response = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id=candidate,
                                review=review, rules=rules, budget="SYNTHETIC_ACCEPTANCE", source_hash=source_hash)
        model = Generator(response)
        for pair in mechanism_pairs("synthetic_check"):
            result, _ = model(pair.offline, jet=pair.identity)
            repeated, _ = model(pair.offline, jet=pair.identity)
            for field in ("p4", "charge", "category", "tracking", "valid"):
                if getattr(result, field).tobytes() != getattr(repeated, field).tobytes():
                    raise AssertionError("Known-mechanism deterministic replay failed")
        checked[candidate] = dict(modules=sorted(response["modules"]), response_hash=response["content_hash"])
    return artifact("SYNTHETIC_MECHANISM_ACCEPTANCE", candidates=checked, passed=True,
                    real_detector_validation=False, location_jets=24, residual_jets=24)
