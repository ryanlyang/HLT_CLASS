"""Bounded aggregate instrumentation. No identities or raw emissions persist."""
from copy import deepcopy
import numpy as np

from .c_diagnostic_generation import COORDINATES
from .features import FEATURE_NAMES

REASONS = ("missing_value_module", "invalid_state", "missing_state_backend", "missing_category",
           "coarse_bin_backoff", "low_statistics", "quantile_tail")


def moments(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic instrumentation")
    return dict(n=len(a), sum=float(a.sum()), sumsq=float(np.square(a).sum()),
                minimum=float(a.min()) if len(a) else None, maximum=float(a.max()) if len(a) else None)


def combine(a, b):
    """Add counts/sums; retain pooled extrema. Both inputs remain unchanged."""
    return accumulate(deepcopy(a), b)


def accumulate(result, b):
    """Bounded in-place aggregation, without copying all previous jets each time."""
    for key, value in b.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(value, dict):
            accumulate(result[key], value)
        elif key in ("minimum", "maximum"):
            nonnull = [v for v in (result[key], value) if v is not None]
            result[key] = (min(nonnull) if key == "minimum" else max(nonnull)) if nonnull else None
        else:
            result[key] += value
    return result


class Counters:
    def __init__(self):
        self.value = {}

    def add(self, info):
        flags = info["flags"]
        row = dict(jet_replicas=1, emissions=len(info["events"]),
                   input_particles=len(info["input_categories"]),
                   flags_per_jet={k: int(bool(v)) for k, v in flags.items()},
                   categorical={m: dict(draws=n, missing_module=info["missing_modules"].count(m))
                                for m, n in info["draws"].items()},
                   emission_groups={}, coordinates={}, support={}, mechanisms={}, topology_operations={})
        for op in info["operations"]:
            key = op["operation"]
            row["topology_operations"][key] = row["topology_operations"].get(key, 0)+1
        # Exposure counts are input particles per feature, not emitted particles.
        for category in np.unique(info["input_categories"]):
            matrix = info["support_excursions"][info["input_categories"] == category]
            for j, name in enumerate(FEATURE_NAMES):
                row["support"][f"pid{category}/{name}"] = dict(particles=len(matrix), excursions=int(matrix[:, j].sum()))
        for event in info["events"]:
            state = event["state"]
            cats = sorted(set(map(int, state[::4]))) if len(state) in (4, 8) else ["unknown"]
            scope = event["module"]+"/"+event["mechanism"]
            for cat in ["all", *[f"pid{c}" for c in cats]]:
                key = scope+"/"+cat
                counts = dict(emissions=1, **{r: int(event.get(r, False)) for r in REASONS})
                counts["response_clipped"] = int("response" in event and np.any(event["response"][0] != event["response"][1]))
                counts["scale_clipped"] = int("scale" in event and np.any(event["scale"][0] != event["scale"][1]))
                accumulate(row["emission_groups"].setdefault(key, {}), counts)
            for stage in ("response", "scale"):
                if stage not in event:
                    continue
                before, after = event[stage]
                for j, (a, b) in enumerate(zip(before, after)):
                    key = f"{scope}/pid{int(state[(j//8)*4])}/{stage}/{COORDINATES[j%8]}"
                    item = dict(exposures=1, clipped=int(a != b), before=moments([a]), after=moments([b]),
                                absolute_correction=moments([abs(a-b)]))
                    accumulate(row["coordinates"].setdefault(key, {}), item)
            output = event.get("output")
            if output is not None:
                for cat in np.unique(output.category):
                    indexes = np.flatnonzero(output.category == cat)
                    p = output.take(indexes)
                    item = dict(particles=len(p), pt=moments(p.pt), eta=moments(p.eta), phi=moments(p.phi))
                    for j, name in enumerate(("d0", "dz", "d0err", "dzerr")):
                        item[name] = moments(p.tracking[p.valid[:, j], j])
                    key = f"{scope}/pid{cat}"
                    accumulate(row["mechanisms"].setdefault(key, {}), item)
        accumulate(self.value, row)
