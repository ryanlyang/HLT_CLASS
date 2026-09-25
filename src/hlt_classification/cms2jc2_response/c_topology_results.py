"""Authenticated stdout-only reader for the observed-topology diagnostic."""
from .contracts import validate
from .dev_campaign import preparation_stage, product, validate_stage
from .c_diagnostic_worker import inputs
from .c_topology import VIEWS


def read(spec):
    validate_stage(spec, source=False)
    if spec["stage"] != "ctopo":
        raise ValueError("Use an observed-topology diagnostic spec")
    samples = product(preparation_stage(spec), "prepare", "samples")
    _, _, _, parents = inputs(spec, dict(samples=samples))
    value = product(spec, "ct_report", "result")
    validate(value, "DEV_C_TOPOLOGY_REPORT", parents=parents)
    shards = [product(spec, f"ct_eval_{i}", "result") for i in range(4)]
    if (value["shard_hashes"] != [s["content_hash"] for s in shards] or set(value["views"]) != set(VIEWS)
            or value["observed_state_privileged"] is not True or value["diagnostic_only"] is not True
            or value["confirmation_accessed"] is not False or value["all_registered_jets_included"] is not True):
        raise ValueError("Observed-topology report registration differs")
    return value


def render(value):
    def f(x):
        return "n/a" if x is None else f"{x:.4f}"
    n, r, c = (value[k] for k in ("jets", "resolved", "comparable"))
    lines = ["FROZEN C: OBSERVED-TOPOLOGY ORACLE DIAGNOSTIC",
        f"Jets: {n}; association resolved: {r} ({100*r/n:.2f}%); comparable: {c}; unresolved: {n-r}; unestimable resolved: {r-c}",
        "Association coverage is not physical matching accuracy. No model refit.",
        "Missing continuous modules (jets): "+str(value["missing_modules"]),
        "Unresolved component reasons: "+str(value["unresolved_reasons"]),
        "Codec maximum errors: "+str(value["closure"]), "",
        "view                    jets     real scalar pT    proxy scalar pT    real ch.frac    proxy ch.frac"]
    for v in VIEWS:
        row = value["views"][v]
        a, b = row["real"], row["proxy"]
        lines.append(f"{v:<23} {row['jets']:>6} {f(a['scalar_pt_per_jet']):>18} {f(b['scalar_pt_per_jet']):>18} "
            f"{f(a['mean_per_nonempty_jet_charged_fraction']):>15} {f(b['mean_per_nonempty_jet_charged_fraction']):>16}")
    lines += ["", "IDENTICAL COMPARABLE JETS: PID MOMENTUM BUDGETS",
              "view                 PID               N/jet     mean particle pT    scalar pT/jet"]
    rows = [("REAL", value["views"]["FREE_COMPARABLE"]["real"])]
    rows += [(v, value["views"][v]["proxy"]) for v in ("FREE_COMPARABLE", "FIXED_FULL", "FIXED_CENTRAL")]
    for v, row in rows:
        for p in row["by_pid"]:
            lines.append(f"{v:<20} {p['pid']:<16} {f(p['particles_per_jet']):>9} "
                         f"{f(p['mean_particle_pt']):>20} {f(p['scalar_pt_per_jet']):>16}")
    lines += ["", "COMPARABLE MECHANISM BUDGETS (associations are hypotheses, not detector truth)",
              "view                 mechanism      PID                N/jet    scalar pT/jet"]
    scopes = [("REAL", value["observed_mechanisms"]["OBSERVED_COMPARABLE"])]
    scopes += [(v, value["views"][v]["mechanisms"]) for v in ("FREE_COMPARABLE", "FIXED_FULL", "FIXED_CENTRAL")]
    for v, entries in scopes:
        for row in entries:
            lines.append(f"{v:<20} {row['mechanism']:<14} {row['pid']:<16} {f(row['particles_per_jet']):>9} "
                         f"{f(row['scalar_pt_per_jet']):>16}")
    lines += ["", "CONDITIONAL COVERAGE (offline-defined cohorts overlap)"]
    lines += [f"{k:<18} jets={r['jets']} resolved={r['resolved']} comparable={r['comparable']}"
              for k, r in sorted(value["coverage"].items())]
    lines += ["", "ACTUALLY SELECTED FIXED RESIDUAL LEVELS (emission-replicas)"]
    for v, rows in value["selected_residual_levels"].items():
        lines.append(v)
        lines += [f"  {k}: {n}" for k, n in sorted(rows.items())]
    lines += ["", "Particle and jet kinematics, tracking comparisons and plots are in the full JSON/report directory.",
        "Three replicas are pooled for descriptive proxy means, not independent jets.",
        "Fixed topology/state uses observed HLT information: NOT a deployable mapping or qualification.",
        "Final test accessed: False"]
    return "\n".join(lines)
