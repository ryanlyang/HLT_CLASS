"""Read-only authenticated B tracking report; no particle reads or scheduling."""
from .contracts import validate
from .dev_campaign import product, preparation_stage, validate_stage
from .b_tracking_generation import VARIANTS
from .b_tracking_worker import inputs, collect_shards
from .c_diagnostic_worker import historical_replay_equal
from .c_diagnostic_metrics import accumulate


def read(spec):
    validate_stage(spec, source=False)
    if spec["stage"] != "btrack":
        raise ValueError("Use a frozen B tracking stage")
    samples = product(preparation_stage(spec), "prepare", "samples")
    _, _, _, parents = inputs(spec, dict(samples=samples))
    report = product(spec, "bt_report", "result")
    validate(report, "DEV_B_TRACKING_REPORT", parents=parents)
    total, hashes = collect_shards(spec, parents)
    if (report["shard_hashes"] != hashes or report["variants"] != list(VARIANTS)
        or report["replicas"] != [0, 1, 2] or report["diagnostic_only"] is not True
        or report["confirmation_accessed"] is not False or report["all_registered_jets_included"] is not True
        or report["unchanged_kinematics_and_state"] is not True
        or any(not historical_replay_equal(report[k], v) for k, v in total.items())):
        raise ValueError("B tracking report differs from registered shards")
    return report


def render(report):
    lines = [f"FROZEN B TRACKING AUDIT: {report['jets']} development jets, replicas 0/1/2.",
             "All variants preserve FULL particle counts, PID, validity and four-vectors.",
             "CENTRAL = zero selected residuals, not a physical conditional mean."]
    for metric in ("particle_d0", "particle_dz", "particle_d0err", "particle_dzerr",
                   "particle_d0_significance", "particle_dz_significance", "jet_mass", "jet_charged_fraction"):
        lines += ["", metric, f"{'variant':<18} {'real mean':>13} {'proxy mean':>13} {'TV':>10}"]
        for v in VARIANTS:
            rows = [report["comparisons"][v]["comparisons"][f"proxy{i}/all"][metric] for i in range(3)]
            if any(r["status"] != "available" for r in rows):
                lines.append(v+": unavailable (see counts)")
            else:
                values = [sum(r[k] for r in rows)/3 for k in ("real_mean", "proxy_mean", "histogram_tv")]
                lines.append(f"{v:<18} {values[0]:>13.6g} {values[1]:>13.6g} {values[2]:>10.4f}")
    lines += ["", "APPLICABLE TRACKING COORDINATES (pooled replicas, not independent samples)",
              "Error-response coordinates below are log(mm), NOT mm.",
              f"{'variant/coordinate':<34} {'valid N':>10} {'central out%':>13} {'clip%':>9} {'scale clip%':>12} {'|clip| mean':>12}"]
    for v in VARIANTS:
        totals = {}
        for key, row in report["instrumentation"][v]["tracking_coordinates"].items():
            accumulate(totals.setdefault(key.rsplit("/", 1)[1], {}), row)
        for coord, row in sorted(totals.items()):
            n, scales = row["applicable"], row.get("scale_evaluated", 0)
            if not n:
                lines.append(v+"/"+coord+": no applicable coordinates")
                continue
            scale = f"{100*row['scale_clipped']/scales:.2f}" if scales else "n/a"
            lines.append(f"{v+'/'+coord:<34} {n:>10} {100*row['central_outside']/n:>13.2f} "
                f"{100*(row['low_clipped']+row['high_clipped'])/n:>9.2f} {scale:>12} "
                f"{row['absolute_correction']['sum']/n:>12.6g}")
    lines += ["", "Emission missing-state/backoff reasons:"]
    for v in VARIANTS:
        totals = {}
        for key, row in report["instrumentation"][v]["emission_groups"].items():
            if key.endswith("/all"):
                accumulate(totals, row)
        lines.append(v+": "+str(totals))
    lines += ["", "Full JSON includes PID/mask/mechanism counters, central and residual moments, selected cells,",
              "conditional distributions, model envelopes and plot paths. No automatic winner.",
              "Development only. Final test accessed: False."]
    return "\n".join(lines)
