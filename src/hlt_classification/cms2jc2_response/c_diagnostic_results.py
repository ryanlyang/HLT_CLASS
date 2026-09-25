"""Compact read-only display; conditional metrics remain in authenticated JSON."""
from pathlib import Path

from .c_diagnostic_generation import VARIANTS
from .contracts import validate
from .dev_campaign import product, stage_dir, validate_stage, verified_outputs


def render(spec):
    validate_stage(spec)
    if spec["stage"] != "cdiag":
        raise ValueError("Use a frozen-C diagnostic specification")
    lines = ["Frozen C: same 10,000 development jets, replicas 0/1/2; no refit or rematching.",
             "Means and histogram TV below are averages across replicas, not independent measurements."]
    metrics = ("jet_multiplicity", "jet_mass", "jet_width", "jet_pt_ratio", "jet_axis_dr", "jet_charged_fraction",
               "jet_scalar_pt", "particle_d0", "particle_dz", "particle_d0err", "particle_dzerr")
    reports = {}
    for variant in VARIANTS:
        owner = "c_report_"+variant
        if not (stage_dir(spec)/"receipts"/(owner+".json")).is_file():
            lines.append(variant+": no completed report yet")
            continue
        report = product(spec, owner, "result")
        validate(report, "DEV_C_REPORT")
        if report["parents"]["stage"] != spec["content_hash"] or report["variant"] != variant:
            raise ValueError("Frozen C report does not belong to this study/variant")
        reports[variant] = report
    for metric in metrics:
        lines += ["", metric, f"{'variant':<14} {'real mean':>13} {'proxy mean':>13} {'TV':>10}"]
        for variant, report in reports.items():
            rows = [report["comparisons"][f"proxy{i}/all"][metric] for i in range(3)]
            if any(r["status"] != "available" for r in rows):
                lines.append(f"{variant:<14} unavailable (see counts in report)")
            else:
                values = [sum(r[k] for r in rows)/3 for k in ("real_mean", "proxy_mean", "histogram_tv")]
                lines.append(f"{variant:<14} {values[0]:>13.6g} {values[1]:>13.6g} {values[2]:>10.4f}")
    lines += ["", "EMISSION-LEVEL INSTRUMENTATION (not old any-particle-per-jet rates)"]
    for variant, report in reports.items():
        stats = report["instrumentation"]
        groups = [v for k, v in stats["emission_groups"].items() if k.endswith("/all")]
        denominator = stats["emissions"]
        def rate(key):
            return f"{100*sum(g[key] for g in groups)/denominator:.2f}%" if denominator else "n/a"
        lines.append(f"{variant}: emissions={denominator}, response clipped={rate('response_clipped')}, "
                     f"coarse-bin backoff={rate('coarse_bin_backoff')}, low-statistics={rate('low_statistics')}")
        row = verified_outputs(spec, "c_report_"+variant)["outputs"]["result"]
        lines.append("  Report: "+str(Path(spec["root"])/row["relative"]))
        lines.append("  Figures: "+str(Path(spec["root"])/"figures"/spec["name"]/variant))
    lines += ["", "Development only. No automatic winner. Final test accessed: False."]
    return "\n".join(lines)
