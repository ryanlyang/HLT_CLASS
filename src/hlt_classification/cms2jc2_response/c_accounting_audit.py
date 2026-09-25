"""Read-only accounting from frozen-C reports and the fitted residual tables.

No ROOT reads, generation, fitting, rematching, scheduler calls or publication.
Pooled pT shares are deliberately not called mean per-jet charged fractions.
"""
from __future__ import annotations

import math
from pathlib import Path

from .contracts import artifact, load_json, sha256_file, validate
from .dev_campaign import product, validate_stage
from .dev_data import checked_file, COUNTS
from .c_diagnostic_generation import VARIANTS
from .c_diagnostic_worker import historical_replay_equal
from .dev_diagnostics import PID, CONDITIONS


def ratio(a, b):
    return a / b if b else None


def close(a, b, what, *, exact=False):
    equal = a == b if exact else math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)
    if not math.isfinite(a) or not math.isfinite(b) or not equal:
        raise ValueError("Frozen C accounting does not close: " + what)


def particle_accounting(cells):
    """Pool replicas with explicit jet-replica denominators, never independence."""
    jets = sum(c["jets"] for c in cells)
    empty = sum(c["empty_jets"] for c in cells)

    def total(name, field):
        return sum(c["variables"].get(name, {}).get(field, 0) for c in cells)

    # Jet count/scalar-pT metrics include empty jets; absent PID particle
    # histograms with zero count mean zero particles, not a missing jet sample.
    for name in ("jet_multiplicity", "jet_scalar_pt", *[f"jet_count_{i}" for i in range(6)]):
        if total(name, "count") != jets:
            raise ValueError("Incomplete jet accounting: " + name)
    particles = total("particle_pt", "count")
    pt = total("particle_pt", "sum")
    close(particles, total("jet_multiplicity", "sum"), "particle count", exact=True)
    close(pt, total("jet_scalar_pt", "sum"), "scalar pT")
    rows = []
    for cat, pid in enumerate(PID):
        n, momentum = total(pid + "_pt", "count"), total(pid + "_pt", "sum")
        if n < 0 or momentum < 0 or (n == 0 and momentum != 0):
            raise ValueError("Invalid PID momentum accounting")
        close(n, total(f"jet_count_{cat}", "sum"), "PID count " + pid, exact=True)
        rows.append(dict(pid=pid, particles=n, sum_pt=momentum,
            particles_per_jet=ratio(n, jets), scalar_pt_per_jet=ratio(momentum, jets),
            mean_particle_pt=ratio(momentum, n), pooled_pt_share=ratio(momentum, pt)))
    close(sum(r["particles"] for r in rows), particles, "sum PID counts", exact=True)
    close(sum(r["sum_pt"] for r in rows), pt, "sum PID momenta")
    known_charged = sum(rows[i]["sum_pt"] for i in (0, 3, 4))
    known_neutral = sum(rows[i]["sum_pt"] for i in (1, 2))
    unknown = rows[5]["sum_pt"]
    frac_count = total("jet_charged_fraction", "count")
    if frac_count != jets - empty:
        raise ValueError("Charged-fraction nonempty-jet denominator differs")
    return dict(jet_observations=jets, empty_jets=empty, particles=particles, sum_pt=pt,
        scalar_pt_per_jet=ratio(pt, jets), by_pid=rows,
        mean_per_nonempty_jet_charged_fraction=ratio(total("jet_charged_fraction", "sum"), frac_count),
        charged_fraction_jet_denominator=frac_count,
        pooled_known_charged_pt_share=ratio(known_charged, pt),
        pooled_known_neutral_pt_share=ratio(known_neutral, pt),
        pooled_unknown_pt_share=ratio(unknown, pt),
        pooled_actual_charged_pt_share_bounds=[ratio(known_charged, pt), ratio(known_charged + unknown, pt)])


def mechanism_accounting(stats, accounting):
    """Generated output budgets only; a mechanism is not a truth matching label."""
    jets = stats["jet_replicas"]
    if jets != accounting["jet_observations"]:
        raise ValueError("Mechanism jet-replica denominator differs")
    rows = []
    for key, value in sorted(stats["mechanisms"].items()):
        module, mechanism, pid_key = key.split("/")
        cat = int(pid_key.removeprefix("pid"))
        if cat not in range(6) or value["pt"]["n"] != value["particles"]:
            raise ValueError("Mechanism PID/count differs")
        n, pt = value["particles"], value["pt"]["sum"]
        rows.append(dict(module=module, mechanism=mechanism, pid=PID[cat], particles=n, sum_pt=pt,
            particles_per_jet=ratio(n, jets), scalar_pt_per_jet=ratio(pt, jets),
            mean_particle_pt=ratio(pt, n), pooled_pt_share=ratio(pt, accounting["sum_pt"])))
    for pid_row in accounting["by_pid"]:
        selected = [r for r in rows if r["pid"] == pid_row["pid"]]
        close(sum(r["particles"] for r in selected), pid_row["particles"], "mechanism PID counts", exact=True)
        close(sum(r["sum_pt"] for r in selected), pid_row["sum_pt"], "mechanism PID pT")
    return rows


def emission_accounting(stats):
    rows = []
    for key, value in sorted(stats["emission_groups"].items()):
        # /pid groups can overlap for split siblings; /all is disjoint.
        if not key.endswith("/all"):
            continue
        module, mechanism, _ = key.split("/")
        n = value["emissions"]
        if any(v < 0 or v > n for v in value.values()):
            raise ValueError("Invalid emission reason denominator")
        rows.append(dict(module=module, mechanism=mechanism, emissions=n,
                         counts=dict(value), rates={k: ratio(v, n) for k, v in value.items() if k != "emissions"}))
    if sum(r["emissions"] for r in rows) != stats["emissions"]:
        raise ValueError("Emission groups do not close")
    return rows


def coordinate_accounting(stats):
    rows = []
    for key, v in sorted(stats["coordinates"].items()):
        if not key.endswith("/response/log_pt_ratio"):
            continue
        n, clipped = v["exposures"], v["clipped"]
        if not 0 <= clipped <= n or any(v[k]["n"] != n for k in ("before", "after", "absolute_correction")):
            raise ValueError("Coordinate exposure denominator differs")
        rows.append(dict(scope=key, exposures=n, clipped=clipped, clipped_fraction=ratio(clipped, n),
            mean_before=ratio(v["before"]["sum"], n), mean_after=ratio(v["after"]["sum"], n),
            mean_absolute_correction=ratio(v["absolute_correction"]["sum"], n),
            mean_absolute_correction_when_clipped=ratio(v["absolute_correction"]["sum"], clipped),
            maximum_absolute_correction=v["absolute_correction"]["maximum"]))
    return rows


def residual_inventory(response):
    """Available cell counts; NOT evaluation occupancy or independent jet counts."""
    levels = {1: "category", 2: "category_pt", 3: "category_pt_eta", 4: "category_pt_eta_crowding"}
    rows, modules = {}, []
    for name, module in sorted(response["modules"].items()):
        if module.get("kind") != "continuous":
            continue
        backends = module["backends"]
        state_module = response["modules"].get(name.removesuffix("_value") + "_state", {})
        modules.append(dict(module=name, categorical_states=len(state_module.get("states", [])),
                            residual_state_backends=len(backends)))
        for state_index, row in enumerate(backends):
            backend = row["backend"]
            validate(backend, "RESIDUAL_BACKEND")
            for cell in backend["cells"]:
                key = cell["key"]
                if len(key) not in levels or key[0] not in range(6):
                    raise ValueError("Residual cell key differs")
                group = (name, int(key[0]), levels[len(key)])
                bucket = rows.setdefault(group, dict(cells=0, supported_cells=0, state_indexes=set(), jet_counts=[]))
                bucket["cells"] += 1
                bucket["supported_cells"] += int(cell["supported"])
                bucket["state_indexes"].add(state_index)
                bucket["jet_counts"].append(cell["jets"])
    return dict(modules=modules, cells=[dict(module=key[0], input_pid=PID[key[1]], level=key[2],
        cells=v["cells"], supported_cells=v["supported_cells"], states=len(v["state_indexes"]),
        minimum_jets_per_cell=min(v["jet_counts"]), maximum_jets_per_cell=max(v["jet_counts"]))
        for key, v in sorted(rows.items())], observed_selected_level_counts=None,
        occupancy_status="not recorded in DEV_C_REPORT/v1; inventory is not usage",
        support_rule="at least 1000 distinct calibration jets per retained supported cell",
        overlapping_cells=True)


def build(spec):
    # Authenticate old registration/receipts. This new reporting source must not
    # pretend to be the source that generated the old model or reports.
    study = validate_stage(spec, source=False)
    if spec["stage"] != "cdiag":
        raise ValueError("Accounting audit requires a frozen-C diagnostic stage")
    donor = load_json(checked_file(spec["parent_spec"]))
    response = product(donor, "candidate_C_L", "response")
    validate(response, "FITTED_RESPONSE")
    parents = dict(stage=spec["content_hash"], response=response["content_hash"],
        samples=spec["reuse"]["parents"]["samples"], ranges=spec["reuse"]["parents"]["ranges"])
    reports = {}
    for variant in VARIANTS:
        value = product(spec, "c_report_" + variant, "result")
        validate(value, "DEV_C_REPORT", parents=parents)
        if (value["variant"] != variant or value["jets"] != COUNTS["evaluation"] or value["replicas"] != [0, 1, 2]
                or value["all_registered_jets_included"] is not True or value["diagnostic_only"] is not True):
            raise ValueError("Frozen C report population/variant differs")
        reports[variant] = value
    full = reports["FULL"]["payload"]["cells"]
    accounting = {}
    for cohort in CONDITIONS:
        real = full.get("real/" + cohort)
        if real is None:
            continue
        sides = {side: particle_accounting([full[side + "/" + cohort]]) for side in ("offline", "real")}
        for variant, report in reports.items():
            cells = report["payload"]["cells"]
            for side in ("offline", "real"):
                if not historical_replay_equal(cells[side + "/" + cohort], full[side + "/" + cohort]):
                    raise ValueError("Frozen C reference populations differ")
            proxy = [cells[f"proxy{i}/{cohort}"] for i in range(3)]
            if any(c["jets"] != real["jets"] for c in proxy):
                raise ValueError("Frozen C proxy populations differ")
            sides[variant] = particle_accounting(proxy)
        accounting[cohort] = sides
    if accounting["all"]["real"]["jet_observations"] != COUNTS["evaluation"]:
        raise ValueError("Incomplete evaluation accounting")
    variants = {}
    for variant, report in reports.items():
        stats = report["instrumentation"]
        variants[variant] = dict(mechanisms=mechanism_accounting(stats, accounting["all"][variant]),
            emissions=emission_accounting(stats), pt_coordinate_clipping=coordinate_accounting(stats),
            categorical=stats["categorical"], topology_operations=stats["topology_operations"])
    return artifact("DEV_C_ACCOUNTING_AUDIT", parents={**parents,
        **{"report_" + v: r["content_hash"] for v, r in reports.items()}},
        audit_source_sha256=sha256_file(Path(__file__)), donor_source_commit=study["source"]["commit"],
        read_only=True, raw_data_accessed=False, generated_particles=False, fitted_model_modified=False,
        accounting=accounting, variants=variants, residual_inventory=residual_inventory(response),
        calibration_counts={k: v["counts"] for k, v in response["calibration"].items()},
        physical_status=response["physical_status"], diagnostic_only=True, no_automatic_winner=True,
        limitations=["Pooled pT shares are ratios of totals, not means of per-jet fractions.",
            "Unknown PID may be charged or neutral; retained separately, with charged-share bounds.",
            "Generated mechanism budgets include identity fallbacks and have no real-HLT mechanism truth labels.",
            "No input-pT-by-mechanism denominator was saved; output budgets are not response efficiencies.",
            "Detailed selected residual-bin occupancy and per-jet unknown-charge breakdown were not saved.",
            "Particle/replica observations are not independent jets; no uncertainty or transfer qualification."])


def render(audit):
    validate(audit, "DEV_C_ACCOUNTING_AUDIT")
    def f(value, percent=False):
        return "n/a" if value is None else f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    lines = ["FROZEN C MOMENTUM ACCOUNTING (read-only; no fits or generation)",
             "pT in GeV. Proxies pooled across three replicas; not independent jets.",
             "PID pT share = sum(pT in PID)/sum(all pT), NOT mean per-jet fraction."]
    sides = audit["accounting"]["all"]
    for name in ("offline", "real", *VARIANTS):
        row = sides[name]
        lines += [f"\n{name}: jet observations={row['jet_observations']} empty={row['empty_jets']}",
            f"Mean per-nonempty-jet charged pT fraction: {f(row['mean_per_nonempty_jet_charged_fraction'])}",
            "Pooled actual charged share bounds (unknown PID charge unavailable): " +
            " .. ".join(f(x, True) for x in row["pooled_actual_charged_pt_share_bounds"]),
            f"{'PID':<18} {'N/jet':>10} {'pT/particle':>13} {'sum pT/jet':>13} {'pT share':>12}"]
        for p in row["by_pid"]:
            lines.append(f"{p['pid']:<18} {f(p['particles_per_jet']):>10} {f(p['mean_particle_pt']):>13} "
                         f"{f(p['scalar_pt_per_jet']):>13} {f(p['pooled_pt_share'], True):>12}")
    lines += ["\nGENERATED MECHANISM/PID BUDGETS (FULL and NO_PT; others in --json)",
              "Output budgets only; no observed real-HLT mechanism labels or input-pT denominators."]
    for variant in ("FULL", "NO_PT"):
        lines += [variant, f"{'module/mechanism/PID':<48} {'N/jet':>9} {'sum pT/jet':>12} {'pT share':>11}"]
        for r in audit["variants"][variant]["mechanisms"]:
            key = f"{r['module']}/{r['mechanism']}/{r['pid']}"
            lines.append(f"{key:<48} {f(r['particles_per_jet']):>9} {f(r['scalar_pt_per_jet']):>12} {f(r['pooled_pt_share'], True):>11}")
    lines += ["\nFULL EMISSION REASONS (disjoint groups; reasons can overlap)",
        f"{'module/mechanism':<26} {'emissions':>10} {'backoff':>9} {'lowstat':>9} {'no backend':>11} {'no value':>9} {'bad state':>10}"]
    for r in audit["variants"]["FULL"]["emissions"]:
        rates = r["rates"]
        lines.append(f"{r['module']+'/'+r['mechanism']:<26} {r['emissions']:>10} " + " ".join(
            f"{f(rates[k], True):>10}" for k in ("coarse_bin_backoff", "low_statistics", "missing_state_backend", "missing_value_module", "invalid_state")))
    lines += ["\nFULL LOG-pT RESPONSE CLIPPING (log coordinates, not physical pT averages)",
        f"{'scope':<57} {'clipped':>10} {'mean |correction| when clipped':>32}"]
    for r in audit["variants"]["FULL"]["pt_coordinate_clipping"]:
        lines.append(f"{r['scope']:<57} {f(r['clipped_fraction'], True):>10} {f(r['mean_absolute_correction_when_clipped']):>32}")
    inventory = audit["residual_inventory"]
    lines += ["\nAVAILABLE RESIDUAL CELLS -- NOT EVALUATION USAGE", inventory["support_rule"],
        "States/cells overlap in calibration jets; their jet counts must not be summed."]
    for r in inventory["modules"]:
        lines.append(f"{r['module']}: categorical states={r['categorical_states']}, "
                     f"residual state backends={r['residual_state_backends']}")
    lines.append(f"{'module / input PID / level':<76} {'cells':>6} {'supported':>10} {'states':>7} {'jets/cell min..max':>19}")
    for r in inventory["cells"]:
        key = f"{r['module']} / {r['input_pid']} / {r['level']}"
        lines.append(f"{key:<76} {r['cells']:>6} {r['supported_cells']:>10} {r['states']:>7} "
                     f"{r['minimum_jets_per_cell']}..{r['maximum_jets_per_cell']}")
    lines += ["Selected-bin occupancy: unavailable in existing reports; a new replay would be required.",
        "\nCHARGED pT FRACTION BY OFFLINE-DEFINED COHORT (mean per nonempty jet)",
        f"{'cohort':<20} {'real jets':>10} {'real':>10} {'FULL':>10} {'CENTRAL':>10} {'NO_PT':>10}"]
    for cohort, rows in audit["accounting"].items():
        lines.append(f"{cohort:<20} {rows['real']['jet_observations']:>10} " + " ".join(
            f"{f(rows[k]['mean_per_nonempty_jet_charged_fraction']):>10}" for k in ("real", "FULL", "CENTRAL", "NO_PT")))
    lines += ["\nCalibration resolved/total jets (not matching accuracy):"]
    for role, counts in audit["calibration_counts"].items():
        lines.append(f"{role}: {counts['resolved']}/{counts['jets']}")
    lines += ["\nCount/pT accounting closure: PASS (integrity only, not mapping quality).", *audit["limitations"],
              "Producer conventions remain as registered; this does not verify physical compatibility.",
              "Development only. Final test accessed: False."]
    return "\n".join(lines)
