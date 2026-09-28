"""One finite, separately authorized tracking repair screen after the audit."""
from pathlib import Path
import math

from . import dev_campaign as dev, bdz_audit_campaign as audit, bdz_campaign as bdz
from . import bdz_joint_maps as maps
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref
from .dev_restart import EXECUTION_ONLY_FILES

KIND = "BDZ_JOINT_STAGE"
CONTRACT = "CMS2JC2_RESPONSE_BDZ_JOINT_STAGE/v1"
STAGES = ("joint_gate", "joint_compare")


def protocol():
    return artifact("BDZ_JOINT_PROTOCOL", candidates=list(maps.CANDIDATES),
        calibration_jets=COUNTS["residual"], comparison_jets=COUNTS["evaluation"],
        shards=4, replicas=[0, 1, 2], minimum_pair_jets=maps.MIN_PAIR_JETS,
        minimum_bin_jets=maps.MIN_BIN_JETS, quantiles=list(maps.old.QUANTILES),
        quartiles=list(maps.QUARTILES), calibration_max_bytes=maps.MAX_BYTES,
        score="half_historical_half_pid_equal_tv_tails_pair_joint_tv",
        minimum_diagnostic_jets=50, guard_absolute=.02,
        audit_protocol=audit.protocol(), confirmation_accessed=False,
        production_qualified=False, transfer_authorized=False)


def tasks(stage):
    if stage == "joint_gate":
        return [dev.task("bj_acceptance", "bj_acceptance", 2, 32, 2),
                dev.task("bj_calibrate", "bj_calibrate", 36, 128, 8, ["bj_acceptance"])]
    if stage != "joint_compare":
        raise ValueError("Unregistered joint repair stage")
    names = [f"bj_eval_{i}" for i in range(4)]
    return [*[dev.task(n, "bj_evaluate", 36, 128, 8, shard=i) for i, n in enumerate(names)],
            dev.task("bj_select", "bj_report", 1, 32, 4, names)]


def completed(parent):
    kind = parent.get("contract")
    if kind == "CMS2JC2_RESPONSE_BDZ_AUDIT_DEBUG/v1":
        from .bdz_audit_debug import read
    elif kind == audit.CONTRACT:
        read = audit.read
    else:
        raise PermissionError("Joint repair requires a completed significance audit")
    row = read(parent)
    study = load_json(checked_file(parent["study"]))
    comparison = load_json(checked_file(parent["parent_spec"]))
    _, reg, selected = audit.completed(comparison)
    if row["historical_choice"] != selected["selected"]:
        raise ValueError("Audit and historical selection differ")
    return study, comparison, reg, row


def gate(spec):
    if spec["stage"] == "joint_gate":
        return spec
    if spec["stage"] == "joint_compare":
        return load_json(checked_file(spec["parent_spec"]))
    raise ValueError("Unregistered joint repair ancestry")


def donor(spec):
    return load_json(checked_file(gate(spec)["parent_spec"]))


def comparison(spec):
    return load_json(checked_file(donor(spec)["parent_spec"]))


def reuse(parent, study):
    previous, comp, reg, row = completed(parent)
    for k in ("imported", "review", "numerical_environment"):
        if previous[k] != study[k]:
            raise ValueError("Joint repair changed donor interface: "+k)
    for name, digest in previous["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("Joint repair changed frozen scientific source: "+name)
    audit.disjoint(study["root"], previous["root"])
    return artifact("BDZ_JOINT_REUSE", parents={"study": study["content_hash"],
        "audit": row["content_hash"], "audit_stage": parent["content_hash"],
        "receipt": dev.verified_outputs(parent, "ba_report")["content_hash"],
        "comparison": comp["content_hash"], "registry": reg["content_hash"]},
        historical_choice=row["historical_choice"], old_artifacts_modified=False,
        confirmation_accessed=False)


def accepted(spec):
    g = gate(spec)
    value = dev.product(g, "bj_acceptance", "result")
    validate(value, "BDZ_JOINT_ACCEPTANCE", parents={"stage": g["content_hash"],
        "protocol": protocol()["content_hash"], "reuse": g["reuse"]["content_hash"]})
    if (value["resource_envelope_ok"] is not True or value["serial_process_parity"] is not True
            or value["nonidentity_probe_exercised"] is not True or value["confirmation_accessed"] is not False
            or value["scientific_quality_gate"] is not False or value["probe_support_is_synthetic"] is not True
            or value["jets"] != min(32, COUNTS["residual"])):
        raise PermissionError("Joint acceptance incomplete or resource envelope exceeded")
    measurements = value["measurements"]
    if set(measurements) != {"calibrate", "evaluate"}:
        raise ValueError("Joint acceptance measurement registry differs")
    for row in measurements.values():
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("wall_seconds", "sampled_peak_tree_rss_bytes")):
            raise ValueError("Invalid joint acceptance measurement")
    seconds = {k: 2*m["wall_seconds"]*(COUNTS["residual"] if k == "calibrate" else COUNTS["evaluation"]//4)/value["jets"]
               for k, m in measurements.items()}
    ram = 1.5*max(m["sampled_peak_tree_rss_bytes"] for m in measurements.values())*18+2*maps.MAX_BYTES
    if (value["projected_seconds"] != seconds or value["projected_peak_bytes"] != ram
            or max(seconds.values()) > 8*3600 or ram > 128*1024**3):
        raise PermissionError("Joint acceptance resource projections differ or exceed envelope")
    return value


def registry(spec):
    g = gate(spec)
    acceptance = accepted(spec)
    row = dev.product(g, "bj_calibrate", "result")
    validate(row, "BDZ_JOINT_REGISTRY", parents={"stage": g["content_hash"],
        "protocol": protocol()["content_hash"], "acceptance": acceptance["content_hash"]})
    old = bdz.registry(comparison(spec))
    maps.validate_map(row["mapping"])
    validate(row["mapping"], "BDZ_JOINT_MAP", parents=old["mapping"]["parents"])
    if (row["calibration_jets"] != COUNTS["residual"] or row["ordered_identities"] != old["ordered_identities"]
            or row["historical_map_hash"] != old["mapping"]["content_hash"]
            or row["confirmation_accessed"] is not False or row["durable_particle_arrays"] is not False
            or row["candidates"] != list(maps.CANDIDATES)):
        raise ValueError("Joint calibration registry differs")
    return row


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec["study"]))
    dev.validate_study(study, source=source)
    validate(spec, KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]})
    stage = spec["stage"]
    if (stage not in STAGES or spec["name"] != stage+"_r1" or spec["root"] != study["root"]
            or spec["protocol"] != protocol() or spec["tasks"] != tasks(stage)
            or study["site"]["partition"] not in ("debug", "tier3") or spec["b_threads"] != 1
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Joint stage registration differs")
    parent = load_json(checked_file(spec["parent_spec"]))
    if stage == "joint_gate":
        if spec["reuse"] != reuse(parent, study):
            raise ValueError("Joint donor reuse differs")
    else:
        validate_stage(parent, source=source)
        if parent["stage"] != "joint_gate" or parent["study"] != spec["study"] or spec["reuse"] is not None:
            raise ValueError("Joint stage ancestry differs")
        registry(spec)
    if parent["policy"] != spec["policy"]:
        raise ValueError("Joint association policy changed")
    return study


def publish(study, parent_ref, stage):
    parent = load_json(checked_file(parent_ref))
    spec = artifact(KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]},
        study=file_ref(Path(study["root"])/"study_spec.json"), root=study["root"], name=stage+"_r1",
        stage=stage, parent_spec=parent_ref, policy=parent["policy"], b_threads=1,
        protocol=protocol(), tasks=tasks(stage), reuse=reuse(parent, study) if stage == "joint_gate" else None,
        scientific_qualification=False, resources_are_development_envelopes=True)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(spec["root"], f"stages/{spec['name']}/stage_spec.json", spec, KIND)
    dev.write(spec["root"], f"stages/{spec['name']}/command_plan.json", dev.command_plan(spec, study), "DEV_PLAN")
    return spec


def create(*, parent_spec, project_dir, source_commit, root, partition="debug"):
    if partition not in ("debug", "tier3"):
        raise ValueError("Choose debug or tier3 before creating this study")
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    previous, _, _, _ = completed(parent)
    audit.disjoint(root, previous["root"])
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in Path(root).resolve().parents):
        raise PermissionError("Do not nest inside another campaign")
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root,
        preparation_spec=checked_file(previous["imported"]["preparation_spec"]), partition=partition)
    return publish(study, ref, "joint_gate")


def advance(parent_spec):
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    study = validate_stage(parent)
    if parent["stage"] != "joint_gate":
        raise PermissionError("Stop after the bounded joint comparison; no further stage")
    registry(parent)
    return publish(study, ref, "joint_compare")


def read(spec):
    validate_stage(spec, source=False)
    if spec["stage"] == "joint_gate":
        return registry(spec)
    reg = registry(spec)
    row = dev.product(spec, "bj_select", "result")
    validate(row, "BDZ_JOINT_SELECTION", parents={"stage": spec["content_hash"],
        "registry": reg["content_hash"], "protocol": protocol()["content_hash"]})
    from .bdz_joint_metrics import choose
    winner, reasons = choose(row["scores"])
    if (row["selected"] != winner or row["guard_failures"] != reasons
            or row["jets"] != COUNTS["evaluation"] or row["historical_choice"] != gate(spec)["reuse"]["historical_choice"]
            or row["shard_hashes"] != [dev.product(spec, f"bj_eval_{i}", "result")["content_hash"] for i in range(4)]
            or any(row[k] is not False for k in ("confirmation_accessed", "production_qualified", "transfer_authorized", "automatic_followup"))):
        raise ValueError("Joint selection differs")
    return row


def render(spec, *, statistics=False):
    row = read(spec)
    if spec["stage"] == "joint_gate":
        fitted = sum(c["error"]["status"] == "fitted" for c in row["mapping"]["cells"].values())
        bins = sum(b["status"] == "fitted" for c in row["mapping"]["cells"].values() for b in c["significance"])
        return (f"Calibration jets: {row['calibration_jets']}; fitted PID pairs: {fitted}/12; "
                f"fitted conditional bins: {bins}/48. Separately authorize comparison.")
    lines = [f"Development jets: {row['jets']}; historical choice retained: {row['historical_choice']}",
             "candidate        score  historical   PID score  guard failures"]
    for name in maps.CANDIDATES:
        score = row["scores"][name]
        lines.append(f"{name:<14} {score['score']:8.5f} {score['historical']['score']:10.5f} "
                     f"{score['pid_score']:11.5f}  {', '.join(row['guard_failures'][name]) or 'none'}")
    lines += [f"New frozen development choice: {row['selected']}",
              "Source files: "+str(len(row["audit"])),
              "No confirmation/test access or automatic follow-up. Not production qualified."]
    if statistics:
        from .bdz_joint_worker import summaries
        from .bdz_audit_metrics import tv
        h, _ = summaries(row)
        def moment(parts):
            count = sum(p["count"] for p in parts)
            if not count:
                return "missing"
            mean = sum(p["sum"] for p in parts)/count
            sd = math.sqrt(max(0., sum(p["sumsq"] for p in parts)/count-mean**2))
            return f"{mean:.6g} +/- {sd:.6g}"
        for candidate in maps.CANDIDATES:
            cells = h[candidate]["cells"]
            real = cells["real/all"]["variables"]
            proxies = [cells[f"proxy{i}/all"]["variables"] for i in range(3)]
            lines += ["", candidate, f"{'observable':<32} {'CMS mean +/- SD':>26} {'proxy mean +/- SD':>26} {'TV':>9}"]
            for name in sorted(set(real).union(*(p.keys() for p in proxies))):
                r, ps = real.get(name), [p.get(name) for p in proxies]
                values = [tv(r["bins"], p["bins"]) if r and p else None for p in ps]
                distance = "n/a" if any(v is None for v in values) else f"{sum(values)/3:.5f}"
                lines.append(f"{name:<32} {moment([r] if r else []):>26} "
                             f"{moment([p for p in ps if p]):>26} {distance:>9}")
            lines.append("PID-specific significance (all six PIDs retained in JSON):")
            for pid in ("all", *map(str, range(6))):
                for name in ("d0_significance", "dz_significance"):
                    rr = row["pooled_proxy_diagnostics"]["real"][pid]["variables"][name]
                    pp = row["pooled_proxy_diagnostics"][candidate][pid]["variables"][name]
                    lines.append(f"  PID={pid} {name}: real n={rr['count']} mean={rr['mean']} SD={rr['sd']}; "
                                 f"proxy exposures={pp['count']} mean={pp['mean']} SD={pp['sd']} TV={pp['histogram_tv']}")
            lines.append("Correction counters (particle exposures): "+str(row["corrections"][candidate]))
        lines += ["SD is distribution width, not uncertainty. Proxy moments pool three dependent replicas.",
                  "Overall-chart TV averages separate replica distances; PID-chart TV compares the pooled proxy histogram.",
                  "Replicas are not additional independent jets. Full support/route/conditional diagnostics remain in the JSON.",
                  f"Frozen generator flags (any occurrence per jet-replica; denominator={3*row['jets']}): {row['flags']}"]
    return "\n".join(lines)
