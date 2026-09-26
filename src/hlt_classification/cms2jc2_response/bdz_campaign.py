"""Separately authorized B_DZ calibration and development comparison only."""
from pathlib import Path

from . import dev_campaign as dev, bounded_campaign as bounded
from . import bdz_maps as maps, bdz_metrics as metrics
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref

KIND = "BDZ_STAGE"
STAGES = ("bdz_gate", "bdz_compare")


def protocol():
    return artifact("BDZ_PROTOCOL", candidates=list(maps.CANDIDATES), strengths=maps.STRENGTHS,
        calibration_jets=COUNTS["residual"], comparison_jets=COUNTS["evaluation"], replicas=[0, 1, 2], shards=4,
        quantiles=list(maps.QUANTILES), min_unique_jets=maps.MIN_JETS, calibration_max_bytes=maps.MAX_BYTES,
        coordinate_names=list(metrics.NAMES), thresholds=[list(t) for t in metrics.THRESHOLDS],
        score="equal_core_tv_tail_balanced_tracking_correlation", per_variable_guard=.02,
        confirmation_accessed=False, production_qualified=False, transfer_authorized=False)


def tasks(stage):
    if stage == "bdz_gate":
        return [dev.task("bz_acceptance", "bz_acceptance", 2, 32, 2),
                dev.task("bz_calibrate", "bz_calibrate", 36, 128, 8, ["bz_acceptance"])]
    if stage != "bdz_compare":
        raise ValueError("Unregistered B_DZ stage")
    names = [f"bz_eval_{i}" for i in range(4)]
    return [*[dev.task(n, "bz_evaluate", 36, 128, 8, shard=i) for i, n in enumerate(names)],
            dev.task("bz_select", "bz_report", 1, 32, 4, names)]


def read_selection(parent):
    """The old reason-list order depends on JSON key order, not science."""
    bounded.validate_stage(parent, source=False)
    if parent["stage"] != "bounded_compare":
        raise PermissionError("Import the completed development comparison, not confirmation")
    value, reg = dev.product(parent, "bc_select", "result"), bounded.registry(parent)
    validate(value, "BOUNDED_SELECTION", parents={"stage": parent["content_hash"],
        "protocol": bounded.protocol()["content_hash"], "registry": reg["content_hash"]})
    from .bounded_metrics import choose
    winner, reasons = choose(value["scores"])
    if (set(value["guard_failures"]) != set(reasons)
            or any(sorted(value["guard_failures"][k]) != sorted(reasons[k]) for k in reasons)
            or winner != "B_DZ" or value["selected"] != winner or value["selected_model"] != reg["models"][winner]
            or value["jets"] != COUNTS["evaluation"] or value["production_qualified"] is not False
            or value["confirmation_accessed"] is not False):
        raise ValueError("Authenticated B_DZ selection differs")
    for i in range(4):
        dev.verified_outputs(parent, f"bc_eval_{i}")
    if (Path(parent["root"])/"stages/bounded_confirm_r1/claims").exists():
        raise PermissionError("Confirmation has started; reusing this tuning protocol needs fresh review")
    return value


def gate(spec):
    if spec["stage"] == "bdz_gate":
        return spec
    if spec["stage"] == "bdz_compare":
        return load_json(checked_file(spec["parent_spec"]))
    raise ValueError("Invalid B_DZ ancestry")


def parent_compare(spec):
    return load_json(checked_file(gate(spec)["parent_spec"]))


def reuse(parent, study):
    from .dev_restart import EXECUTION_ONLY_FILES
    donor = bounded.validate_stage(parent, source=False)
    selected = read_selection(parent)
    for name in ("imported", "review", "numerical_environment"):
        if donor[name] != study[name]:
            raise ValueError("B_DZ reuse interface changed: "+name)
    for name, digest in donor["source"]["files"].items():
        if name not in EXECUTION_ONLY_FILES and study["source"]["files"].get(name) != digest:
            raise ValueError("B_DZ scientific donor source changed: "+name)
    a, b = Path(donor["root"]).resolve(), Path(study["root"]).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise PermissionError("B_DZ tuning must use a disjoint root")
    return artifact("BDZ_REUSE", parents={"study": study["content_hash"], "selection": selected["content_hash"],
        "parent_stage": parent["content_hash"], "model": selected["selected_model"]["content_hash"]},
        old_artifacts_modified=False, confirmation_accessed=False)


def registry(spec):
    g = gate(spec)
    value = dev.product(g, "bz_calibrate", "result")
    validate(value, "BDZ_REGISTRY", parents={"stage": g["content_hash"], "protocol": protocol()["content_hash"]})
    selected = read_selection(parent_compare(spec))
    maps.validate_map(value["mapping"])
    original, _, _ = bounded.donors(parent_compare(spec))
    samples = dev.product(dev.preparation_stage(original), "prepare", "samples")
    validate(value["mapping"], "BDZ_MAP", parents={"model": selected["selected_model"]["content_hash"],
                                                   "samples": samples["content_hash"]})
    if (value["calibration_jets"] != COUNTS["residual"] or value["candidates"] != list(maps.CANDIDATES)
            or value["confirmation_accessed"] is not False or value["durable_particle_arrays"] is not False):
        raise ValueError("B_DZ calibration registry differs")
    return value


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec["study"]))
    dev.validate_study(study, source=source)
    validate(spec, KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]})
    stage = spec["stage"]
    if (stage not in STAGES or spec["name"] != stage+"_r1" or spec["root"] != study["root"]
            or spec["protocol"] != protocol() or spec["tasks"] != tasks(stage)
            or study["site"]["partition"] != "debug" or spec["b_threads"] != 1
            or spec["scientific_qualification"] is not False or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("B_DZ stage registration differs")
    parent = load_json(checked_file(spec["parent_spec"]))
    if stage == "bdz_gate":
        if spec["reuse"] != reuse(parent, study):
            raise ValueError("B_DZ reuse evidence differs")
    else:
        validate_stage(parent, source=source)
        if parent["stage"] != "bdz_gate" or parent["study"] != spec["study"] or spec["reuse"] is not None:
            raise ValueError("B_DZ comparison ancestry differs")
        acceptance = dev.product(parent, "bz_acceptance", "result")
        validate(acceptance, "BDZ_ACCEPTANCE", parents={"stage": parent["content_hash"], "protocol": protocol()["content_hash"]})
        if acceptance["resource_envelope_ok"] is not True:
            raise PermissionError("B_DZ measured resource envelope exceeded")
        registry(spec)
    if spec["policy"] != parent["policy"]:
        raise ValueError("B_DZ association policy changed")
    return study


def publish(study, parent_ref, stage):
    parent = load_json(checked_file(parent_ref))
    spec = artifact(KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]},
        study=file_ref(Path(study["root"])/"study_spec.json"), root=study["root"], name=stage+"_r1", stage=stage,
        parent_spec=parent_ref, policy=parent["policy"], b_threads=1, protocol=protocol(), tasks=tasks(stage),
        reuse=reuse(parent, study) if stage == "bdz_gate" else None,
        scientific_qualification=False, resources_are_development_envelopes=True)
    validate_stage(spec)
    dev.stage_dir(spec).mkdir(parents=True, exist_ok=False)
    dev.write(spec["root"], f"stages/{spec['name']}/stage_spec.json", spec, KIND)
    dev.write(spec["root"], f"stages/{spec['name']}/command_plan.json", dev.command_plan(spec, study), "DEV_PLAN")
    return spec


def create(*, parent_spec, project_dir, source_commit, root):
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    donor = bounded.validate_stage(parent, source=False)
    read_selection(parent)
    a, b = Path(root).resolve(), Path(donor["root"]).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise PermissionError("Use a disjoint new B_DZ tuning root")
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in a.parents):
        raise PermissionError("Do not nest in another campaign")
    study = dev.create_study(project_dir=project_dir, source_commit=source_commit, root=root, partition="debug",
                             preparation_spec=checked_file(donor["imported"]["preparation_spec"]))
    return publish(study, ref, "bdz_gate")


def advance(parent_spec):
    ref = file_ref(parent_spec)
    parent = load_json(checked_file(ref))
    study = validate_stage(parent)
    if parent["stage"] != "bdz_gate":
        raise PermissionError("B_DZ tuning stops after this comparison; no automatic confirmation")
    return publish(study, ref, "bdz_compare")


def render(spec):
    validate_stage(spec, source=False)
    owner = "bz_calibrate" if spec["stage"] == "bdz_gate" else "bz_select"
    if not (dev.stage_dir(spec)/"receipts"/(owner+".json")).is_file():
        return "No completed B_DZ report yet; use monitor."
    if spec["stage"] == "bdz_gate":
        row = registry(spec)
        fitted = sum(c["status"] == "fitted" for c in row["mapping"]["cells"].values())
        return f"Calibration jets: {row['calibration_jets']}; fitted cells: {fitted}/28. Separately authorize comparison."
    row = dev.product(spec, owner, "result")
    reg = registry(spec)
    validate(row, "BDZ_SELECTION", parents={"stage": spec["content_hash"], "registry": reg["content_hash"],
                                            "protocol": protocol()["content_hash"]})
    winner, reasons = metrics.choose(row["scores"])
    if (row["selected"] != winner or row["guard_failures"] != reasons
            or row["selected_strength"] != maps.STRENGTHS[winner]
            or row["map_hash"] != reg["mapping"]["content_hash"] or row["jets"] != COUNTS["evaluation"]
            or row["confirmation_accessed"] is not False or row["production_qualified"] is not False):
        raise ValueError("B_DZ selection differs")
    if row["shard_hashes"] != [dev.product(spec, f"bz_eval_{i}", "result")["content_hash"] for i in range(4)]:
        raise ValueError("B_DZ report shard lineage differs")
    lines = ["candidate       score    core TV    tails     joint   guard failures"]
    for n in maps.CANDIDATES:
        r = row["scores"][n]
        lines.append(f"{n:<14} {r['score']:.5f} " + " ".join(f"{r['blocks'][k]:9.5f}" for k in ("core_tv", "tail_balanced", "joint"))
                     + "   " + (", ".join(reasons[n]) or "none"))
    lines += [f"Frozen development choice: {winner}", f"Source files: {len(row['by_file'])}; leave-one-file-out gains: {row['leave_one_file_out']}",
              f"Figures: {Path(spec['root'])/'figures'/spec['name']}",
              "Exploratory reuse of development jets. Confirmation/test untouched. Not production qualified."]
    from .bdz_worker import summaries
    import math
    p, tracks = summaries(row)
    def moments(parts):
        count = sum(r["count"] for r in parts)
        if not count:
            return "missing"
        mean = sum(r["sum"] for r in parts)/count
        sd = math.sqrt(max(0., sum(r["sumsq"] for r in parts)/count-mean*mean))
        return f"{mean:.6g} +/- {sd:.6g} (n={count})"
    lines += ["", "SELECTED TRACKING: mean +/- distribution SD (not uncertainty); proxy pools three replicas"]
    for n in metrics.NAMES:
        key = "particle_"+n
        real = p[winner]["cells"]["real/all"]["variables"]
        proxy = [p[winner]["cells"][f"proxy{i}/all"]["variables"] for i in range(3)]
        lines.append(f"{n}: CMS {moments([real[key]] if key in real else [])}; "
                     f"{winner} {moments([v[key] for v in proxy if key in v])}")
        r = tracks[winner]["real"]["tails"][n]
        pr = [tracks[winner][f"proxy{i}"]["tails"][n] for i in range(3)]
        for i, threshold in enumerate(metrics.THRESHOLDS[metrics.NAMES.index(n)]):
            a = f"{r['exceed'][i]}/{r['count']}"
            b = f"{sum(v['exceed'][i] for v in pr)}/{sum(v['count'] for v in pr)}"
            lines.append(f"  abs > {threshold:g}: real {a}; proxy {b}")
    lines += ["", "CORRECTION ACCOUNTING (valid-particle exposures, not any-event jet flags)",
              str(row["corrections"][winner])]
    return "\n".join(lines)
