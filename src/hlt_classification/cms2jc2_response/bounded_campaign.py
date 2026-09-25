"""Three fixed stages, frozen donors, and a separate one-shot confirmation gate."""
from pathlib import Path

from . import dev_campaign as c
from .contracts import artifact, load_json, validate
from .dev_data import COUNTS, checked_file, file_ref
from .bounded_models import CANDIDATES, SCALES, build, check_donors, validate_model
from .bounded_metrics import BLOCKS, BIAS_NAMES, choose

KIND = "BOUNDED_STAGE"
STAGES = ("bounded_gate", "bounded_compare", "bounded_confirm")
CONFIRM_JETS = 20_000


def protocol():
    return artifact("BOUNDED_PROTOCOL", candidates=list(CANDIDATES), dz_scales=list(SCALES),
        calibration_jets=COUNTS["residual"], comparison_jets=COUNTS["evaluation"],
        confirmation_jets=CONFIRM_JETS, shards=4, replicas=[0, 1, 2],
        blocks=BLOCKS, joint_block="overall_JET_JOINT_Pearson_abs_difference_over_two",
        missing="both_absent_zero_one_absent_one; always_disclose", conditional_min_jets=1000,
        bias_names=list(BIAS_NAMES), block_regression_limit=.02, bias_regression_limit=.02,
        simplicity_order=list(CANDIDATES), bootstrap=dict(draws=200, seed=20260925, unit="source_file", minimum_files=4),
        confirmation_min_absolute_gain=.005, confirmation_min_relative_gain=.1,
        quality_failure_is_result=True, production_qualified=False, no_automatic_transfer=True,
        confirmation_consumes_outer_confirm_subset=True, tuning_stops_after_confirmation=True)


def tasks(stage):
    if stage == "bounded_gate":
        return [c.task("bc_acceptance", "bc_acceptance", 2, 32, 2),
                c.task("bc_calibrate", "bc_calibrate", 36, 128, 8, ["bc_acceptance"])]
    if stage not in STAGES[1:]:
        raise ValueError("Unregistered bounded stage")
    prefix = "bc" if stage == "bounded_compare" else "bf"
    names = [f"{prefix}_eval_{i}" for i in range(4)]
    return [*[c.task(name, "bc_evaluate", 36, 128, 8, shard=i) for i, name in enumerate(names)],
            c.task("bc_select" if prefix == "bc" else "bf_report", "bc_report", 1, 32, 4, names)]


def gate_spec(spec):
    current = spec
    for _ in range(3):
        if current.get("stage") == "bounded_gate":
            return current
        current = load_json(checked_file(current["parent_spec"]))
    raise ValueError("Bounded gate ancestry differs")


def donors(spec):
    original = load_json(checked_file(gate_spec(spec)["parent_spec"]))
    b = c.product(original, "candidate_B_L", "response")
    ct = c.product(original, "candidate_C_L", "response")
    check_donors(b, ct)
    return original, b, ct


def registry(spec):
    gate = gate_spec(spec)
    value = c.product(gate, "bc_calibrate", "result")
    validate(value, "BOUNDED_REGISTRY", parents={"stage": gate["content_hash"], "protocol": protocol()["content_hash"]})
    _, b, ct = donors(spec)
    if set(value["models"]) != set(CANDIDATES) or len(value["calibration_scores"]) != len(SCALES):
        raise ValueError("Bounded registry coverage differs")
    if [r["scale"] for r in value["calibration_scores"]] != list(SCALES):
        raise ValueError("Dz grid changed")
    selected = min(value["calibration_scores"], key=lambda r: (r["score"], -r["scale"]))["scale"]
    if value["dz_scale"] != selected or value["calibration_jets"] != COUNTS["residual"]:
        raise ValueError("Dz calibration selection differs")
    for name, model in value["models"].items():
        validate_model(model, b, ct)
        if model["candidate"] != name or model["dz_scale"] != (selected if name == "B_DZ" else 1.):
            raise ValueError("Bounded registered model differs")
    return value


def selection(compare):
    if compare["stage"] != "bounded_compare":
        raise PermissionError("Confirmation requires completed comparison selection")
    value = c.product(compare, "bc_select", "result")
    reg = registry(compare)
    validate(value, "BOUNDED_SELECTION", parents={"stage": compare["content_hash"],
        "protocol": protocol()["content_hash"], "registry": reg["content_hash"]})
    winner, rejected = choose(value["scores"])
    if (value["selected"] != winner or value["guard_failures"] != rejected
            or value["selected_model"] != reg["models"][winner]
            or value["jets"] != COUNTS["evaluation"] or value["production_qualified"] is not False
            or value["confirmation_accessed"] is not False):
        raise ValueError("Frozen bounded selection differs")
    for i in range(4):
        c.verified_outputs(compare, f"bc_eval_{i}")
    return value


def reuse(original, study):
    from .b_tracking import reuse as reuse_b
    from .c_diagnostic import reuse as reuse_c
    return {"B": reuse_b(original, study), "C": reuse_c(original, study)}


def validate_stage(spec, *, source=True):
    study = load_json(checked_file(spec["study"]))
    c.validate_study(study, source=source)
    validate(spec, KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]})
    stage = spec["stage"]
    if (stage not in STAGES or spec["name"] != stage+"_r1" or spec["root"] != study["root"]
            or spec["protocol"] != protocol() or spec["tasks"] != tasks(stage)
            or study["site"]["partition"] != "debug" or spec["b_threads"] != 1
            or spec["scientific_qualification"] is not False
            or spec["resources_are_development_envelopes"] is not True):
        raise ValueError("Bounded stage registration differs")
    parent = load_json(checked_file(spec["parent_spec"]))
    if stage == "bounded_gate":
        if parent.get("stage") != "compare" or spec["reuse"] != reuse(parent, study):
            raise ValueError("Bounded donor reuse changed")
    else:
        validate_stage(parent, source=source)
        expected = STAGES[STAGES.index(stage)-1]
        if parent["stage"] != expected or parent["study"] != spec["study"] or spec["reuse"] is not None:
            raise ValueError("Bounded stage ancestry differs")
        c.verified_outputs(gate_spec(spec), "bc_acceptance")
        registry(spec)
        if stage == "bounded_confirm":
            locked = selection(parent)
            if spec["selection_hash"] != locked["content_hash"]:
                raise ValueError("Confirmation selection lock changed")
            from .bounded_data import validate_membership
            validate_membership(spec["membership"], study)
    if spec["policy"] != parent["policy"]:
        raise ValueError("Bounded association policy changed")
    if stage != "bounded_confirm" and (spec["membership"] is not None or spec["selection_hash"] is not None):
        raise PermissionError("Confirmation capability before selection")
    return study


def _publish(study, parent_ref, stage, evidence=None, membership=None, selection_hash=None):
    parent = load_json(checked_file(parent_ref))
    spec = artifact(KIND, parents={"study": study["content_hash"], "protocol": protocol()["content_hash"]},
        study=file_ref(Path(study["root"])/"study_spec.json"), root=study["root"], stage=stage,
        name=stage+"_r1", parent_spec=parent_ref, policy=parent["policy"], b_threads=1,
        protocol=protocol(), reuse=evidence, tasks=tasks(stage), scientific_qualification=False,
        resources_are_development_envelopes=True, membership=membership, selection_hash=selection_hash)
    validate_stage(spec)
    directory = c.stage_dir(spec)
    directory.mkdir(parents=True, exist_ok=False)
    c.write(spec["root"], f"stages/{spec['name']}/stage_spec.json", spec, KIND)
    c.write(spec["root"], f"stages/{spec['name']}/command_plan.json", c.command_plan(spec, study), "DEV_PLAN")
    return spec


def create(*, parent_spec, project_dir, source_commit, root):
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    donor = c.validate_stage(parent, source=False)
    for name in ("report_B", "report_C"):
        c.verified_outputs(parent, name)
    destination, old = Path(root).resolve(), Path(donor["root"]).resolve()
    if destination == old or destination.is_relative_to(old) or old.is_relative_to(destination):
        raise PermissionError("Use a new disjoint bounded study root")
    if any((p/"study_spec.json").exists() or (p/"campaign_spec.json").exists() for p in destination.parents):
        raise PermissionError("Do not nest the bounded study in another campaign")
    study = c.create_study(project_dir=project_dir, source_commit=source_commit, root=root, partition="debug",
        preparation_spec=checked_file(donor["imported"]["preparation_spec"]))
    return _publish(study, parent_ref, "bounded_gate", evidence=reuse(parent, study))


def advance(parent_spec):
    parent_ref = file_ref(parent_spec)
    parent = load_json(checked_file(parent_ref))
    study = validate_stage(parent)
    if parent["stage"] == "bounded_gate":
        acceptance = c.product(parent, "bc_acceptance", "result")
        validate(acceptance, "BOUNDED_ACCEPTANCE")
        if not acceptance["resource_envelope_ok"]:
            raise PermissionError("Measured bounded execution exceeds registered resource envelope")
        registry(parent)
        return _publish(study, parent_ref, "bounded_compare")
    if parent["stage"] == "bounded_compare":
        locked = selection(parent)
        from .bounded_data import build_membership
        return _publish(study, parent_ref, "bounded_confirm", membership=build_membership(study),
                        selection_hash=locked["content_hash"])
    raise PermissionError("Bounded version ends after confirmation; no further tuning stage")


def render(spec):
    validate_stage(spec, source=False)
    owner = {"bounded_gate": "bc_calibrate", "bounded_compare": "bc_select", "bounded_confirm": "bf_report"}[spec["stage"]]
    if not (c.stage_dir(spec)/"receipts"/(owner+".json")).is_file():
        return f"{spec['stage']}: no completed report yet. Use monitor for exact job states."
    row = c.product(spec, owner, "result")
    if spec["stage"] == "bounded_gate":
        return f"Dz scale frozen: {row['dz_scale']}; 4k calibration only. Next: separately create comparison."
    lines = [f"Stage: {spec['stage']}   Jets: {row['jets']}", "candidate        score    counts      p4  tracking     jet   joint"]
    order = ("counts_state", "particle_kinematics", "tracking", "jet_shape", "joint_jet")
    for name in CANDIDATES:
        if name not in row["scores"]:
            continue
        result = row["scores"][name]
        lines.append(f"{name:<14} {result['score']:.5f} " + " ".join(f"{result['blocks'][key]['score']:8.5f}" for key in order))
    lines.extend([f"Frozen choice: {row['selected']}", f"Status: {row.get('status', 'development_selection_only')}",
                  "Lower scores are better. Not full response qualification. Final test accessed: False."])
    if "uncertainty" in row:
        lines.append(f"Paired file-bootstrap evidence: {row['uncertainty']}")
    return "\n".join(lines)
