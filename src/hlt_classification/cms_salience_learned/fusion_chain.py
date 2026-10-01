"""Adjacent paired-teacher KD, with direct and same-view bridge endpoints.

Only graph, immutable import and reporting semantics live here. Production
model, input, optimization and loss kernels are reused without modification.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from .contracts import COARSE_RUNG_ORDER, artifact, graph, node, validate
from .storage import checked_file, fingerprint, load_receipt, receipt_path

ACQUISITION_TASKS = ("train_ACQUIRE_U050", "reduce_ACQUIRE_U050")
ENDPOINTS = ("FINAL_D000_DIRECT", "FINAL_D000_BRIDGE")


def chain_graph():
    # Preserve the imported references and first acquisition's exact node/seed
    # identities. Later teacher edges are intentionally NOT withdrawal carriers.
    donor = graph("coarse")
    nodes = deepcopy(donor["nodes"][:4])
    nodes.append(deepcopy(next(n for n in donor["nodes"] if n["node_id"] == "ACQUIRE_U050")))
    tasks = deepcopy(donor["tasks"][:7])

    def fit(n, *, reduce=False):
        nodes.append(n)
        name = n["node_id"]
        tasks.append(dict(task_id="train_" + name, kind="train",
            dependencies=["reduce_" + n["teacher_distribution"]], model=name))
        if reduce:
            tasks.append(dict(task_id="reduce_" + name, kind="reduce",
                dependencies=["train_" + name], model=name))

    teacher = "ACQUIRE_U050"
    for higher, lower in zip(COARSE_RUNG_ORDER[1:], COARSE_RUNG_ORDER[2:]):
        name = "FUSION_" + lower
        fit(node(name, "fusion_pair_kd", lower, higher, teacher, alias=lower), reduce=True)
        teacher = name
    fit(node(ENDPOINTS[0], "direct_kd", "D000", teacher=teacher, alias="D000"))
    fit(node("FUSION_D000_D000", "fusion_pair_kd", "D000", "D000", teacher,
             alias="D000_D000"), reduce=True)
    fit(node(ENDPOINTS[1], "direct_kd", "D000", teacher="FUSION_D000_D000", alias="D000"))
    tasks.append(dict(task_id="aggregate", kind="aggregate", dependencies=[t["task_id"] for t in tasks], model=None))
    tasks.append(dict(task_id="complete", kind="complete", dependencies=["aggregate"], model=None))
    return artifact("GRAPH", contract_version=4, nodes=nodes, tasks=tasks,
        rung_order=list(COARSE_RUNG_ORDER), training=deepcopy(donor["training"]),
        budgets=deepcopy(donor["budgets"]), matcher=deepcopy(donor["matcher"]),
        fit_count=len(nodes), extraction_count=0,
        reducer_count=sum(t["kind"] == "reduce" for t in tasks),
        endpoint_order=list(ENDPOINTS), fresh_fit_count=7, final_test_accessed=False)


def _source(consumer, path):
    from .campaign import command_plan, gate_check, validate_campaign
    from .coarse_submission import _command
    from .shared_import import reference_code

    path = Path(path).resolve()
    source = load_json(path)
    validate_campaign(source)
    if (source["schema_version"] != 5 or source.get("ladder") != "coarse"
        or path != Path(source["campaign_root"]) / "campaign_spec.json"):
        raise ValueError("First fusion donor must be the canonical accepted coarse-v5 campaign")
    if consumer.get("ladder") != "fusion_chain" or consumer["schema_version"] != 7:
        raise ValueError("Acquisition reuse requires the registered fusion-chain consumer")
    old, new = path.parent, Path(consumer["campaign_root"]).resolve()
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise ValueError("Acquisition donor and consumer roots overlap")
    for key in ("budgets", "split_manifest", "split_roles", "data_root", "view_config_sha256", "site", "resources"):
        if source[key] != consumer[key]:
            raise ValueError(f"Acquisition source identity differs: {key}")
    if (source["graph"]["training"] != consumer["graph"]["training"]
        or source["preparation_import"]["source_spec"] != consumer["preparation_import"]["source_spec"]
        or source["shared_source"]["source_spec"] != consumer["shared_source"]["source_spec"]):
        raise ValueError("Acquisition optimization, preparation or reference source differs")
    find = lambda spec: next(n for n in spec["graph"]["nodes"] if n["node_id"] == "ACQUIRE_U050")
    if find(source) != find(consumer):
        raise ValueError("Acquisition view, seed or teacher differs")
    gate_check(source)
    if reference_code(consumer["project_dir"], consumer["source_commit"]) != reference_code(consumer["project_dir"], source["source_commit"]):
        raise ValueError("Acquisition scientific kernels changed")

    ledger = load_json(old / "science_submission_ledger.json")
    validate_submission_ledger(ledger)
    rows = command_plan(source, "science")["commands"]
    if (ledger["dry_run"] is not False or ledger["campaign_spec_sha256"] != source["content_hash"]
        or set(ledger["jobs"]) != {r["task_id"] for r in rows}
        or set(ledger["commands"]) != set(ledger["jobs"])
        or len(set(ledger["jobs"].values())) != len(rows)
        or any(re.fullmatch(r"[1-9][0-9]*", str(j)) is None for j in ledger["jobs"].values())):
        raise ValueError("Acquisition source ledger is not the exact live coarse DAG")
    for row in rows:
        external = source["shared_source"]["jobs"].get(row["task_id"])
        possible = [_command(row, ledger["jobs"], external), _command(row, ledger["jobs"], None)]
        if ledger["commands"][row["task_id"]] not in possible:
            raise ValueError("Acquisition source submitted command differs")
    return source


def acquisition_outputs(source, task):
    """Authenticate the two completed first-rung artifacts, never other jobs."""
    from .preparation_import import preparation_spec
    from .production import registered_node
    if task not in ACQUISITION_TASKS:
        raise ValueError("Only the first completed acquisition and reducer may be imported")
    base = Path(source["campaign_root"])
    selected_path = base / "training/ACQUIRE_U050/report.json"
    selected = load_json(selected_path)
    validate(selected, "MODEL_REPORT")
    n = registered_node(source, "ACQUIRE_U050")
    foundation = load_json(Path(preparation_spec(source)["campaign_root"]) / "foundation/foundation_lock.json")
    fit = selected.get("training", {})
    validate(fit, "TRAINING_REPORT")
    if (selected["campaign_spec_sha256"] != source["content_hash"] or selected["node"] != n
        or selected.get("selection_role") != "validation_checkpoint"
        or selected.get("reporting_role") != "validation_report"
        or selected["final_test_accessed"] is not False or fit["final_test_accessed"] is not False
        or fit["node"] != n or fit["foundation_sha256"] != foundation["content_hash"]
        or fit["scientific_fit"] is not True or fit["selected_weights_restored"] is not True):
        raise ValueError("Acquisition model/report identity or scientific-fit status differs")
    checkpoint = checked_file(selected["checkpoint"])
    if checkpoint != base / "training/ACQUIRE_U050/selected_model.pt":
        raise ValueError("Acquisition checkpoint is not canonical")
    parent_path = base / "probabilities/U000/bank.json"
    load_receipt(source, "reduce_U000")
    parent = load_json(parent_path)
    validate(parent, "BANK")
    if (parent["campaign_spec_sha256"] != source["content_hash"] or parent["model"] != "U000"
        or parent["foundation_sha256"] != foundation["content_hash"]
        or parent["train_temperature"] != 2. or parent["validation_temperature"] != 1.
        or parent["final_test_accessed"] is not False
        or selected["parents"] != {"teacher_bank": parent["content_hash"]}):
        raise ValueError("Acquisition did not cold-train from the registered U000 bank")
    if task.startswith("train_"):
        paths, payload = [checkpoint, selected_path], selected
    else:
        acquisition_outputs(source, ACQUISITION_TASKS[0])
        bank_path = base / "probabilities/ACQUIRE_U050/bank.json"
        payload = load_json(bank_path)
        validate(payload, "BANK")
        if (payload["campaign_spec_sha256"] != source["content_hash"] or payload["model"] != "ACQUIRE_U050"
            or payload["model_report_sha256"] != selected["content_hash"]
            or payload["foundation_sha256"] != foundation["content_hash"]
            or payload["train_temperature"] != 2. or payload["validation_temperature"] != 1.
            or payload["final_test_accessed"] is not False):
            raise ValueError("Acquisition bank/checkpoint/foundation lineage differs")
        paths = [base / "probabilities/ACQUIRE_U050" / f"{r}.npz" for r in ("train", "validation")]
        for role, path in zip(("train", "validation"), paths):
            if checked_file(payload[role]) != path:
                raise ValueError("Acquisition bank payload is not canonical")
        paths.append(bank_path)
    if load_receipt(source, task)["outputs"] != [fingerprint(p) for p in paths]:
        raise ValueError("Acquisition task receipt output coverage differs")
    return payload


def build_acquisition_source(spec, path):
    source = _source(spec, path)
    base = Path(source["campaign_root"])
    outputs = {task: acquisition_outputs(source, task) for task in ACQUISITION_TASKS}
    return artifact("ACQUISITION_SOURCE", source_spec=fingerprint(path),
        source_commit=source["source_commit"], source_campaign_sha256=source["content_hash"],
        ledger=fingerprint(base / "science_submission_ledger.json"),
        receipts={task: fingerprint(receipt_path(source, task)) for task in ACQUISITION_TASKS},
        artifact_hashes={task: value["content_hash"] for task, value in outputs.items()},
        completed_only=True, final_test_accessed=False)


def validate_acquisition_source(spec):
    value = spec["acquisition_source"]
    validate(value, "ACQUISITION_SOURCE")
    path = checked_file(value["source_spec"])
    if value != build_acquisition_source(spec, path):
        raise ValueError("Acquisition source changed after registration")
    return load_json(path)


def import_acquisition(spec, task):
    source = validate_acquisition_source(spec)
    original = acquisition_outputs(source, task)
    base = Path(spec["campaign_root"])
    fields = {k: v for k, v in original.items() if k not in {"content_hash", "contract", "schema_version"}}
    fields.update(campaign_spec_sha256=spec["content_hash"], imported_from=dict(
        source_campaign_sha256=source["content_hash"], source_commit=source["source_commit"],
        task=task, artifact_sha256=original["content_hash"], source_receipt=fingerprint(receipt_path(source, task))))
    paths = []
    if task.startswith("train_"):
        target = base / "training/ACQUIRE_U050/selected_model.pt"
        atomic_publish_bytes(target, checked_file(original["checkpoint"]).read_bytes())
        fields["checkpoint"] = fingerprint(target)
        paths.append(target)
        path, kind = base / "training/ACQUIRE_U050/report.json", "MODEL_REPORT"
    else:
        load_receipt(spec, ACQUISITION_TASKS[0])
        selected = load_json(base / "training/ACQUIRE_U050/report.json")
        validate(selected, "MODEL_REPORT")
        if selected["imported_from"]["artifact_sha256"] != original["model_report_sha256"]:
            raise ValueError("Imported acquisition bank/model join differs")
        fields["model_report_sha256"] = selected["content_hash"]
        for role in ("train", "validation"):
            target = base / "probabilities/ACQUIRE_U050" / f"{role}.npz"
            atomic_publish_bytes(target, checked_file(original[role]).read_bytes())
            fields[role] = fingerprint(target)
            paths.append(target)
        path, kind = base / "probabilities/ACQUIRE_U050/bank.json", "BANK"
    write_immutable_json(path, artifact(kind, **fields))
    return paths + [path]


def result_rows(spec, *, require_complete=False):
    from .production import metric_recovery, registered_node
    reports = {}
    for n in spec["graph"]["nodes"]:
        name = n["node_id"]
        if not receipt_path(spec, "train_" + name).exists() and not require_complete:
            continue
        load_receipt(spec, "train_" + name)
        report = load_json(Path(spec["campaign_root"]) / "training" / name / "report.json")
        validate(report, "MODEL_REPORT")
        if (report["campaign_spec_sha256"] != spec["content_hash"] or report["node"] != registered_node(spec, name)
            or report.get("reporting_role") != "validation_report" or report["final_test_accessed"] is not False):
            raise ValueError("Fusion-chain report identity differs")
        reports[name] = report
    rows = []
    for name, report in reports.items():
        n = report["node"]
        hlt = n["primary_coordinate"] == "D000" and n["context_coordinate"] in (None, "D000")
        rows.append(dict(model=name, report_sha256=report["content_hash"], metrics=report["report_metrics"],
            input_role="hlt_only" if hlt else "privileged", branches=2 if n["context_coordinate"] else 1,
            selected_pass=report["training"]["selected_pass"], passes=report["training"]["passes"],
            recovery_to_offline=(metric_recovery(report["report_metrics"], reports["M0HLT"]["report_metrics"],
                reports["OFFLINE"]["report_metrics"]) if {"M0HLT", "OFFLINE"} <= reports.keys() else None)))
    return rows


def aggregate_chain(spec):
    rows = result_rows(spec, require_complete=True)
    metrics = {row["model"]: row["metrics"] for row in rows}
    contrasts = []
    for left, right in ((ENDPOINTS[0], "DIRECT_D000"), (ENDPOINTS[1], "DIRECT_D000"), (ENDPOINTS[1], ENDPOINTS[0])):
        r = lambda m: None if m["macro_mean_log_qcd_rejection_at_50pct_signal"] is None else math.exp(m["macro_mean_log_qcd_rejection_at_50pct_signal"])
        a, b = metrics[left], metrics[right]
        contrasts.append(dict(left=left, right=right, accuracy=a["accuracy"]-b["accuracy"],
            macro_ovr_auc=a["macro_ovr_auc"]-b["macro_ovr_auc"],
            macro_r50_linear=None if r(a) is None or r(b) is None else r(a)-r(b)))
    path = Path(spec["campaign_root"]) / "aggregate.json"
    write_immutable_json(path, artifact("FUSION_CHAIN_AGGREGATE", campaign_spec_sha256=spec["content_hash"],
        rows=rows, contrasts=contrasts, fresh_fit_count=7, bridge_extra_fit_count=1,
        reporting_role="validation_report", final_test_accessed=False, poor_metrics_do_not_control_graph=True))
    return [path]


def print_results(spec):
    from .campaign import validate_campaign
    validate_campaign(spec)
    if spec.get("ladder") != "fusion_chain":
        raise ValueError("Use a fusion-chain campaign for these results")
    rows = {r["model"]: r for r in result_rows(spec)}
    print(f"Campaign: {spec['campaign_root']}")
    print("All metrics: validation REPORT subset. Final test untouched.")
    print("Recovery: M0HLT=0%, pure OFFLINE=100%; linear R50 recovery.")
    print("P=privileged; H2=HLT-only two-branch; H1=HLT-only single ParT.")
    print(f"{'model':<24} {'view':>4} {'pick/done':>10} {'accuracy':>10} {'AUC':>10} {'R50':>10} {'AUC rec.':>10} {'R50 rec.':>10}")
    number = lambda x, places=6: "n/a" if x is None else f"{x:.{places}f}"
    percent = lambda x: "n/a" if x is None else f"{100*x:+.1f}%"
    for n in spec["graph"]["nodes"]:
        name = n["node_id"]
        if name not in rows:
            print(f"{name:<24} NO REPORT")
            continue
        row = rows[name]
        m, rec = row["metrics"], row["recovery_to_offline"] or {}
        r50 = m["macro_mean_log_qcd_rejection_at_50pct_signal"]
        role = "P" if row["input_role"] == "privileged" else f"H{row['branches']}"
        pick = f"{row['selected_pass']}/{row['passes']}"
        print(f"{name:<24} {role:>4} {pick:>10} {number(m['accuracy']):>10} {number(m['macro_ovr_auc']):>10} "
              f"{number(None if r50 is None else math.exp(r50), 1):>10} "
              f"{percent(rec.get('macro_ovr_auc')):>10} {percent(rec.get('macro_r50_linear')):>10}")
    for left, right in ((ENDPOINTS[0], "DIRECT_D000"), (ENDPOINTS[1], "DIRECT_D000"), (ENDPOINTS[1], ENDPOINTS[0])):
        if left in rows and right in rows:
            a, b = rows[left]["metrics"], rows[right]["metrics"]
            key = "macro_mean_log_qcd_rejection_at_50pct_signal"
            dr = "n/a" if a[key] is None or b[key] is None else f"{math.exp(a[key])-math.exp(b[key]):+.1f}"
            print(f"{left} minus {right}: dAccuracy={a['accuracy']-b['accuracy']:+.6f} "
                  f"dAUC={a['macro_ovr_auc']-b['macro_ovr_auc']:+.8f} dR50={dr}")
    print("\nPER-CLASS QCD REJECTION AT 50% SIGNAL EFFICIENCY (REPORT)")
    for name, row in rows.items():
        recovery = (row["recovery_to_offline"] or {}).get("per_class_r50", {})
        values = [f"{label}={number(m['qcd_rejection']['50pct']['rejection'], 1)} ({percent(recovery.get(label))})"
                  for label, m in row["metrics"]["per_class"].items() if "qcd_rejection" in m]
        print(f"{name}: " + "; ".join(values))
    print("Bridge endpoint uses one additional fit; not compute matched. No endpoint is selected on these report rows.")
