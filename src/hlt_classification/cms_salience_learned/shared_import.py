"""Authenticated reuse of only the dense campaign's common reference tasks."""
from __future__ import annotations

from pathlib import Path
import ast
import re
import subprocess

from hlt_classification.data.cache_contracts import atomic_publish_bytes, canonical_sha256, load_json, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
from .contracts import SHARED_TASKS, artifact, graph, validate
from .storage import checked_file, fingerprint, load_receipt


REFERENCE_CODE = (
    "src/hlt_classification/cms_salience_learned/training.py",
    "src/hlt_classification/cms_salience_learned/model.py",
    "src/hlt_classification/cms_salience_learned/data.py",
    "src/hlt_classification/cms_salience_learned/storage.py",
    "src/hlt_classification/models", "src/hlt_classification/scouting", "src/hlt_classification/data",
)


def reference_code(project, commit):
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Exact shared-reference commit required")
    code = {p: subprocess.run(["git", "-C", str(project), "rev-parse", "--verify", f"{commit}:{p}"],
        check=True, capture_output=True, text=True).stdout.strip() for p in REFERENCE_CODE}
    # Orchestration changes are allowed, scientific reference kernels are not.
    for module, names in {
        "contracts": ("node", "learning_rate", "alpha_for_pass"),
        "production": ("root", "build_cache", "registered_node", "model_task", "load_model",
                       "publish_model", "partitions", "load_bank", "run_fit", "reduce_model"),
    }.items():
        path = f"src/hlt_classification/cms_salience_learned/{module}.py"
        source = subprocess.run(["git", "-C", str(project), "show", f"{commit}:{path}"],
            check=True, capture_output=True, text=True, encoding="utf-8").stdout
        functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        for name in names:
            code[f"{module}.{name}:ast"] = canonical_sha256(ast.dump(functions[name], include_attributes=False))
    return code


def dense_ledger(source):
    from .campaign import command_plan
    path = Path(source["campaign_root"]) / "science_submission_ledger.json"
    ledger = load_json(path)
    validate_submission_ledger(ledger)
    rows = command_plan(source, "science")["commands"]
    if (ledger["dry_run"] is not False or ledger["campaign_spec_sha256"] != source["content_hash"]
        or set(ledger["jobs"]) != {r["task_id"] for r in rows}
        or len(set(ledger["jobs"].values())) != len(rows)
        or any(re.fullmatch(r"[1-9][0-9]*", str(j)) is None for j in ledger["jobs"].values())
        or ledger["commands"] != {r["task_id"]: _resolved(r, ledger["jobs"]) for r in rows}):
        raise ValueError("Shared-source live ledger differs from the dense plan")
    return ledger


def _compatible_source(consumer, source_path):
    from .campaign import gate_check, validate_campaign
    from .preparation_import import preparation_spec
    source_path = Path(source_path).resolve()
    source = load_json(source_path)
    validate_campaign(source)
    if source.get("shared_source") is not None or source["graph"] != graph():
        raise ValueError("Shared source must be an original dense scientific campaign")
    if source_path != Path(source["campaign_root"]) / "campaign_spec.json":
        raise ValueError("Shared source path is not canonical")
    old, new = source_path.parent, Path(consumer["campaign_root"]).resolve()
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise ValueError("Shared source and consumer roots overlap")
    if consumer.get("ladder") != "coarse" or consumer.get("preparation_import") is None:
        raise ValueError("Shared coarse references require the same imported preparation")
    producer = preparation_spec(source)
    if checked_file(consumer["preparation_import"]["source_spec"]).resolve() != Path(producer["campaign_root"]) / "campaign_spec.json":
        raise ValueError("Shared reference preparation identity differs")
    for field in ("budgets", "split_manifest", "split_roles", "data_root", "view_config_sha256", "site", "resources"):
        if consumer[field] != source[field]:
            raise ValueError(f"Shared reference identity differs: {field}")
    if consumer["graph"]["training"] != source["graph"]["training"]:
        raise ValueError("Shared optimization differs")
    for name in ("M0HLT", "OFFLINE", "U000", "DIRECT_D000"):
        find = lambda spec: next(n for n in spec["graph"]["nodes"] if n["node_id"] == name)
        if find(consumer) != find(source):
            raise ValueError("Shared node/seed identity differs")
    gate_check(source)
    if reference_code(consumer["project_dir"], consumer["source_commit"]) != reference_code(consumer["project_dir"], source["source_commit"]):
        raise ValueError("Shared model/training implementation changed")
    return source


def build_shared_source(consumer, source_path):
    source = _compatible_source(consumer, source_path)
    ledger = dense_ledger(source)
    return artifact("SHARED_SOURCE", source_spec=fingerprint(source_path),
        source_campaign_sha256=source["content_hash"], source_commit=source["source_commit"],
        ledger=fingerprint(Path(source["campaign_root"]) / "science_submission_ledger.json"),
        jobs={task: ledger["jobs"][task] for task in SHARED_TASKS},
        reference_code=reference_code(consumer["project_dir"], consumer["source_commit"]), final_test_accessed=False)


def validate_shared_source(consumer):
    value = consumer["shared_source"]
    validate(value, "SHARED_SOURCE")
    source = _compatible_source(consumer, checked_file(value["source_spec"]))
    ledger_path = checked_file(value["ledger"])
    if ledger_path.resolve() != Path(source["campaign_root"]) / "science_submission_ledger.json":
        raise ValueError("Shared ledger path differs")
    ledger = dense_ledger(source)
    if (value["source_campaign_sha256"] != source["content_hash"] or value["source_commit"] != source["source_commit"]
        or value["jobs"] != {task: ledger["jobs"][task] for task in SHARED_TASKS}
        or value["reference_code"] != reference_code(consumer["project_dir"], consumer["source_commit"])
        or value["final_test_accessed"] is not False):
        raise ValueError("Shared-source provenance differs")
    return source


def source_task_outputs(source, task):
    """Fail closed on a receipt with missing, corrupt or cross-task outputs."""
    if task not in SHARED_TASKS:
        raise ValueError("Only common reference tasks may be imported")
    from .production import registered_node
    base = Path(source["campaign_root"])
    receipt = load_receipt(source, task)
    name = task.removeprefix("train_").removeprefix("reduce_")
    report_path = base / "training" / name / "report.json"
    report = load_json(report_path)
    validate(report, "MODEL_REPORT")
    if (report["campaign_spec_sha256"] != source["content_hash"] or report["node"] != registered_node(source, name)
        or report["final_test_accessed"] is not False):
        raise ValueError("Shared model report identity differs")
    checkpoint = checked_file(report["checkpoint"])
    if checkpoint.resolve() != base / "training" / name / "selected_model.pt":
        raise ValueError("Shared checkpoint path differs")
    if task.startswith("train_"):
        expected = [checkpoint, report_path]
        payload = report
    else:
        source_task_outputs(source, "train_U000")
        bank_path = base / "probabilities" / name / "bank.json"
        payload = load_json(bank_path)
        validate(payload, "BANK")
        from .preparation_import import preparation_spec
        foundation = load_json(Path(preparation_spec(source)["campaign_root"]) / "foundation/foundation_lock.json")
        if (payload["campaign_spec_sha256"] != source["content_hash"] or payload["model"] != name
            or payload["model_report_sha256"] != report["content_hash"]
            or payload["foundation_sha256"] != foundation["content_hash"]
            or payload["train_temperature"] != 2. or payload["validation_temperature"] != 1.
            or payload["final_test_accessed"] is not False):
            raise ValueError("Shared bank lineage differs")
        expected = [base / "probabilities" / name / f"{r}.npz" for r in ("train", "validation")]
        for role, path in zip(("train", "validation"), expected):
            if checked_file(payload[role]).resolve() != path:
                raise ValueError("Shared probability path differs")
        expected.append(bank_path)
    if receipt["outputs"] != [fingerprint(p) for p in expected]:
        raise ValueError("Shared receipt output coverage differs")
    return payload


def import_shared_task(spec, task):
    source = validate_shared_source(spec)
    original = source_task_outputs(source, task)
    base = Path(spec["campaign_root"])
    fields = {k: v for k, v in original.items() if k not in {"content_hash", "contract", "schema_version"}}
    fields.update(campaign_spec_sha256=spec["content_hash"], imported_from=dict(
        source_campaign_sha256=source["content_hash"], task=task, artifact_sha256=original["content_hash"],
        source_receipt=fingerprint(Path(source["campaign_root"]) / "receipts" / f"{task}.json")))
    outputs = []
    name = task.removeprefix("train_").removeprefix("reduce_")
    if task.startswith("train_"):
        target = base / "training" / name / "selected_model.pt"
        atomic_publish_bytes(target, checked_file(original["checkpoint"]).read_bytes())
        fields["checkpoint"] = fingerprint(target)
        outputs.append(target)
        path, kind = base / "training" / name / "report.json", "MODEL_REPORT"
    else:
        load_receipt(spec, "train_U000")
        selected = load_json(base / "training/U000/report.json")
        validate(selected, "MODEL_REPORT")
        if selected["imported_from"]["artifact_sha256"] != original["model_report_sha256"]:
            raise ValueError("Imported U000 bank/checkpoint join differs")
        fields["model_report_sha256"] = selected["content_hash"]
        for role in ("train", "validation"):
            target = base / "probabilities" / name / f"{role}.npz"
            atomic_publish_bytes(target, checked_file(original[role]).read_bytes())
            fields[role] = fingerprint(target)
            outputs.append(target)
        path, kind = base / "probabilities" / name / "bank.json", "BANK"
    write_immutable_json(path, artifact(kind, **fields))
    return outputs + [path]
