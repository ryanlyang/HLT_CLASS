"""Read-only reuse of completed native preparation, never of science or gates."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess

from hlt_classification.data.cache_contracts import canonical_sha256, load_json, write_immutable_json
from .contracts import artifact, validate
from .storage import checked_file, fingerprint, load_receipt, receipt_path


PREPARATION_CODE = (
    "src/hlt_classification/cms_salience_learned/data.py",
    "src/hlt_classification/cms_salience_learned/storage.py",
    "src/hlt_classification/scouting",
    "src/hlt_classification/data",
)

# Reviewed migration: only new coordinate names are added; all existing dense
# values and the preparation Git objects must remain identical. Unknown future
# ASTs do not inherit this exception.
_DENSE_COORDINATE_AST = "50f6b29e6b2af44b85a03bbbb41fa0f7e079efbf9007e3f0afb8a6713ef2049c"
_COARSE_COORDINATE_AST = "ff091af4c897832d984f45cd4b7689ad8927f3768b7359611d66031478684972"


def compatible_preparation_code(source, consumer):
    if source == consumer:
        return True
    key = "coordinate_ast_sha256"
    return (set(source) == set(consumer) == {*PREPARATION_CODE, key}
        and all(source[p] == consumer[p] for p in PREPARATION_CODE)
        and source[key] == _DENSE_COORDINATE_AST and consumer[key] == _COARSE_COORDINATE_AST)


def preparation_code(project, commit):
    """Conservative code-identity proof; no producer Python is imported."""
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Preparation requires an exact source commit")
    def git(*args):
        return subprocess.run(["git", "-C", str(project), *args], check=True,
                              text=True, encoding="utf-8", capture_output=True).stdout.strip()
    objects = {path: git("rev-parse", "--verify", f"{commit}:{path}") for path in PREPARATION_CODE}
    text = git("show", f"{commit}:src/hlt_classification/cms_salience_learned/contracts.py")
    coordinate = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "coordinate")
    objects["coordinate_ast_sha256"] = canonical_sha256(ast.dump(coordinate, include_attributes=False))
    return objects


def _source(consumer, path):
    from .campaign import validate_campaign
    path = Path(path).resolve()
    source = load_json(path)
    if source.get("preparation_import") is not None:
        raise ValueError("Chained preparation imports are not supported; name the original producer")
    validate_campaign(source)
    old = Path(source["campaign_root"]).resolve()
    new = Path(consumer["campaign_root"]).resolve()
    if path != old / "campaign_spec.json":
        raise ValueError("Preparation source spec is not canonical")
    if old == new or old.is_relative_to(new) or new.is_relative_to(old):
        raise ValueError("Preparation import roots must be disjoint")
    if consumer.get("ladder") == "coarse":
        from .contracts import graph
        if consumer["graph"] != graph("coarse"):
            raise ValueError("Unregistered coarse preparation consumer")
        for key in ("training", "matcher"):
            if source["graph"][key] != consumer["graph"][key]:
                raise ValueError("Preparation recipe differs")
    elif source["graph"] != consumer["graph"]:
        raise ValueError("Preparation science/input identity differs: graph")
    for field in ("budgets", "split_manifest", "split_roles", "data_root", "view_config_sha256"):
        if source[field] != consumer[field]:
            raise ValueError(f"Preparation science/input identity differs: {field}")
    return source


def build_import(consumer, source_path):
    from .campaign import tasks
    source = _source(consumer, source_path)
    code = preparation_code(consumer["project_dir"], source["source_commit"])
    new_code = preparation_code(consumer["project_dir"], consumer["source_commit"])
    coarse = consumer.get("ladder") == "coarse"
    if not (compatible_preparation_code(code, new_code) if coarse else code == new_code):
        raise ValueError("Preparation code changed; recompute or implement a reviewed migration")
    root = Path(source["campaign_root"])
    extra = dict(source_preparation_code=code, coordinate_migration="add_coarse_coordinates_v1") if coarse else {}
    result = artifact("PREPARATION_IMPORT", contract_version=2 if coarse else 1, source_spec=fingerprint(source_path),
        source_campaign_sha256=source["content_hash"], source_commit=source["source_commit"],
        foundation=fingerprint(root / "foundation/foundation_lock.json"), preparation_code=new_code,
        receipts={t["task_id"]: fingerprint(receipt_path(source, t["task_id"])) for t in tasks(source)["prepare"]},
        final_test_accessed=False, **extra)
    validate_import(dict(consumer, preparation_import=result), deep=True)
    return result


def validate_import(consumer, *, deep=False):
    from .campaign import tasks
    value = consumer["preparation_import"]
    validate(value, "PREPARATION_IMPORT")
    source = _source(consumer, checked_file(value["source_spec"]))
    root = Path(source["campaign_root"]).resolve()
    expected_code = preparation_code(consumer["project_dir"], consumer["source_commit"])
    source_code = preparation_code(consumer["project_dir"], source["source_commit"])
    if value["schema_version"] == 2:
        code_ok = (consumer.get("ladder") == "coarse"
            and value.get("coordinate_migration") == "add_coarse_coordinates_v1"
            and value.get("source_preparation_code") == source_code
            and compatible_preparation_code(source_code, expected_code))
    else:
        code_ok = source_code == expected_code
    if (value["source_campaign_sha256"] != source["content_hash"]
        or value["source_commit"] != source["source_commit"]
        or value["final_test_accessed"] is not False
        or value["preparation_code"] != expected_code
        or not code_ok):
        raise ValueError("Preparation import provenance differs")
    lock_path = checked_file(value["foundation"]).resolve()
    if lock_path != root / "foundation/foundation_lock.json":
        raise ValueError("Preparation foundation path differs")
    lock = load_json(lock_path)
    validate(lock, "FOUNDATION")
    if (lock["campaign_spec_sha256"] != source["content_hash"] or lock["budgets"] != source["budgets"]
        or lock["final_test_accessed"] is not False or lock["final_test_materialized"] is not False):
        raise ValueError("Imported foundation lineage differs")
    expected = {t["task_id"] for t in tasks(source)["prepare"]}
    if set(value["receipts"]) != expected:
        raise ValueError("Imported preparation receipt coverage differs")
    for task, row in value["receipts"].items():
        if checked_file(row).resolve() != receipt_path(source, task).resolve():
            raise ValueError("Imported preparation receipt path differs")
        if deep:
            receipt = load_receipt(source, task)
            if task == "foundation" and value["foundation"] not in receipt["outputs"]:
                raise ValueError("Foundation is not a receipted producer output")
    if deep:
        # Receipts check the compact NPZ payload bytes; these checks additionally
        # bind the source lock's references to the canonical producer paths.
        references = [(lock[key], root / "foundation" / filename) for key, filename in (
            ("selection", "selection.json"), ("scales", "scales.json"),
            ("validation_partition", "validation_partition.npz"))]
        count = sum(len(source["split_roles"][r]) for r in ("train", "validation"))
        if [s["index"] for s in lock["sources"]] != list(range(count)):
            raise ValueError("Imported foundation source coverage differs")
        for row in lock["sources"]:
            for kind in ("assignment", "coupling"):
                references.append((row[kind], root / "foundation" / f"{kind}_{row['index']:04d}.json"))
        for row, expected_path in references:
            if checked_file(row).resolve() != expected_path:
                raise ValueError("Imported foundation payload path differs")
    return source


def publish_import(spec):
    validate_import(spec, deep=True)
    path = Path(spec["campaign_root"]) / "foundation/preparation_import.json"
    write_immutable_json(path, spec["preparation_import"])
    return [path]


def preparation_spec(spec):
    if spec.get("preparation_import") is None:
        return spec
    source = validate_import(spec)
    load_receipt(spec, "foundation")
    imported = load_json(Path(spec["campaign_root"]) / "foundation/preparation_import.json")
    if imported != spec["preparation_import"]:
        raise ValueError("Published preparation import differs from the campaign")
    return source
