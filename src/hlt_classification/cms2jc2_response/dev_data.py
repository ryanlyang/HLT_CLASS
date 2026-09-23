"""Frozen, file-disjoint development samples inside authenticated response_fit."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import hashlib

import awkward as ak
import numpy as np

from .audit import cms_particle_branches, latest_tree, validate_inventory
from .bridge import from_cms
from .contracts import artifact, load_json, safe_relative, sha256_file, validate
from .readers import Pair, authenticated_open
from .splits import _apportion, pack_entries, unpack_entries, validate_roles

COUNTS = dict(location=16000, residual=4000, evaluation=10000, pilot=2000)
EVAL_SHARDS = 4
IMPORT_FILES = ("cms_inventory.json", "response_roles.json")
SEMANTIC_IMPORTS = (
    "src/hlt_classification/cms2jc2_response/audit.py",
    "src/hlt_classification/cms2jc2_response/splits.py",
    "src/hlt_classification/scouting/schema.py",
    "src/hlt_classification/scouting/labels.py",
    "src/hlt_classification/scouting/splits.py",
)


def file_ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path))


def checked_file(row):
    path = Path(row["path"]).resolve(strict=True)
    if not path.is_file() or sha256_file(path) != row["sha256"]:
        raise ValueError("Development input bytes changed")
    return path


def import_preparation(preparation_spec: Path, project: Path) -> dict:
    """Reuse CMS metadata only, not the donor's old JC2 inputs/acceptance."""
    spec = load_json(preparation_spec)
    validate(spec, "PREPARATION_SPEC")
    validate(spec["source"], "SOURCE")
    if spec["parents"] != {"source": spec["source"]["content_hash"]}:
        raise ValueError("Preparation source lineage differs")
    if not spec["source"]["executable"] or spec["source"]["dirty"]:
        raise PermissionError("Development import needs production preparation provenance")
    root = Path(spec["campaign_root"]).resolve(strict=True)
    if preparation_spec.resolve() != root / "preparation_spec.json":
        raise ValueError("Preparation location differs")
    report_path = root / "preparation_report.json"
    report = load_json(report_path)
    validate(report, "PREPARATION_REPORT", parents={"spec": spec["content_hash"]})
    if report.get("remote_cpu_metadata_job_completed") is not True:
        raise PermissionError("Preparation did not complete on RC")
    outputs = {str(Path(r["path"]).resolve()): r for r in report["outputs"]}
    refs = {}
    for name in IMPORT_FILES:
        path = root / name
        if str(path) not in outputs:
            raise ValueError("Missing authenticated CMS preparation product")
        checked_file(outputs[str(path)])
        refs[name] = file_ref(path)
    for name in SEMANTIC_IMPORTS:
        if sha256_file(project / name) != spec["source"]["files"].get(name):
            raise ValueError(f"CMS preparation semantics changed: {name}")
    inventory, roles = (load_json(checked_file(refs[n])) for n in IMPORT_FILES)
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    return artifact("DEV_IMPORT", parents={"preparation": spec["content_hash"],
                    "receipt": report["content_hash"]},
                    preparation_spec=file_ref(preparation_spec), receipt=file_ref(report_path),
                    files=refs, cms_root=spec["cms_root"], donor_root=str(root),
                    donor_commit=spec["source"]["commit"],
                    semantic_files={n: spec["source"]["files"][n] for n in SEMANTIC_IMPORTS},
                    old_jc2_inputs_imported=False, acceptance_imported=False)


def validate_import(value, project):
    validate(value, "DEV_IMPORT")
    if import_preparation(checked_file(value["preparation_spec"]), project) != value:
        raise ValueError("Development CMS import changed")


def _hash(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).digest()


def _sample(files, total, domain, source_capacities):
    files = sorted(files, key=lambda f: f["path"])
    # Hold the original response_fit source mixture fixed across all roles.
    # File availability must not turn evaluation into a different class mixture.
    sources = sorted(source_capacities)
    source_counts = _apportion(total, [source_capacities[s] for s in sources])
    by_path = {}
    for source, count in zip(sources, source_counts):
        rows = [f for f in files if f["source"] == source]
        by_path.update({f["path"]: n for f, n in zip(rows, _apportion(count, [f["selected_entries"] for f in rows]))})
    counts = [by_path[f["path"]] for f in files]
    rows = []
    for f, count in zip(files, counts):
        if count == 0:
            raise ValueError("Development budget does not cover every selected source file")
        eligible = unpack_entries(f["entry_mask"], f["raw_entries"])
        order = sorted(map(int, eligible), key=lambda i: (_hash(domain, f["sha256"], f["tree_key"], i), i))
        rows.append(dict(path=f["path"], selected_entries=count,
                         entry_mask=pack_entries(sorted(order[:count]), f["raw_entries"])))
    return rows


def build_samples(inventory, roles):
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    eligible = [r for r in roles["files"] if r["response_role"] == "response_fit"]
    residual = [r for r in eligible if r["fit_role"] == "fit_residual"]
    location_pool = [r for r in eligible if r["fit_role"] == "fit_location"]
    evaluation = []
    sources = sorted({r["source"] for r in eligible})
    source_capacities = {s: sum(r["selected_entries"] for r in eligible if r["source"] == s) for s in sources}
    for source in sources:
        rows = sorted((r for r in location_pool if r["source"] == source),
                      key=lambda r: (_hash("CMS2JC2_DEV_FILES/v1", r["sha256"]), r["path"]))
        if len(rows) < 2:
            raise ValueError("Need separate evaluation/location files per source; no held-out borrowing")
        evaluation.extend(rows[:min(2, len(rows)-1)])
    eval_paths = {r["path"] for r in evaluation}
    location = [r for r in location_pool if r["path"] not in eval_paths]
    if any({r["source"] for r in rows} != set(sources) for rows in (location, residual, evaluation)):
        raise ValueError("Development source-category coverage differs")
    members = {role: _sample(files, COUNTS[role], "CMS2JC2_DEV_ROWS/v1", source_capacities)
               for role, files in (("location", location), ("residual", residual), ("evaluation", evaluation))}
    by_path = {r["path"]: r for r in eligible}
    pilot_files = [{**by_path[r["path"]], **r} for r in members["location"]]
    members["pilot"] = _sample(pilot_files, COUNTS["pilot"], "CMS2JC2_DEV_PILOT/v1", source_capacities)
    # Global hash ordering fixes exact 2500-row shards independently of file size.
    entries = [(r["path"], int(i)) for r in members["evaluation"]
               for i in unpack_entries(r["entry_mask"], by_path[r["path"]]["raw_entries"])]
    entries.sort(key=lambda row: (_hash("CMS2JC2_DEV_EVAL_SHARD/v1", by_path[row[0]]["sha256"], row[1]), row))
    shards = []
    for shard in range(EVAL_SHARDS):
        selected = entries[shard::EVAL_SHARDS]
        shard_rows = []
        for path in sorted({p for p, _ in selected}):
            ids = sorted(i for p, i in selected if p == path)
            shard_rows.append(dict(path=path, selected_entries=len(ids),
                                   entry_mask=pack_entries(ids, by_path[path]["raw_entries"])))
        shards.append(shard_rows)
    return artifact("DEV_SAMPLES", parents={"inventory": inventory["content_hash"], "roles": roles["content_hash"]},
                    counts=COUNTS, members=members, evaluation_shards=shards,
                    policy="file_disjoint_fit_only_source_stratified_bottom_hash_v1",
                    source_capacities=source_capacities,
                    development_only=True, selection_accessed=False, confirmation_accessed=False)


def validate_samples(value, inventory, roles, *, canonical=False):
    validate(value, "DEV_SAMPLES", parents={"inventory": inventory["content_hash"], "roles": roles["content_hash"]})
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    if (value["counts"] != COUNTS or set(value["members"]) != set(COUNTS)
            or value["development_only"] is not True or value["selection_accessed"] is not False
            or value["confirmation_accessed"] is not False
            or value["policy"] != "file_disjoint_fit_only_source_stratified_bottom_hash_v1"):
        raise ValueError("Development population registry differs")
    files = {r["path"]: r for r in roles["files"] if r["response_role"] == "response_fit"}
    sources = sorted({f["source"] for f in files.values()})
    capacities = {s: sum(f["selected_entries"] for f in files.values() if f["source"] == s) for s in sources}
    if value["source_capacities"] != capacities:
        raise ValueError("Development source mixture changed")
    selected = {}
    for role, rows in value["members"].items():
        if len({r["path"] for r in rows}) != len(rows):
            raise ValueError("Duplicate sample file")
        identities = set()
        for r in rows:
            if r["path"] not in files:
                raise PermissionError("Development mask escapes response_fit")
            f = files[r["path"]]
            if f["fit_role"] != ("fit_residual" if role == "residual" else "fit_location"):
                raise PermissionError("Development role crosses original inner-file boundary")
            entries = unpack_entries(r["entry_mask"], f["raw_entries"])
            allowed = unpack_entries(f["entry_mask"], f["raw_entries"])
            if len(entries) != r["selected_entries"] or not np.isin(entries, allowed).all():
                raise ValueError("Development mask escapes eligible rows")
            identities.update((r["path"], int(i)) for i in entries)
        if len(identities) != COUNTS[role]:
            raise ValueError("Development count differs")
        expected = _apportion(COUNTS[role], [capacities[s] for s in sources])
        if [sum(files[p]["source"] == s for p, _ in identities) for s in sources] != expected:
            raise ValueError("Development source mixture differs between roles")
        selected[role] = identities
    if not selected["pilot"] <= selected["location"]:
        raise ValueError("Pilot is not nested")
    paths = {k: {p for p, _ in v} for k, v in selected.items()}
    if paths["evaluation"] & (paths["location"] | paths["residual"]) or paths["location"] & paths["residual"]:
        raise PermissionError("Development roles share a source file")
    covered = set()
    if len(value["evaluation_shards"]) != EVAL_SHARDS:
        raise ValueError("Evaluation shard registry differs")
    for rows in value["evaluation_shards"]:
        if len({r["path"] for r in rows}) != len(rows):
            raise ValueError("Duplicate evaluation shard file")
        ids = set()
        for r in rows:
            if r["path"] not in paths["evaluation"]:
                raise PermissionError("Evaluation shard escapes evaluation files")
            entries = unpack_entries(r["entry_mask"], files[r["path"]]["raw_entries"])
            if len(entries) != r["selected_entries"]:
                raise ValueError("Evaluation shard count differs")
            ids.update((r["path"], int(i)) for i in entries)
        if covered & ids or len(ids) != COUNTS["evaluation"] // EVAL_SHARDS:
            raise ValueError("Evaluation shard overlaps or has wrong size")
        covered.update(ids)
    if covered != selected["evaluation"]:
        raise ValueError("Evaluation shards do not cover the population")
    if canonical and build_samples(inventory, roles) != value:
        raise ValueError("Sample registry differs from canonical selection")


def iter_sample(root, inventory, roles, review, samples, role, *, shard=None, file_path=None):
    """No labels, old outer held-out roles, or caller-supplied unchecked masks."""
    from .contracts import validate_compatibility
    validate_samples(samples, inventory, roles)
    validate_compatibility(review, inventory_hash=inventory["content_hash"])
    if role not in COUNTS or (shard is not None and (role != "evaluation" or type(shard) is not int or not 0 <= shard < EVAL_SHARDS)):
        raise PermissionError("Forbidden development read")
    rows = samples["evaluation_shards"][shard] if shard is not None else samples["members"][role]
    if file_path is not None:
        if file_path not in {r["path"] for r in rows}:
            raise PermissionError("File not in requested development role")
        rows = [r for r in rows if r["path"] == file_path]
    files = {r["path"]: r for r in inventory["files"]}
    branches = cms_particle_branches("offline") + cms_particle_branches("hlt")
    for row in rows:
        f = files[row["path"]]
        selected = unpack_entries(row["entry_mask"], f["raw_entries"])
        with authenticated_open(safe_relative(Path(root), f["path"]), f["sha256"]) as handle:
            key, tree = latest_tree(handle)
            if key != f["tree_key"] or tree.num_entries != f["raw_entries"]:
                raise ValueError("CMS development tree changed")
            for start in sorted(set((selected // 256 * 256).tolist())):
                entries = selected[np.searchsorted(selected, start):np.searchsorted(selected, start+256)]
                arrays = tree.arrays(list(branches), entry_start=start, entry_stop=start+256, library="ak", how=dict)
                for entry in entries:
                    cols = {name: ak.to_numpy(arrays[name][int(entry)-start]) for name in branches}
                    from .contracts import canonical_sha256
                    identity = canonical_sha256(["CMS2JC2_ROW/v1", f["sha256"], key, int(entry)])
                    yield Pair(identity, f["sha256"], from_cms(cols, review, side="offline"),
                               from_cms(cols, review, side="hlt"))


def sample_stream(context, role, **kwargs):
    return closing(iter_sample(context["cms_root"], context["inventory"], context["roles"],
                               context["review"], context["samples"], role, **kwargs))
