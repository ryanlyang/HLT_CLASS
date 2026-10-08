"""Source-bound immutable metadata publication and full read-only verification."""
from __future__ import annotations

from pathlib import Path
import platform
import re
import subprocess

import awkward
import numpy
import uproot

from hlt_classification.data.cache_contracts import load_json, sha256_file, write_immutable_json
from hlt_classification.provenance import (
    capture_source_snapshot, validate_source_snapshot, validate_source_snapshot_payload,
)
from .contracts import artifact, validate
from .inventory import CUTFLOW, build_inventory, validate_inventory, verify_source_bytes
from .splits import build_splits, validate_splits

REQUIRED_SOURCE = (
    "src/hlt_classification/luka_fullsim/__init__.py",
    "src/hlt_classification/luka_fullsim/contracts.py",
    "src/hlt_classification/luka_fullsim/inventory.py",
    "src/hlt_classification/luka_fullsim/splits.py",
    "src/hlt_classification/luka_fullsim/foundation.py",
    "scripts/luka_fullsim_foundation.py",
    "sbatch/run_luka_fullsim_foundation.sh",
    "docs/plans/LUKA_FULLSIM_FOUNDATION_PLAN.md",
    "docs/contracts/LUKA_FULLSIM_FOUNDATION.md",
)


def _source(project: Path, expected_commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Expected commit must be a full lowercase Git hash")
    project = Path(project).resolve(strict=True)
    if Path(__file__).resolve() != project / "src/hlt_classification/luka_fullsim/foundation.py":
        raise ValueError("Imported foundation implementation is outside the pinned project")
    source = capture_source_snapshot(project, require_clean=True)
    if source["git_commit"] != expected_commit:
        raise ValueError("Active source commit differs")
    tracked = set(subprocess.run(["git", "-C", str(project), "ls-files"], check=True,
                                 capture_output=True, text=True).stdout.splitlines())
    if not set(REQUIRED_SOURCE).issubset(tracked):
        raise ValueError("Commit the new implementation and contracts before building")
    published = subprocess.run(["git", "-C", str(project), "merge-base", "--is-ancestor",
                                expected_commit, "origin/main"], capture_output=True)
    if published.returncode != 0:
        raise ValueError("Commit is not in fetched origin/main; push and fetch before building")
    return source


def build(container: Path, output: Path, *, project: Path, expected_commit: str) -> dict:
    """Production stage-1 entry: fixed 890-file snapshot and exact 100k/50k."""
    source = _source(project, expected_commit)
    container = Path(container).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(container) or container.is_relative_to(output):
        raise ValueError("Foundation output must be separate from raw transfer container")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive directory claim. Failure leaves inspectable partial metadata.
    output.mkdir(exist_ok=False)
    print("Authenticating all 890 source files and scanning scalar metadata only.", flush=True)
    inventory = build_inventory(container)
    splits = build_splits(inventory)
    return _publish(output, container, project, source, inventory, splits)


def _publish(output: Path, container: Path, project: Path, source: dict,
             inventory: dict, splits: dict) -> dict:
    """Publish a validated bundle, foundation last. Internal, not a fixture flag."""
    validate_inventory(inventory)
    validate_splits(splits, inventory)
    verify_source_bytes(container, inventory)
    validate_source_snapshot(source, repository=project, require_clean=True)
    for name, payload in (("inventory.json", inventory), ("split_manifest.json", splits)):
        write_immutable_json(output / name, payload)
    result = artifact(
        "FOUNDATION", parents=dict(inventory=inventory["content_hash"], splits=splits["content_hash"],
                                   source=source["content_hash"]), source_snapshot=source,
        artifacts={name: dict(sha256=sha256_file(output / name), bytes=(output / name).stat().st_size)
                   for name in ("inventory.json", "split_manifest.json")},
        input_container=str(container), runtime=dict(python=platform.python_version(), numpy=numpy.__version__,
                                                    awkward=awkward.__version__, uproot=uproot.__version__),
        counts=splits["selected_counts"], readiness="scalar_metadata_only_not_training_admission",
        source_particles_validated=False, final_test_accessed=False, test_bytes_hashed=True,
        pre_split_scalar_metadata_accessed=True,
        remaining_checks=["producer_label_semantics", "tracking_units_and_sentinels",
                          "particle_array_validation", "cross_file_event_provenance",
                          "matching_and_views", "genuine_SPORC_GPU_preflight"],
    )
    write_immutable_json(output / "foundation.json", result)
    return result


def load_foundation(root: Path) -> tuple[dict, dict, dict]:
    root = Path(root).resolve(strict=True)
    foundation = load_json(root / "foundation.json")
    validate(foundation, "FOUNDATION")
    source = foundation["source_snapshot"]
    validate_source_snapshot_payload(source)
    if (source["worktree_clean"] is not True or foundation["source_particles_validated"] is not False
            or foundation["final_test_accessed"] is not False or foundation["test_bytes_hashed"] is not True
            or foundation["pre_split_scalar_metadata_accessed"] is not True
            or foundation["readiness"] != "scalar_metadata_only_not_training_admission"
            or set(foundation["artifacts"]) != {"inventory.json", "split_manifest.json"}):
        raise ValueError("Foundation readiness/source semantics differ")
    for name, record in foundation["artifacts"].items():
        path = root / name
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Foundation payload checksum differs: {name}")
    inventory, splits = load_json(root / "inventory.json"), load_json(root / "split_manifest.json")
    expected = dict(inventory=validate_inventory(inventory), splits=validate_splits(splits, inventory),
                    source=source["content_hash"])
    if foundation["parents"] != expected or foundation["counts"] != splits["selected_counts"]:
        raise ValueError("Foundation parents/counts differ")
    return foundation, inventory, splits


def verify(root: Path, *, container: Path | None = None) -> dict:
    foundation, inventory, _ = load_foundation(root)
    location = Path(container) if container is not None else Path(foundation["input_container"])
    replay = build_inventory(location, expected_files=inventory["file_count"])
    if replay != inventory:
        raise ValueError("Raw snapshot/scalar inventory replay differs")
    print("PASS: source bytes, scalar inventory and exact split replay verified. No particles decoded.", flush=True)
    return foundation


def print_summary(root: Path) -> None:
    foundation, inventory, splits = load_foundation(root)
    totals = inventory["selected_class_counts"]
    print(f"Source files: {inventory['file_count']}; raw entries: {inventory['raw_entries']:,}")
    print(f"Eligible paired jets: {sum(totals):,}; offline jet_pt > 200 GeV; no HLT pT cut")
    for name in CUTFLOW:
        print(f"  {name}: {sum(r['cutflow'][name] for r in inventory['files']):,}")
    print("Cutflow exclusions are sequential; each row is counted once.")
    print(f"{'class':<15} {'eligible':>10} {'fraction':>10} {'train':>10} {'validation':>12}")
    for i, name in enumerate(inventory["selection"]["class_names"]):
        print(f"{name:<15} {totals[i]:>10,} {totals[i]/sum(totals):>9.3%} "
              f"{splits['quotas']['train'][i]:>10,} {splits['quotas']['validation'][i]:>12,}")
    for role in ("train", "validation"):
        unused = sum(splits["unused_ordinary_class_counts"][role])
        print(f"Unused eligible rows in {role} files: {unused:,} (not test)")
    print(f"Sealed file-disjoint test reserve: {foundation['counts']['final_test']:,}")
    print(f"Foundation hash: {foundation['content_hash']}")
    print("Metadata validated; raw bytes are rechecked only by verify. Stage 2 remains required.")
    print("No particle arrays decoded, test evaluation, training or jobs submitted.")
