"""Root-independent transfer and authenticated relocation of proxy-ladder inputs."""
from __future__ import annotations

from pathlib import Path
import os
import shutil

from hlt_classification.cms2jc2_production.campaign import validate_study
from hlt_classification.cms2jc2_production.contracts import load_json, safe as source_safe
from hlt_classification.data.cache_contracts import sha256_file

from .contracts import (
    artifact, file_ref, safe, validate, validate_file_ref, write_json,
)
from .data import validate_foundation
from .release import release_request, validate_release


def _copy(source: Path, *, root: Path, relative: str) -> dict:
    source = Path(source).resolve(strict=True)
    target = safe(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Portable payload target already exists: {target}")
    temporary = target.with_name("." + target.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale portable payload temporary exists: {temporary}")
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != source.stat().st_size or sha256_file(temporary) != sha256_file(source):
        raise ValueError(f"Portable payload copy differs: {source}")
    os.replace(temporary, target)
    return file_ref(target, root=root)


def _foundation_envelope(value: dict) -> str:
    parents = {
        "release": value["release"]["content_hash"],
        "matcher": value["matcher"]["content_hash"],
        "views": value["views"]["content_hash"],
        "inputs": value["inputs"]["content_hash"],
    }
    return validate(value, "FOUNDATION", parents=parents)


def export_bundle(*, source_gate_root: Path, output_root: Path) -> dict:
    """Copy only ordinary-role bytes needed by the frozen 200k/100k release."""
    from .gate import validate_gate

    source_root = Path(source_gate_root).resolve(strict=True)
    gate = load_json(source_root / "gate_spec.json")
    validate_gate(gate)
    foundation_root = Path(gate.get("foundation_root", source_root / "foundation"))
    foundation = load_json(foundation_root / "foundation.json")
    validate_foundation(foundation, root=foundation_root)
    release_root = Path(foundation["release_root"])
    release = load_json(release_root / "release.json")
    validate_release(release, root=release_root)
    if foundation["release"] != release:
        raise ValueError("Portable source foundation/release lineage differs")

    study_root = Path(release["study_root"]).resolve(strict=True)
    offline_root = Path(release["offline_root"]).resolve(strict=True)
    study = load_json(study_root / "study_spec.json")
    validate_study(study)
    if study["content_hash"] != release["parents"]["study"]:
        raise ValueError("Portable source study/release lineage differs")
    inventory_record = study["imports"]["inventory"]
    inventory_path = source_safe(study_root, inventory_record["relative"])
    if (
        inventory_path.stat().st_size != inventory_record["bytes"]
        or sha256_file(inventory_path) != inventory_record["sha256"]
    ):
        raise ValueError("Portable source inventory bytes differ")

    target = Path(output_root).resolve()
    stage = target.with_name(target.name + ".partial")
    if target.exists() or stage.exists():
        raise FileExistsError("Portable bundle destination and staging root must be fresh")
    stage.mkdir(parents=True, exist_ok=False)
    payload = []

    def copy(source: Path, relative: str, category: str) -> None:
        payload.append({"category": category, **_copy(source, root=stage, relative=relative)})

    copy(source_root / "gate_spec.json", "metadata/source_gate.json", "source_gate")
    copy(release_root / "release.json", "release/release.json", "release")
    copy(release_root / "release_index.npz", "release/release_index.npz", "release")
    copy(foundation_root / "foundation.json", "foundation/foundation.json", "foundation")
    copy(foundation_root / "assignments.npz", "foundation/assignments.npz", "foundation")
    copy(study_root / "study_spec.json", "study/study_spec.json", "study")
    copy(
        inventory_path,
        "study/" + inventory_record["relative"],
        "ordinary_inventory",
    )
    for row in release["source_files"]:
        copy(offline_root / row["path"], "offline/" + row["path"], "offline_source")
    for row in release["proxy_blocks"]:
        copy(source_safe(study_root, row["path"]), "study/" + row["path"], "proxy_block")

    manifest = artifact(
        "PORTABLE_BUNDLE",
        parents={
            "source_gate": gate["content_hash"],
            "release": release["content_hash"],
            "foundation": foundation["content_hash"],
            "study": study["content_hash"],
        },
        source_gate=gate,
        release_sha256=release["content_hash"],
        foundation_sha256=foundation["content_hash"],
        study_sha256=study["content_hash"],
        counts=release["counts"],
        identity_sha256=release["identity_sha256"],
        source_locations={
            "gate_root": str(source_root),
            "release_root": str(release_root),
            "foundation_root": str(foundation_root),
            "study_root": str(study_root),
            "offline_root": str(offline_root),
        },
        layout={
            "release_root": "release",
            "foundation_root": "foundation",
            "study_root": "study",
            "offline_root": "offline",
        },
        payload=payload,
        payload_bytes=sum(row["bytes"] for row in payload),
        roles=["train", "validation"],
        labels_read_for_selection=False,
        final_test_files_included=False,
    )
    write_json(stage / "bundle.json", manifest)
    validate_bundle(manifest, root=stage)
    os.replace(stage, target)
    return manifest


def _expected_payload(bundle: dict, *, root: Path) -> set[str]:
    release = load_json(root / "release/release.json")
    study = load_json(root / "study/study_spec.json")
    return {
        "metadata/source_gate.json",
        "release/release.json",
        "release/release_index.npz",
        "foundation/foundation.json",
        "foundation/assignments.npz",
        "study/study_spec.json",
        "study/" + study["imports"]["inventory"]["relative"],
        *("offline/" + row["path"] for row in release["source_files"]),
        *("study/" + row["path"] for row in release["proxy_blocks"]),
    }


def validate_bundle(bundle: dict, *, root: Path) -> str:
    digest = validate(
        bundle,
        "PORTABLE_BUNDLE",
        parents={
            "source_gate": bundle["source_gate"]["content_hash"],
            "release": bundle["release_sha256"],
            "foundation": bundle["foundation_sha256"],
            "study": bundle["study_sha256"],
        },
    )
    base = Path(root).resolve(strict=True)
    paths = [row["path"] for row in bundle["payload"]]
    if len(paths) != len(set(paths)) or set(paths) != _expected_payload(bundle, root=base):
        raise ValueError("Portable bundle payload coverage differs")
    for row in bundle["payload"]:
        validate_file_ref(row, root=base)
    release = load_json(base / "release/release.json")
    validate_release(release, root=base / "release")
    foundation = load_json(base / "foundation/foundation.json")
    _foundation_envelope(foundation)
    validate_file_ref(foundation["assignments"], root=base / "foundation")
    study = load_json(base / "study/study_spec.json")
    validate_study(study)
    if (
        release["content_hash"] != bundle["release_sha256"]
        or foundation["content_hash"] != bundle["foundation_sha256"]
        or study["content_hash"] != bundle["study_sha256"]
        or foundation["release"] != release
        or bundle["counts"] != release["counts"]
        or bundle["identity_sha256"] != release["identity_sha256"]
        or bundle["roles"] != ["train", "validation"]
        or bundle["labels_read_for_selection"] is not False
        or bundle["final_test_files_included"] is not False
        or bundle["payload_bytes"] != sum(row["bytes"] for row in bundle["payload"])
    ):
        raise ValueError("Portable bundle frozen lineage differs")
    return digest


def _relocated_release(original: dict, *, release_root: Path, study_root: Path,
                       offline_root: Path) -> dict:
    request = release_request(study_root=study_root, offline_root=offline_root)
    fields = {
        key: value for key, value in original.items()
        if key not in {"contract", "schema_version", "content_hash", "parents", "final_test_accessed"}
    }
    fields.update(
        request=request,
        study_root=str(Path(study_root).resolve()),
        offline_root=str(Path(offline_root).resolve()),
        relocated_from=original["content_hash"],
    )
    return artifact(
        "RELEASE",
        parents={"request": request["content_hash"], "study": original["parents"]["study"]},
        **fields,
    )


def _relocated_foundation(original: dict, *, release: dict, release_root: Path) -> dict:
    fields = {
        key: value for key, value in original.items()
        if key not in {"contract", "schema_version", "content_hash", "parents", "final_test_accessed"}
    }
    fields.update(
        release=release,
        release_root=str(Path(release_root).resolve()),
        relocated_from=original["content_hash"],
    )
    return artifact(
        "FOUNDATION",
        parents={
            "release": release["content_hash"],
            "matcher": original["matcher"]["content_hash"],
            "views": original["views"]["content_hash"],
            "inputs": original["inputs"]["content_hash"],
        },
        **fields,
    )


def materialize_bundle(*, bundle_root: Path, output_root: Path) -> dict:
    """Publish site-local wrappers while retaining all frozen physical bytes."""
    bundle_root = Path(bundle_root).resolve(strict=True)
    bundle = load_json(bundle_root / "bundle.json")
    validate_bundle(bundle, root=bundle_root)
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("Portable materialization root must be fresh")
    (root / "release").mkdir(parents=True, exist_ok=False)
    (root / "foundation").mkdir()
    shutil.copyfile(bundle_root / "release/release_index.npz", root / "release/release_index.npz")
    shutil.copyfile(bundle_root / "foundation/assignments.npz", root / "foundation/assignments.npz")
    original_release = load_json(bundle_root / "release/release.json")
    release = _relocated_release(
        original_release,
        release_root=root / "release",
        study_root=bundle_root / bundle["layout"]["study_root"],
        offline_root=bundle_root / bundle["layout"]["offline_root"],
    )
    write_json(root / "release/release.json", release)
    validate_release(release, root=root / "release")
    original_foundation = load_json(bundle_root / "foundation/foundation.json")
    foundation = _relocated_foundation(
        original_foundation, release=release, release_root=root / "release",
    )
    write_json(root / "foundation/foundation.json", foundation)
    validate_foundation(foundation, root=root / "foundation")
    value = artifact(
        "PORTABLE_MATERIALIZATION",
        parents={
            "bundle": bundle["content_hash"],
            "release": release["content_hash"],
            "foundation": foundation["content_hash"],
        },
        bundle_root=str(bundle_root),
        bundle_sha256=bundle["content_hash"],
        release_root=str((root / "release").resolve()),
        release=release,
        foundation_root=str((root / "foundation").resolve()),
        foundation=foundation,
        identity_sha256=release["identity_sha256"],
        physical_payload_reused=True,
        assignments_recomputed=False,
    )
    write_json(root / "materialization.json", value)
    validate_materialization(value, root=root)
    return value


def validate_materialization(value: dict, *, root: Path) -> str:
    digest = validate(
        value,
        "PORTABLE_MATERIALIZATION",
        parents={
            "bundle": value["bundle_sha256"],
            "release": value["release"]["content_hash"],
            "foundation": value["foundation"]["content_hash"],
        },
    )
    base = Path(root).resolve(strict=True)
    bundle_root = Path(value["bundle_root"])
    bundle = load_json(bundle_root / "bundle.json")
    validate_bundle(bundle, root=bundle_root)
    if (
        base / "release" != Path(value["release_root"])
        or base / "foundation" != Path(value["foundation_root"])
        or value["bundle_sha256"] != bundle["content_hash"]
        or value["identity_sha256"] != value["release"]["identity_sha256"]
        or value["physical_payload_reused"] is not True
        or value["assignments_recomputed"] is not False
    ):
        raise ValueError("Portable materialization paths or semantics differ")
    release = load_json(base / "release/release.json")
    foundation = load_json(base / "foundation/foundation.json")
    validate_release(release, root=base / "release")
    validate_foundation(foundation, root=base / "foundation")
    if release != value["release"] or foundation != value["foundation"]:
        raise ValueError("Portable materialization embedded artifacts differ")
    return digest


__all__ = [
    "export_bundle", "materialize_bundle", "validate_bundle",
    "validate_materialization",
]
