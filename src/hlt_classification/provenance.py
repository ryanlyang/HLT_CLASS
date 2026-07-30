"""Clean-source snapshots and fail-closed active-source validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    validate_content_hash,
    with_content_hash,
)

SOURCE_SNAPSHOT_CONTRACT = "hlt_classification_source_snapshot_v1"
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1


def validate_source_snapshot_payload(payload: Mapping[str, Any]) -> str:
    """Validate serialized source semantics without consulting a worktree."""

    digest = validate_content_hash(
        payload,
        expected_contract=SOURCE_SNAPSHOT_CONTRACT,
    )
    commit = payload.get("git_commit")
    tree = payload.get("git_tree")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("source git commit must be a 40-character object id")
    if not isinstance(tree, str) or len(tree) != 40:
        raise ValueError("source git tree must be a 40-character object id")
    require_sha256(
        payload.get("tracked_files_sha256"),
        name="tracked_files_sha256",
    )
    require_sha256(
        payload.get("source_snapshot_sha256"),
        name="source_snapshot_sha256",
    )
    expected_identity = canonical_sha256(
        {
            "git_commit": commit,
            "git_tree": tree,
            "tracked_files_sha256": payload["tracked_files_sha256"],
        }
    )
    if payload["source_snapshot_sha256"] != expected_identity:
        raise ValueError("source snapshot scientific identity differs")
    count = payload.get("tracked_file_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("source snapshot tracked-file count is invalid")
    if not isinstance(payload.get("worktree_clean"), bool):
        raise ValueError("source snapshot clean-state flag is invalid")
    return digest


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def _tracked_file_digest(repository: Path) -> tuple[str, int]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    relative_paths = [
        value.decode("utf-8")
        for value in raw.split(b"\0")
        if value
    ]
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = repository / relative
        if not path.is_file():
            raise ValueError(f"tracked source file is absent: {relative}")
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(relative_paths)


def capture_source_snapshot(
    repository: str | Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise ValueError("source snapshot repository root differs")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise ValueError("production source worktree is dirty")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    tracked_digest, tracked_count = _tracked_file_digest(root)
    payload = {
        "contract": SOURCE_SNAPSHOT_CONTRACT,
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "git_commit": commit,
        "git_tree": tree,
        "tracked_files_sha256": tracked_digest,
        "tracked_file_count": tracked_count,
        "worktree_clean": status == "",
        "source_snapshot_sha256": canonical_sha256(
            {
                "git_commit": commit,
                "git_tree": tree,
                "tracked_files_sha256": tracked_digest,
            }
        ),
    }
    return with_content_hash(payload)


def validate_source_snapshot(
    payload: Mapping[str, Any],
    *,
    repository: str | Path,
    require_clean: bool = True,
) -> str:
    digest = validate_source_snapshot_payload(payload)
    active = capture_source_snapshot(repository, require_clean=require_clean)
    compared = (
        "git_commit",
        "git_tree",
        "tracked_files_sha256",
        "tracked_file_count",
        "worktree_clean",
        "source_snapshot_sha256",
    )
    differences = [
        key for key in compared if payload.get(key) != active.get(key)
    ]
    if differences:
        raise ValueError(f"active source snapshot differs: {differences}")
    return digest


def validate_campaign_source(
    campaign_spec: Mapping[str, Any],
    *,
    repository: str | Path,
) -> str:
    source = campaign_spec.get("source_snapshot")
    if not isinstance(source, Mapping):
        raise ValueError("campaign specification has no source snapshot")
    return validate_source_snapshot(
        source,
        repository=repository,
        require_clean=True,
    )


__all__ = [
    "SOURCE_SNAPSHOT_CONTRACT",
    "capture_source_snapshot",
    "validate_campaign_source",
    "validate_source_snapshot",
    "validate_source_snapshot_payload",
]
