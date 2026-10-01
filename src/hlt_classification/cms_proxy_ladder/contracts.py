"""Versioned artifacts and safe publication for the proxy ladder."""
from __future__ import annotations

from pathlib import Path
import json

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    sha256_file,
    validate_content_hash,
    with_content_hash,
)

PREFIX = "JETCLASS2_CMS_PROXY_LADDER_"


def artifact(kind: str, *, version: int = 1, parents=None, **fields) -> dict:
    reserved = {
        "contract", "schema_version", "content_hash", "parents",
        "final_test_accessed",
    }
    if reserved.intersection(fields) or type(version) is not int or version < 1:
        raise ValueError("Reserved or invalid proxy-ladder artifact fields")
    return with_content_hash({
        "contract": f"{PREFIX}{kind}/v{version}",
        "schema_version": version,
        "parents": dict(parents or {}),
        "final_test_accessed": False,
        **fields,
    })


def validate(value: dict, kind: str, *, version: int = 1, parents=None) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=f"{PREFIX}{kind}/v{version}",
        expected_schema_version=version,
    )
    if value.get("final_test_accessed") is not False:
        raise PermissionError("Proxy-ladder ordinary artifacts cannot access final test")
    if parents is not None and value.get("parents") != parents:
        raise ValueError(f"Proxy-ladder {kind} parent lineage differs")
    return digest


def safe(root: Path, relative: str) -> Path:
    base = Path(root).resolve()
    parts = str(relative).split("/")
    if any(part in {"", ".", ".."} for part in parts) or "\\" in str(relative) or ":" in str(relative):
        raise ValueError("Unsafe proxy-ladder relative path")
    path = base
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in proxy-ladder outputs")
    if not path.resolve().is_relative_to(base):
        raise ValueError("Proxy-ladder path escaped its root")
    return path


def write_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_publish_bytes(Path(path), payload)


def file_ref(path: Path, *, root: Path | None = None) -> dict:
    target = Path(path).resolve(strict=True)
    name = str(target) if root is None else target.relative_to(Path(root).resolve()).as_posix()
    return {"path": name, "sha256": sha256_file(target), "bytes": target.stat().st_size}


def validate_file_ref(record: dict, *, root: Path | None = None) -> Path:
    path = Path(record["path"]) if root is None else safe(root, record["path"])
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Proxy-ladder file bytes differ: {path}")
    return path


__all__ = [
    "PREFIX", "artifact", "file_ref", "safe", "validate", "validate_file_ref",
    "write_json",
]
