"""Immutable small artifacts and exact task receipts, without resume state."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, write_immutable_json,
)
from .contracts import artifact, validate


def publish_npz(path, **arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_publish_bytes(path, stream.getvalue())
    return fingerprint(path)


def fingerprint(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size)


def checked_file(row):
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise ValueError(f"Artifact payload changed: {path}")
    return path


def arrays_from(row):
    with np.load(checked_file(row), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def receipt_path(spec, task):
    return Path(spec["campaign_root"]) / "receipts" / f"{task}.json"


def load_receipt(spec, task):
    row = load_json(receipt_path(spec, task))
    validate(row, "RECEIPT")
    if row["campaign_spec_sha256"] != spec["content_hash"] or row["task"] != task:
        raise ValueError("Task receipt source differs")
    for output in row["outputs"]:
        path = checked_file(output)
        if not path.resolve().is_relative_to(Path(spec["campaign_root"]).resolve()):
            raise ValueError("Task output escapes campaign")
    return row


def publish_receipt(spec, task, outputs):
    result = artifact("RECEIPT", campaign_spec_sha256=spec["content_hash"], task=task,
                      outputs=[fingerprint(path) for path in outputs], final_test_accessed=False)
    write_immutable_json(receipt_path(spec, task), result)
    return result
