"""Shared import/bootstrap helpers for thin HCWDL-RKD command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json  # noqa: E402


def artifact(path: Path) -> dict[str, Any]:
    return load_json(path)


def publish(path: Path, value: Mapping[str, Any]) -> None:
    write_immutable_json(path, value)


def scheduler(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True,
    ).stdout.strip()


def load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON mapping file must contain an object")
    return {str(key): item for key, item in value.items()}
