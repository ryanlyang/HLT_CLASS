#!/usr/bin/env python3
"""Publish the four pre-campaign HCWDL-RKD parent-evidence artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact, load_json_mapping, publish
from hlt_classification.models.hcwdl_surfaces import (
    HCWDL_ARCHITECTURE_MODEL_SOURCE_KEYS,
    build_architecture_attestation_from_files,
    tap_schema,
    validate_architecture_attestation,
)
from hlt_classification.scouting.hcwdl_parent_loss import (
    PARENT_LOSS_RUNTIME_SOURCE_FILES,
    build_parent_loss_attestation_from_reports,
    validate_parent_loss_attestation,
)
from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
    build_installed_weaver_surface_parity_artifact,
)


def _path_mapping(path: Path, *, name: str) -> dict[str, Path]:
    raw = load_json_mapping(path)
    if not raw:
        raise ValueError(f"{name} registry is empty")
    result: dict[str, Path] = {}
    for key, value in sorted(raw.items()):
        candidate = Path(str(value))
        if not candidate.is_absolute() or not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"{name} path is not an immutable absolute file: {key}")
        result[key] = candidate
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--parent-campaign-spec", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-reports", type=Path, required=True)
    parser.add_argument("--model-sources", type=Path, required=True)
    parser.add_argument("--runtime-sources", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    root = args.representation_root
    if not root.is_absolute():
        raise ValueError("representation root must be absolute")
    for name, path in (
        ("parent campaign specification", args.parent_campaign_spec),
        ("parent recipe", args.parent_recipe),
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"{name} must be an existing absolute regular file")
    parent_reports = _path_mapping(args.parent_reports, name="parent report")
    model_sources = _path_mapping(args.model_sources, name="model source")
    if set(model_sources) != HCWDL_ARCHITECTURE_MODEL_SOURCE_KEYS:
        raise ValueError(
            "model source registry is incomplete or expanded; "
            f"missing={sorted(HCWDL_ARCHITECTURE_MODEL_SOURCE_KEYS - set(model_sources))}, "
            f"extra={sorted(set(model_sources) - HCWDL_ARCHITECTURE_MODEL_SOURCE_KEYS)}"
        )
    runtime_sources = _path_mapping(args.runtime_sources, name="runtime source")
    if set(runtime_sources) != set(PARENT_LOSS_RUNTIME_SOURCE_FILES):
        raise ValueError(
            "runtime source registry is incomplete or expanded; "
            f"missing={sorted(set(PARENT_LOSS_RUNTIME_SOURCE_FILES) - set(runtime_sources))}, "
            f"extra={sorted(set(runtime_sources) - set(PARENT_LOSS_RUNTIME_SOURCE_FILES))}"
        )

    tap_path = root / "architecture" / "tap.json"
    parity_path = root / "architecture" / "surface_parity.json"
    architecture_path = root / "import" / "architecture_attestation.json"
    parent_loss_path = root / "import" / "parent_loss_attestation.json"

    publish(tap_path, tap_schema())
    publish(
        parity_path,
        build_installed_weaver_surface_parity_artifact(device=args.device),
    )
    architecture = build_architecture_attestation_from_files(
        tap_schema_path=tap_path, surface_parity_path=parity_path,
        parent_reports=parent_reports, model_source_paths=model_sources,
    )
    validate_architecture_attestation(architecture, require_authorized=True)
    publish(architecture_path, architecture)

    parent_loss = build_parent_loss_attestation_from_reports(
        parent_recipe_path=args.parent_recipe,
        parent_campaign_spec_path=args.parent_campaign_spec,
        parent_reports=parent_reports,
        runtime_source_paths=runtime_sources,
    )
    validate_parent_loss_attestation(
        parent_loss, parent_recipe=artifact(args.parent_recipe),
    )
    publish(parent_loss_path, parent_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
