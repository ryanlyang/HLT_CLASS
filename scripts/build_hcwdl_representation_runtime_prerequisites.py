#!/usr/bin/env python3
"""Assemble the closed HCWDL-RKD runtime prerequisite registries."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

from _hcwdl_representation_common import artifact, load_json_mapping, publish
from hlt_classification.data.cache_contracts import load_json, require_sha256, sha256_file
from hlt_classification.scouting.hcwdl_representation_campaign import (
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_runtime_rows import (
    build_runtime_prerequisites,
)
from hlt_classification.scouting.engine import validate_pmard_training_report
from hlt_classification.scouting.hcwdl_training import (
    validate_hcwdl_training_report,
)


def _paired_wrapper_and_engine(
    root: Path, relative_text: object, *, node_id: str,
) -> dict[str, object]:
    """Open one registered HCWDL wrapper and its authenticated PMARD sibling."""

    relative = PurePosixPath(str(relative_text))
    if (
        not str(relative_text) or relative.is_absolute() or "\\" in str(relative_text)
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"paired parent wrapper path is unsafe: {node_id}")
    resolved_root = root.resolve()
    wrapper_path = root.joinpath(*relative.parts)
    try:
        wrapper_path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"paired parent wrapper escapes its authenticated bundle: {node_id}"
        ) from exc
    if not wrapper_path.is_file() or wrapper_path.is_symlink():
        raise ValueError(f"paired parent wrapper is absent or a symlink: {node_id}")
    wrapper = load_json(wrapper_path)
    wrapper_sha256 = validate_hcwdl_training_report(wrapper)
    if wrapper.get("node_id") != node_id:
        raise ValueError(f"paired parent wrapper node differs: {node_id}")

    engine_path = wrapper_path.with_name("training_report.json")
    try:
        engine_relative = engine_path.resolve().relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"paired parent engine report escapes its authenticated bundle: {node_id}"
        ) from exc
    if not engine_path.is_file() or engine_path.is_symlink():
        raise ValueError(
            f"paired parent wrapper lacks sibling PMARD training_report.json: {node_id}"
        )
    engine = load_json(engine_path)
    engine_sha256 = validate_pmard_training_report(engine)
    expected_engine_sha256 = require_sha256(
        wrapper.get("pmard_engine_report_sha256"),
        name=f"{node_id} wrapper PMARD engine report",
    )
    if engine_sha256 != expected_engine_sha256:
        raise ValueError(f"paired parent wrapper/engine hash differs: {node_id}")
    if (
        wrapper.get("selected_checkpoint_sha256")
        != engine.get("selected_checkpoint_sha256")
        or wrapper.get("pmard_execution_config_sha256")
        != engine.get("execution_config_sha256")
    ):
        raise ValueError(f"paired parent wrapper/engine lineage differs: {node_id}")
    return {
        "wrapper_relative": relative.as_posix(),
        "wrapper": wrapper,
        "wrapper_sha256": wrapper_sha256,
        "engine_relative": engine_relative,
        "engine_report": engine,
        "engine_sha256": engine_sha256,
    }


def _paired_screen_model_member(
    *, node_id: str, parent_reports_root: Path, parent_wrapper_relative: object,
    finalist_models_root: Path, finalist_wrapper_relative: object,
) -> dict[str, object]:
    """Cross-bind both registered wrapper copies and return the PMARD member."""

    parent = _paired_wrapper_and_engine(
        parent_reports_root, parent_wrapper_relative, node_id=node_id,
    )
    finalist = _paired_wrapper_and_engine(
        finalist_models_root, finalist_wrapper_relative, node_id=node_id,
    )
    if parent["wrapper_sha256"] != finalist["wrapper_sha256"]:
        raise ValueError(
            f"paired parent screening wrapper/model source differs: {node_id}"
        )
    if parent["engine_sha256"] != finalist["engine_sha256"]:
        raise ValueError(
            f"paired parent screening engine/model source differs: {node_id}"
        )
    return {
        "relative": finalist["engine_relative"],
        "report": finalist["engine_report"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-campaign-spec", type=Path, required=True)
    parser.add_argument("--runtime-facts", type=Path, required=True)
    parser.add_argument("--runtime-signatures", type=Path, required=True)
    parser.add_argument("--static-inputs", type=Path, required=True)
    parser.add_argument("--artifact-content-hashes", type=Path, required=True)
    parser.add_argument("--target-generations", type=Path, required=True)
    parser.add_argument("--bundle-members", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = artifact(args.planning_campaign_spec)
    validate_campaign_spec(spec, executable=False)
    static_inputs = load_json_mapping(args.static_inputs)

    def registered_file(logical: str) -> Path:
        reference = static_inputs.get(logical)
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"static inputs lack the exact {logical} reference")
        path = Path(str(reference["path"]))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"registered input is not an immutable file: {logical}")
        if sha256_file(path) != require_sha256(
            reference["sha256"], name=f"{logical} byte hash",
        ):
            raise ValueError(f"registered input bytes differ: {logical}")
        return path

    def registered_directory(logical: str) -> Path:
        reference = static_inputs.get(logical)
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"static inputs lack the exact {logical} reference")
        path = Path(str(reference["path"]))
        if not path.is_absolute() or not path.is_dir() or path.is_symlink():
            raise ValueError(f"registered input is not an immutable directory: {logical}")
        from hlt_classification.scouting.hcwdl_representation_runtime_adapters import (
            _directory_inventory,
        )
        if _directory_inventory(path)["inventory_sha256"] != require_sha256(
            reference["sha256"], name=f"{logical} inventory hash",
        ):
            raise ValueError(f"registered directory inventory differs: {logical}")
        return path

    def member(root: Path, relative_text: object):
        relative = PurePosixPath(str(relative_text))
        if (
            not str(relative_text) or relative.is_absolute() or "\\" in str(relative_text)
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("parent finalist member path is unsafe")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("parent finalist member is absent or a symlink")
        path.resolve().relative_to(root.resolve())
        return load_json(path)

    split_path = registered_file("${split_manifest}")
    settings = load_json_mapping(args.settings)
    bundle_members = load_json_mapping(args.bundle_members)
    dense_training_only = spec.get("disposition") == "dense_training_only"

    def relative_member(root: Path, path: Path) -> tuple[str, dict]:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("parent finalist report escapes its authenticated bundle") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("parent finalist report is absent or a symlink")
        return relative, load_json(resolved)

    def bundle_relative(bundle: str, key: str) -> str:
        registry = bundle_members.get(bundle)
        if not isinstance(registry, dict) or key not in registry:
            raise ValueError(f"runtime bundle lacks {bundle}:{key}")
        raw = registry[key]
        relative = raw.get("relative") if isinstance(raw, dict) else raw
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"runtime bundle member differs: {bundle}:{key}")
        return relative

    if dense_training_only:
        if bundle_members:
            raise ValueError("dense runtime cannot accept parent/final bundle members")
        final_authorities = {}
    else:
        final = settings.get("final")
        if not isinstance(final, dict):
            raise ValueError("operator settings lack the final registry")
        finalist_root = registered_directory("${parent_finalist_registry}")
        finalist_models_root = registered_directory("${finalist_models}")
        parent_reports_root = registered_directory("${parent_reports}")
        parent_lock = member(
            finalist_root, final.get("parent_finalist_registry_relative"),
        )
        locked_model_members = {}
        for row in parent_lock.get("payload", {}).get("finalists", ()):
            if not isinstance(row, dict):
                raise ValueError("parent finalist lock row differs")
            key = f"{row.get('node_id')}:{row.get('seed')}"
            relative, report = relative_member(
                finalist_models_root, Path(str(row.get("report_path"))),
            )
            locked_model_members[key] = {"relative": relative, "report": report}
        paired_screen_model_members = {}
        for node_id in ("M0", "M1c", "M1w"):
            model_relative = bundle_relative("finalist_models", node_id)
            paired_screen_model_members[node_id] = _paired_screen_model_member(
                node_id=node_id,
                parent_reports_root=parent_reports_root,
                parent_wrapper_relative=bundle_relative("parent_reports", node_id),
                finalist_models_root=finalist_models_root,
                finalist_wrapper_relative=model_relative,
            )
        final_authorities = {
            "shared_final_population": artifact(
                registered_file("${shared_final_population}")
            ),
            "parent_final_state": artifact(
                registered_file("${parent_final_state}")
            ),
            "final_disposition": artifact(registered_file("${final_disposition}")),
            "parent_import": artifact(registered_file("${prebuilt_parent_import}")),
            "parent_finalist_registry": parent_lock,
            "parent_confirmation_registry": artifact(
                registered_file("${parent_confirmation_registry_lock}")
            ),
            "locked_model_members": locked_model_members,
            "paired_screen_model_members": paired_screen_model_members,
            "matcher_resources": artifact(registered_file("${matcher_resources}")),
        }
    target_forward_specs = {
        logical: artifact(registered_file(logical))
        for logical in sorted(static_inputs)
        if logical.startswith("${target_forward_spec:")
    }
    value = build_runtime_prerequisites(
        spec,
        runtime_facts=load_json_mapping(args.runtime_facts),
        runtime_signatures=load_json_mapping(args.runtime_signatures),
        static_inputs=static_inputs,
        artifact_content_hashes=load_json_mapping(args.artifact_content_hashes),
        target_generations=load_json_mapping(args.target_generations),
        bundle_members=bundle_members,
        settings=settings,
        split_manifest=artifact(split_path),
        final_authorities=final_authorities,
        target_forward_specs=target_forward_specs,
        representation_recipe=artifact(
            registered_file("${prebuilt_representation_recipe}")
        ),
    )
    publish(args.output, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
