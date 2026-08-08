"""Filesystem-bound HCWDL task dispatcher used by the production shell worker."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, validate_content_hash, with_content_hash,
    write_immutable_json,
)

from .hcwdl_campaign import validate_campaign_spec
from .hcwdl_contracts import (
    artifact_envelope, authenticate_source_files, validate_artifact,
)
from .hcwdl_ladder import NODE_REGISTRY
from .hcwdl_locks import (
    create_assignment_lock, create_confirmation_registry_lock, create_execution_lock,
    create_finalist_lock, create_recipe_lock, create_shell_endpoint_qualification_lock,
    validate_lock,
)
from .hcwdl_qualification import (
    QUALIFIERS, build_qualification_report, compute_shell_strata,
    validate_diagnostic_acknowledgement,
)
from .hcwdl_recipe import validate_recipe, validate_recipe_class_weight_lineage
from .hcwdl_reporting import build_confirmation_registry, build_final_report, build_screen_aggregate
from .highcov_resources import resource_validation_report
from .audit import SOURCE_MANIFEST_CONTRACT, SOURCE_MANIFEST_VERSION
from .splits import source_file_record_from_manifest_row, validate_split_manifest


def _array_index() -> int:
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise RuntimeError("HCWDL array task lacks SLURM_ARRAY_TASK_ID")
    return int(value)


def _run(command: list[object], *, repository: Path) -> None:
    subprocess.run([str(item) for item in command], cwd=repository, check=True)


class HcwdlWorkflow:
    def __init__(self, spec: dict[str, Any], *, repository: str | Path) -> None:
        validate_campaign_spec(spec, executable=False)
        self.spec = spec
        self.repository = Path(repository).resolve()
        self.root = Path(spec["campaign_root"])
        self.data_root = Path(spec["data_root"])
        self.source_manifest = Path(spec["source_manifest_path"])
        self.split_manifest = Path(spec["split_manifest_path"])
        self.selection = self.root / "source/row_selection.json"
        self.resources = self.root / "matcher/resources_validation.json"
        self.assignment_root = self.root / "matcher/assignments"
        self.manifests = {
            role: self.root / f"matcher/{role}_assignment_manifest.json"
            for role in ("train", "validation", "final_test")
        }
        self.audits = {
            role: self.root / f"matcher/{role}_recomputation_audit.json"
            for role in ("train", "validation", "final_test")
        }
        self.locks = {name: self.root / f"locks/{name}.json" for name in (
            "assignment", "recipe", "shell_endpoint_qualification",
            "confirmation_registry", "finalist", "execution",
        )}
        self.recipe = None if spec.get("recipe_path") is None else Path(spec["recipe_path"])

    def _script(self, name: str, *arguments: object) -> list[object]:
        return [sys.executable, "-s", self.repository / "scripts" / name, *arguments]

    def _training_report(self, node_id: str) -> Path:
        return self.root / f"training/{node_id}/training_report.json"

    def _hcwdl_training_report(self, node_id: str) -> Path:
        return self.root / f"training/{node_id}/hcwdl_training_report.json"

    def _node_command(self, node_id: str, *, output: Path, seed: int) -> list[object]:
        if self.recipe is None:
            raise PermissionError("HCWDL training cannot run without a locked recipe")
        node = NODE_REGISTRY[node_id]
        command: list[object] = self._script(
            "train_hcwdl_node.py", "--node-id", node_id, "--recipe", self.recipe,
            "--split-manifest", self.split_manifest, "--selection-manifest", self.selection,
            "--data-root", self.data_root,
            "--assignment-manifest", f"train={self.manifests['train']}",
            "--assignment-manifest", f"validation={self.manifests['validation']}",
            "--output-dir", output, "--replicate-seed", seed,
            "--source-snapshot-sha256", self.spec["source_manifest_sha256"],
            "--assignment-lock-sha256", load_json(self.locks["assignment"])["content_hash"],
            "--qualification-lock-sha256", load_json(self.locks["shell_endpoint_qualification"])["content_hash"],
            "--device", "cuda",
        )
        for teacher in node.teachers:
            command.extend(("--teacher-report", f"{teacher.node_id}={self._training_report(teacher.node_id)}"))
        if node.initialization_parent is not None:
            command.extend(("--warm-parent-report", self._training_report(node.initialization_parent)))
        if self.spec["mode"] == "smoke":
            command.append("--smoke")
        return command

    def run(self, task_id: str) -> list[Path]:
        registered = {row["task_id"] for row in self.spec["tasks"]}
        if task_id not in registered:
            raise ValueError(f"task {task_id!r} is absent from this HCWDL spec")
        if task_id == "source_audit":
            source = load_json(self.source_manifest)
            source_hash = validate_content_hash(
                source, expected_contract=SOURCE_MANIFEST_CONTRACT,
                expected_schema_version=SOURCE_MANIFEST_VERSION,
            )
            if source_hash != self.spec["source_manifest_sha256"]:
                raise ValueError("HCWDL source manifest differs from campaign spec")
            authentication = authenticate_source_files(self.data_root, source.get("files", ()))
            output = self.root / "source/source_audit.json"
            write_immutable_json(output, artifact_envelope(
                kind="source_audit", payload={
                    "valid": True, "read_only": True, **authentication,
                },
                parents={"source_manifest": source["content_hash"]},
            )); return [output]
        if task_id == "splits":
            split = load_json(self.split_manifest)
            source = load_json(self.source_manifest)
            split_hash = validate_split_manifest(
                split, source_manifest_sha256=self.spec["source_manifest_sha256"],
                expected_inventory=(
                    source_file_record_from_manifest_row(row) for row in source["files"]
                ),
            )
            if split_hash != self.spec["split_manifest_sha256"]:
                raise ValueError("HCWDL split manifest differs from campaign spec")
            output = self.root / "source/split_audit.json"
            write_immutable_json(output, artifact_envelope(
                kind="split_audit", payload={"file_disjoint": True, "fractions": [0.6, 0.2, 0.2]},
                parents={"split_manifest": split["content_hash"]},
            )); return [output]
        if task_id == "data_lock":
            output = self.root / "locks/data_lock.json"
            source_audit = load_json(self.root / "source/source_audit.json")
            split_audit = load_json(self.root / "source/split_audit.json")
            source_audit_hash = validate_artifact(
                source_audit, expected_kind="source_audit",
                expected_parents={"source_manifest": self.spec["source_manifest_sha256"]},
            )
            split_audit_hash = validate_artifact(
                split_audit, expected_kind="split_audit",
                expected_parents={"split_manifest": self.spec["split_manifest_sha256"]},
            )
            write_immutable_json(output, artifact_envelope(
                kind="data_lock", payload={"authorized_roles": ["train", "validation"]},
                parents={
                    "source_manifest": self.spec["source_manifest_sha256"],
                    "split_manifest": self.spec["split_manifest_sha256"],
                    "source_audit": source_audit_hash,
                    "split_audit": split_audit_hash,
                },
            )); return [output]
        if task_id == "matcher_resources":
            write_immutable_json(self.resources, resource_validation_report()); return [self.resources]
        if task_id == "row_selection":
            counts = self.spec["role_counts"]
            command = self._script(
                "build_hcwdl_row_selection.py", "--split-manifest", self.split_manifest,
                "--data-root", self.data_root, "--output", self.selection,
                "--role-budget", f"train={'all' if counts['train'] is None else counts['train']}",
                "--role-budget", f"validation={'all' if counts['validation'] is None else counts['validation']}",
            )
            _run(command, repository=self.repository); return [self.selection]
        if task_id in {"assign_train", "assign_validation"}:
            role = task_id.removeprefix("assign_")
            _run(self._script(
                "build_highcov_assignment_shard.py", "--split-manifest", self.split_manifest,
                "--selection-manifest", self.selection, "--resources-report", self.resources,
                "--data-root", self.data_root, "--assignment-root", self.assignment_root,
                "--role", role, "--source-index", _array_index(),
            ), repository=self.repository)
            base = self.assignment_root / role / f"shard_{_array_index():04d}"
            return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if task_id == "assignment_manifest":
            outputs = []
            for role in ("train", "validation"):
                _run(self._script(
                    "finalize_highcov_assignments.py", "--split-manifest", self.split_manifest,
                    "--selection-manifest", self.selection, "--resources-report", self.resources,
                    "--assignment-root", self.assignment_root, "--role", role,
                    "--output", self.manifests[role],
                ), repository=self.repository)
                rows = int(load_json(self.manifests[role])["scanned_mapped_jets"])
                _run(self._script(
                    "audit_highcov_assignments.py", "--manifest", self.manifests[role],
                    "--split-manifest", self.split_manifest, "--data-root", self.data_root,
                    "--role", role, "--sample-size", min(256, rows), "--seed", 1337,
                    "--output", self.audits[role],
                ), repository=self.repository)
                outputs.extend((self.manifests[role], self.audits[role]))
            return outputs
        if task_id == "assignment_lock":
            parents = {
                "split_manifest_sha256": self.spec["split_manifest_sha256"],
                "row_selection_sha256": load_json(self.selection)["content_hash"],
                "matcher_resources_sha256": load_json(self.resources)["content_hash"],
            }
            lock = create_assignment_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                train_manifest_path=self.manifests["train"],
                validation_manifest_path=self.manifests["validation"],
                expected_train_rows=int(load_json(self.selection)["roles"]["train"]["rows"]),
                expected_validation_rows=int(load_json(self.selection)["roles"]["validation"]["rows"]),
                expected_parents=parents,
                train_recomputation_sha256=load_json(self.audits["train"])["content_hash"],
                validation_recomputation_sha256=load_json(self.audits["validation"])["content_hash"],
                matcher_resources_sha256=parents["matcher_resources_sha256"],
            )
            write_immutable_json(self.locks["assignment"], lock); return [self.locks["assignment"]]
        if task_id == "cache_miniature":
            output = self.root / "runtime/cache_miniature.json"
            _run(self._script(
                "run_hcwdl_cache_miniature.py", "--split-manifest", self.split_manifest,
                "--selection-manifest", self.selection, "--assignment-manifest", self.manifests["validation"],
                "--data-root", self.data_root, "--rows", 4096, "--output", output,
            ), repository=self.repository); return [output]
        if task_id == "recipe_lock":
            if self.recipe is None:
                raise PermissionError("HCWDL recipe remains unresolved")
            recipe = load_json(self.recipe); recipe_hash = validate_recipe(
                recipe, require_authorized=True, expected_profile="primary_ladder",
            )
            validate_recipe_class_weight_lineage(recipe, load_json(self.selection))
            if recipe_hash != self.spec["recipe_sha256"]:
                raise ValueError("HCWDL campaign recipe lineage differs")
            lock = create_recipe_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                assignment_lock=load_json(self.locks["assignment"]), recipe_sha256=recipe_hash,
                evidence_hashes=recipe["evidence"],
            )
            write_immutable_json(self.locks["recipe"], lock); return [self.locks["recipe"]]
        if task_id == "endpoint_qualification":
            qualifier = QUALIFIERS[_array_index()]
            output = self.root / f"qualification/{qualifier}"
            command = self._script(
                "train_hcwdl_qualifier.py", "--qualifier-id", qualifier,
                "--recipe", self.recipe, "--split-manifest", self.split_manifest,
                "--selection-manifest", self.selection, "--data-root", self.data_root,
                "--assignment-manifest", f"train={self.manifests['train']}",
                "--assignment-manifest", f"validation={self.manifests['validation']}",
                "--output-dir", output, "--replicate-seed", 1337,
                "--source-snapshot-sha256", self.spec["source_manifest_sha256"],
                "--assignment-lock-sha256", load_json(self.locks["assignment"])["content_hash"],
                "--device", "cuda",
            )
            if self.spec["mode"] == "smoke": command.append("--smoke")
            _run(command, repository=self.repository); return [output / "training_report.json"]
        if task_id == "shell_endpoint_qualification_lock":
            reports = {name: load_json(self.root / f"qualification/{name}/training_report.json") for name in QUALIFIERS}
            acknowledgement_path = self.root / "authorizations/endpoint_diagnostic_ack.json"
            acknowledgement = load_json(acknowledgement_path)
            assignment_hash = load_json(self.manifests["validation"])["content_hash"]
            miniature = load_json(self.root / "runtime/cache_miniature.json")
            validate_content_hash(
                miniature, expected_contract="HCWDL_CACHE_MINIATURE/v1",
                expected_schema_version=1,
            )
            qualifier_hashes = {
                name: report["content_hash"] for name, report in reports.items()
            }
            acknowledgement_hash = validate_diagnostic_acknowledgement(
                acknowledgement, campaign_spec_sha256=self.spec["content_hash"],
                assignment_manifest_sha256=assignment_hash,
                recipe_sha256=self.spec["recipe_sha256"],
                cache_miniature_sha256=miniature["content_hash"],
                qualifier_report_sha256=qualifier_hashes,
            )
            endpoint_invariants = miniature.get("endpoint_invariants")
            if not isinstance(endpoint_invariants, dict):
                raise ValueError("HCWDL cache miniature lacks endpoint invariant evidence")
            output = self.root / "qualification/qualification_report.json"
            report = build_qualification_report(
                reports, campaign_spec_sha256=self.spec["content_hash"],
                assignment_manifest_sha256=assignment_hash,
                recipe_sha256=self.spec["recipe_sha256"],
                endpoint_invariants=endpoint_invariants,
                shell_strata=compute_shell_strata(self.manifests["validation"], data_root=self.data_root),
                diagnostic_ack_sha256=acknowledgement_hash,
            )
            write_immutable_json(output, report)
            lock = create_shell_endpoint_qualification_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                recipe_lock=load_json(self.locks["recipe"]),
                qualification_report_sha256=report["content_hash"],
                assignment_manifest_sha256=assignment_hash,
                endpoint_invariants_passed=True,
            )
            write_immutable_json(self.locks["shell_endpoint_qualification"], lock)
            return [output, self.locks["shell_endpoint_qualification"]]
        if task_id.startswith("train_"):
            node_id = task_id.removeprefix("train_")
            output = self.root / f"training/{node_id}"
            _run(self._node_command(node_id, output=output, seed=1337), repository=self.repository)
            return [output / "training_report.json", output / "hcwdl_training_report.json"]
        if task_id == "screen_aggregate":
            reports = [load_json(self._training_report(node)) for node in NODE_REGISTRY]
            node_reports = [
                load_json(self._hcwdl_training_report(node)) for node in NODE_REGISTRY
            ]
            output = self.root / "reports/screen_aggregate.json"
            aggregate = build_screen_aggregate(
                reports, node_reports=node_reports,
                campaign_spec_sha256=self.spec["content_hash"],
                recipe_sha256=self.spec["recipe_sha256"],
                assignment_lock_sha256=load_json(self.locks["assignment"])["content_hash"],
            )
            write_immutable_json(output, aggregate); return [output]
        if task_id == "confirmation_registry_lock":
            screen = load_json(self.root / "reports/screen_aggregate.json")
            registry = build_confirmation_registry(
                screen, seeds=(11, 22, 33, 44, 55),
                include_label_only_warm_continuation=bool(
                    self.spec["include_label_only_warm_continuation"]
                ),
            )
            lock = create_confirmation_registry_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                qualification_lock=load_json(self.locks["shell_endpoint_qualification"]),
                screen_aggregate_sha256=screen["content_hash"], registry=registry,
            )
            write_immutable_json(self.locks["confirmation_registry"], lock)
            return [self.locks["confirmation_registry"]]
        if task_id == "confirmation":
            lock = load_json(self.locks["confirmation_registry"])
            row = lock["payload"]["registry"][_array_index()]
            output = self.root / f"confirmation/{_array_index():03d}_{row['node_id']}_seed{row['seed']}"
            if row["kind"] == "primary":
                _run(self._node_command(row["node_id"], output=output, seed=int(row["seed"])), repository=self.repository)
            else:
                teacher = {
                    "NULL_M1_SELF_KD": "M0",
                    "NULL_M6_PREDECESSOR_ONLY": "M5c",
                    "NULL_WARM_LABEL_ONLY": "M5w",
                }[row["node_id"]]
                command = self._script(
                    "train_hcwdl_control.py", "--control-id", row["node_id"],
                    "--recipe", self.recipe, "--split-manifest", self.split_manifest,
                    "--selection-manifest", self.selection, "--data-root", self.data_root,
                    "--output-dir", output, "--replicate-seed", row["seed"],
                    "--teacher-report", self._training_report(teacher),
                    "--source-snapshot-sha256", self.spec["source_manifest_sha256"],
                    "--assignment-lock-sha256", load_json(self.locks["assignment"])["content_hash"],
                    "--qualification-lock-sha256", load_json(self.locks["shell_endpoint_qualification"])["content_hash"],
                    "--device", "cuda",
                )
                if self.spec["mode"] == "smoke":
                    command.append("--smoke")
                _run(command, repository=self.repository)
            return [output / "training_report.json"]
        if task_id in {"finalist_lock", "execution_lock", "test_row_selection", "assign_test", "test_assignment_manifest", "sealed_final_evaluation", "aggregate_report"}:
            return self._run_final_stage(task_id)
        raise RuntimeError(f"registered HCWDL task has no dispatcher branch: {task_id}")

    def _run_final_stage(self, task_id: str) -> list[Path]:
        if task_id == "finalist_lock":
            confirmation = load_json(self.locks["confirmation_registry"])
            reports = []
            for index, row in enumerate(confirmation["payload"]["registry"]):
                path = self.root / f"confirmation/{index:03d}_{row['node_id']}_seed{row['seed']}/training_report.json"
                report = load_json(path); reports.append((row, report))
            finalist_nodes = {
                "M0", "M6c", "M6w",
                load_json(self.root / "reports/screen_aggregate.json")["selected_intermediate_cold"]["selected_node_id"],
                load_json(self.root / "reports/screen_aggregate.json")["selected_intermediate_warm"]["selected_node_id"],
                "NULL_M1_SELF_KD", "NULL_M6_PREDECESSOR_ONLY",
            }
            if self.spec["include_label_only_warm_continuation"]:
                finalist_nodes.add("NULL_WARM_LABEL_ONLY")
            finalists = [
                {"node_id": row["node_id"], "seed": row["seed"],
                 "checkpoint_sha256": report["selected_checkpoint_sha256"],
                 "report_sha256": report["content_hash"], "report_path": str(path)}
                for (row, report), path in zip(reports, [
                    self.root / f"confirmation/{index:03d}_{row['node_id']}_seed{row['seed']}/training_report.json"
                    for index, row in enumerate(confirmation["payload"]["registry"])
                ], strict=True) if row["node_id"] in finalist_nodes
            ]
            for oracle in ("D100", "TOFF"):
                report = load_json(self._training_report(oracle))
                finalists.append({
                    "node_id": oracle, "seed": 1337,
                    "checkpoint_sha256": report["selected_checkpoint_sha256"],
                    "report_sha256": report["content_hash"],
                    "report_path": str(self._training_report(oracle)),
                })
            confirmation_report = with_content_hash({
                "contract": "HCWDL_CONFIRMATION_AGGREGATE/v1", "schema_version": 1,
                "registry_lock_sha256": confirmation["content_hash"],
                "reports": [report["content_hash"] for _, report in reports],
                "finite_bad_performance_retained": True,
            })
            report_path = self.root / "reports/confirmation_aggregate.json"
            write_immutable_json(report_path, confirmation_report)
            lock = create_finalist_lock(
                campaign_spec_sha256=self.spec["content_hash"], confirmation_lock=confirmation,
                confirmation_report_sha256=confirmation_report["content_hash"], finalists=finalists,
            )
            write_immutable_json(self.locks["finalist"], lock); return [report_path, self.locks["finalist"]]
        if task_id == "execution_lock":
            selection_identity = canonical_sha256({
                "split_manifest_sha256": self.spec["split_manifest_sha256"],
                "role": "final_test", "rows": (
                    "all" if self.spec["role_counts"]["final_test"] is None
                    else self.spec["role_counts"]["final_test"]
                ),
                "rule": "per_class_smallest_identity_sha256_rank_v1",
                "seed": 1337,
            })
            lock = create_execution_lock(
                campaign_spec_sha256=self.spec["content_hash"],
                finalist_lock=load_json(self.locks["finalist"]),
                split_manifest_sha256=self.spec["split_manifest_sha256"],
                final_test_selection_rule_sha256=selection_identity,
                matcher_resources_sha256=load_json(self.resources)["content_hash"],
                recipe_sha256=self.spec["recipe_sha256"], source_commit=self.spec["source_commit"],
            )
            write_immutable_json(self.locks["execution"], lock); return [self.locks["execution"]]
        if task_id == "test_row_selection":
            counts = self.spec["role_counts"]["final_test"]
            _run(self._script(
                "build_hcwdl_row_selection.py", "--split-manifest", self.split_manifest,
                "--data-root", self.data_root, "--output", self.root / "source/test_row_selection.json",
                "--role-budget", f"final_test={'all' if counts is None else counts}",
                "--completed-lock", "finalist", "--completed-lock", "execution",
                "--access-lock", f"finalist={self.locks['finalist']}",
                "--access-lock", f"execution={self.locks['execution']}",
            ), repository=self.repository)
            return [self.root / "source/test_row_selection.json"]
        if task_id == "assign_test":
            selection = self.root / "source/test_row_selection.json"
            _run(self._script(
                "build_highcov_assignment_shard.py", "--split-manifest", self.split_manifest,
                "--selection-manifest", selection, "--resources-report", self.resources,
                "--data-root", self.data_root, "--assignment-root", self.assignment_root,
                "--role", "final_test", "--source-index", _array_index(),
                "--completed-lock", "finalist", "--completed-lock", "execution",
            ), repository=self.repository)
            base = self.assignment_root / "final_test" / f"shard_{_array_index():04d}"
            return [base.with_suffix(".npz"), base.with_suffix(".json")]
        if task_id == "test_assignment_manifest":
            selection = self.root / "source/test_row_selection.json"
            _run(self._script(
                "finalize_highcov_assignments.py", "--split-manifest", self.split_manifest,
                "--selection-manifest", selection, "--resources-report", self.resources,
                "--assignment-root", self.assignment_root, "--role", "final_test",
                "--output", self.manifests["final_test"],
            ), repository=self.repository)
            rows = int(load_json(self.manifests["final_test"])["scanned_mapped_jets"])
            _run(self._script(
                "audit_highcov_assignments.py", "--manifest", self.manifests["final_test"],
                "--split-manifest", self.split_manifest, "--data-root", self.data_root,
                "--role", "final_test", "--sample-size", min(256, rows), "--seed", 1337,
                "--completed-lock", "finalist", "--completed-lock", "execution",
                "--output", self.audits["final_test"],
            ), repository=self.repository)
            return [self.manifests["final_test"], self.audits["final_test"]]
        if task_id == "sealed_final_evaluation":
            output = self.root / "final_test/evaluation"
            _run(self._script(
                "evaluate_hcwdl_final.py", "--split-manifest", self.split_manifest,
                "--selection-manifest", self.root / "source/test_row_selection.json",
                "--test-assignment-manifest", self.manifests["final_test"],
                "--finalist-lock", self.locks["finalist"],
                "--execution-lock", self.locks["execution"],
                "--data-root", self.data_root, "--output-root", output,
                "--device", "cuda",
            ), repository=self.repository)
            return [output / "evaluation_manifest.json"]
        if task_id == "aggregate_report":
            output = self.root / "reports/campaign_complete.json"
            if self.spec["mode"] == "smoke":
                write_immutable_json(output, with_content_hash({
                    "contract": "HCWDL_CAMPAIGN_COMPLETION/v1", "schema_version": 1,
                    "campaign_spec_sha256": self.spec["content_hash"],
                    "mode": "smoke", "final_test_accessed": False,
                    "scientific_result_does_not_control_completion": True,
                }))
                return [output]
            screen = load_json(self.root / "reports/screen_aggregate.json")
            validation = {str(row["node_id"]): dict(row["validation"]) for row in screen["rows"]}
            finalist_groups: dict[str, list[dict[str, Any]]] = {}
            for row in load_json(self.locks["finalist"])["payload"]["finalists"]:
                training = load_json(row["report_path"])
                finalist_groups.setdefault(str(row["node_id"]), []).append(training["validation"])
            evaluation_manifest = load_json(self.root / "final_test/evaluation/evaluation_manifest.json")
            grouped: dict[str, list[dict[str, Any]]] = {}
            parent_hashes = {"evaluation_manifest": evaluation_manifest["content_hash"]}
            for index, record in enumerate(evaluation_manifest["reports"]):
                report = load_json(record["path"])
                grouped.setdefault(str(record["node_id"]), []).append(report["metrics"])
                parent_hashes[f"evaluation_{index:03d}"] = report["content_hash"]
            scalar_names = (
                "cross_entropy", "accuracy", "balanced_accuracy", "macro_ovr_auc",
                "macro_mean_log_qcd_rejection_at_50pct_signal", "top_label_ece_15_bin",
                "multiclass_brier_score", "always_qcd_accuracy",
            )
            for node, rows in finalist_groups.items():
                validation[node] = {
                    name: sum(float(row[name]) for row in rows) / len(rows)
                    for name in scalar_names
                }
            test = {
                node: {
                    name: sum(float(row[name]) for row in rows) / len(rows)
                    for name in scalar_names
                }
                for node, rows in grouped.items()
            }
            report = build_final_report(
                campaign_spec_sha256=self.spec["content_hash"],
                execution_lock_sha256=load_json(self.locks["execution"])["content_hash"],
                validation_metrics=validation, test_metrics=test,
                report_hashes=parent_hashes,
            )
            write_immutable_json(output, report); return [output]
        raise RuntimeError("unreachable HCWDL final stage")


__all__ = ["HcwdlWorkflow"]
