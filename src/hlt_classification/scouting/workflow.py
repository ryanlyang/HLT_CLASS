"""Configuration-driven PMARD task execution used by the thin Slurm worker."""

from __future__ import annotations

import json
from io import BytesIO
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import numpy as np

from hlt_classification.data.cache_contracts import atomic_publish_bytes, load_json, sha256_file, with_content_hash, write_immutable_json
from hlt_classification.provenance import validate_source_snapshot
from .campaign import validate_pmard_campaign_spec
from .config_contracts import validate_vendored_preprocessing
from .locks import claim_final_execution, create_lock
from .matcher_validation import select_matcher_variant
from .selection import select_alpha, select_budget, utility_key
from .training import REPRESENTATION_COEFFICIENT

PMARD_TASK_ATTESTATION_CONTRACT = "hlt_classification_pmard_task_attestation_v1"


def _run(command: Sequence[str], *, repository: Path) -> None:
    subprocess.run(list(command), cwd=repository, check=True)


def _script(repository: Path, name: str, *arguments: object) -> list[str]:
    return [sys.executable, "-s", str(repository / "scripts" / name), *map(str, arguments)]


def _array_id() -> int:
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None: raise RuntimeError("array task requires SLURM_ARRAY_TASK_ID")
    return int(value)


class Workflow:
    def __init__(self, spec: Mapping[str, Any], *, repository: Path) -> None:
        validate_pmard_campaign_spec(spec)
        validate_source_snapshot(spec["source_snapshot"], repository=repository, require_clean=True)
        self.spec = spec; self.repository = repository.resolve()
        self.root = Path(spec["campaign_root"]); self.data = Path(spec["site"]["data_root"])
        self.source = self.root / "data/source_manifest.json"
        self.split = self.root / "data/splits/split_manifest.json"
        self.audit = self.root / "data/feature_audit.json"
        self.data_lock = self.root / "locks/01_data.json"
        self.matcher_design_lock = self.root / "locks/02_matcher_design.json"
        self.matcher_result_lock = self.root / "locks/03_matcher_result.json"
        self.training_lock = self.root / "locks/04_training.json"
        self.full_matcher = self.root / "matcher/full/matcher_report.json"

    @property
    def source_snapshot_sha256(self) -> str:
        return self.spec["source_snapshot"]["source_snapshot_sha256"]

    def _fold_args(self) -> list[str]:
        result = []
        for fold in range(5):
            result.extend(("--matcher-fold-report", f"{fold}={self.root / f'matcher/fold_{fold}/matcher_report.json'}"))
        return result

    def _locked_training(self) -> Mapping[str, Any]:
        return load_json(self.training_lock)["payload"]

    def _matcher_settings(self) -> Mapping[str, Any]:
        return load_json(self.matcher_result_lock)["payload"]

    def _eligible_categories(self) -> str:
        settings = self._matcher_settings()
        eligible = [name for name, allowed in settings["category_eligibility"].items() if allowed]
        # Failed matching bounds disable a category; if all fail, run the conservative
        # charged-only arm as the predeclared scientific fallback rather than forcing coverage.
        if not eligible: eligible = ["0", "1", "2"]
        return ",".join(eligible)

    def _training_args(
        self, *, updates: int | None = None, lr: float | None = None,
        seed: int | None = None,
    ) -> list[object]:
        locked = self._locked_training()
        result = [
            "--total-updates", updates if updates is not None else locked["total_updates"],
            "--batch-size", locked["batch_size"],
            "--learning-rate", lr if lr is not None else locked["peak_learning_rate"],
            "--seed", locked.get("screen_seed", 1337) if seed is None else seed,
            "--device", "cuda",
        ]
        if self.spec["mode"] == "smoke": result.extend(("--max-rows-per-role", 4096))
        return result

    def _teacher_command(
        self, *, output: Path, experiment: str, alpha: float = 0.0,
        native_offline: bool = False, extra: Sequence[object] = (),
        updates: int | None = None, lr: float | None = None,
        seed: int | None = None,
    ) -> list[str]:
        command = _script(
            self.repository, "train_scouting_pmard_teacher.py",
            "--split-manifest", self.split, "--data-root", self.data,
            "--output-dir", output, "--experiment-id", experiment,
            "--alpha", alpha, "--class-counts", self.audit,
            "--source-snapshot-sha256", self.source_snapshot_sha256,
            *self._training_args(updates=updates, lr=lr, seed=seed), *extra,
        )
        if native_offline: command.append("--native-offline")
        if alpha:
            if "--matcher-variant" not in map(str, extra):
                matcher = self._matcher_settings()
                command.extend(("--matcher-variant", matcher["selected_variant"],
                                "--matcher-threshold", matcher["threshold"]))
            if "--eligible-categories" not in map(str, extra):
                command.extend(("--eligible-categories", self._eligible_categories()))
            command.extend(("--matcher-report", str(self.full_matcher), *self._fold_args()))
        return command

    def _student_command(
        self, *, output: Path, arm: str, alpha: float, hlt_teacher: Path | None,
        privileged_teacher: Path | None, extra: Sequence[object] = (), seed: int | None = None,
    ) -> list[str]:
        locked = self._locked_training()
        command = _script(
            self.repository, "train_scouting_pmard_student.py",
            "--split-manifest", self.split, "--data-root", self.data,
            "--output-dir", output, "--arm", arm, "--alpha", alpha,
            "--temperature", locked["temperature"], "--class-counts", self.audit,
            "--source-snapshot-sha256", self.source_snapshot_sha256,
            "--total-updates", locked["total_updates"], "--batch-size", locked["batch_size"],
            "--learning-rate", locked["peak_learning_rate"],
            "--seed", locked["screen_seed"] if seed is None else seed, "--device", "cuda",
            *extra,
        )
        if hlt_teacher is not None: command.extend(("--hlt-teacher-report", str(hlt_teacher)))
        if privileged_teacher is not None: command.extend(("--privileged-teacher-report", str(privileged_teacher)))
        if arm not in {"K0", "K1", "K6"} and alpha:
            if "--matcher-variant" not in map(str, extra):
                matcher = self._matcher_settings()
                command.extend(("--matcher-variant", matcher["selected_variant"],
                                "--matcher-threshold", matcher["threshold"]))
            if "--eligible-categories" not in map(str, extra):
                command.extend(("--eligible-categories", self._eligible_categories()))
            command.extend(("--matcher-report", str(self.full_matcher), *self._fold_args()))
        return command

    def _teacher_report(self, alpha: float) -> Path:
        if alpha == 0:
            selection = load_json(self.root / "training/budget_selection.json")
            return Path(selection["selected_report_path"])
        tag = {0.05: "T05", .1: "T10", .25: "T25", .5: "T50", 1.0: "T100"}[alpha]
        return self.root / f"training/teachers/{tag}/training_report.json"

    def run(self, task: str) -> list[Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        if task == "source_audit":
            _run(_script(self.repository, "validate_scouting_data.py", "--data-root", self.data, "--output", self.source), repository=self.repository)
            return [self.source]
        if task == "splits":
            _run(_script(self.repository, "build_scouting_splits.py", "--source-manifest", self.source, "--output-dir", self.split.parent), repository=self.repository)
            return [self.split]
        if task == "feature_audit":
            _run(_script(self.repository, "audit_scouting_features.py", "--split-manifest", self.split, "--data-root", self.data, "--output", self.audit), repository=self.repository)
            return [self.audit]
        if task == "data_lock":
            preprocessing = validate_vendored_preprocessing(self.repository)
            source = load_json(self.source); split = load_json(self.split); audit = load_json(self.audit)
            if source.get("content_hash") != self.spec["source_manifest_sha256"]:
                raise ValueError("executed source manifest differs from campaign specification")
            if split.get("content_hash") != self.spec["split_manifest_sha256"]:
                raise ValueError("executed split manifest differs from campaign specification")
            if audit.get("split_manifest_sha256") != split["content_hash"]:
                raise ValueError("feature audit split lineage differs")
            payload = create_lock(
                "data", campaign_spec_sha256=self.spec["content_hash"],
                payload={"source_manifest_sha256": source["content_hash"],
                         "split_manifest_sha256": split["content_hash"],
                         "feature_audit_sha256": audit["content_hash"],
                         "preprocessing_contract": preprocessing},
            ); write_immutable_json(self.data_lock, payload); return [self.data_lock]
        if task == "matcher_design_lock":
            payload = create_lock(
                "matcher_design", campaign_spec_sha256=self.spec["content_hash"],
                parent_lock=load_json(self.data_lock), payload={
                    "folds": 5, "variants": [f"M{i}" for i in range(6)],
                    "primary_candidate": "M4", "conservative_candidate": "M5",
                    "operating_points": {"ultra_pure": .99, "high_purity": .95, "high_coverage": .80},
                    "edge_feature_contract": "physics_context_type13_v1",
                    "solver": "uot_dustbin_plus_hungarian_control_v1",
                    "downstream_selection_forbidden": True,
                },
            ); write_immutable_json(self.matcher_design_lock, payload); return [self.matcher_design_lock]
        if task == "matcher_crossfit":
            fold = _array_id(); output = self.root / f"matcher/fold_{fold}"
            _run(_script(
                self.repository, "train_scouting_matcher.py", "--split-manifest", self.split,
                "--data-root", self.data, "--output-dir", output,
                "--source-snapshot-sha256", self.source_snapshot_sha256,
                "--holdout-fold", fold, "--device", "cuda", "--synthetic-jets",
                200 if self.spec["mode"] == "smoke" else 10000,
                *(("--native-max-jets", 20000) if self.spec["mode"] == "smoke" else ()),
            ), repository=self.repository)
            return [output / "matcher_report.json"]
        if task == "matcher_validation":
            outputs = []
            for fold in range(5):
                report = self.root / f"matcher/fold_{fold}/matcher_report.json"
                output = self.root / f"matcher/fold_{fold}/validation.json"
                _run(_script(
                    self.repository, "validate_scouting_matcher.py", "--split-manifest", self.split,
                    "--matcher-report", report, "--data-root", self.data, "--output", output,
                    "--holdout-fold", fold, "--device", "cuda",
                ), repository=self.repository); outputs.append(output)
            _run(_script(
                self.repository, "train_scouting_matcher.py", "--split-manifest", self.split,
                "--data-root", self.data, "--output-dir", self.full_matcher.parent,
                "--source-snapshot-sha256", self.source_snapshot_sha256, "--device", "cuda",
                "--synthetic-jets", 200 if self.spec["mode"] == "smoke" else 10000,
                *(("--native-max-jets", 20000) if self.spec["mode"] == "smoke" else ()),
            ), repository=self.repository)
            validation = self.root / "matcher/full/validation.json"
            _run(_script(
                self.repository, "validate_scouting_matcher.py", "--split-manifest", self.split,
                "--matcher-report", self.full_matcher, "--data-root", self.data,
                "--output", validation, "--role", "validation", "--device", "cuda",
            ), repository=self.repository); return [*outputs, self.full_matcher, validation]
        if task == "matcher_result_lock":
            validations = [load_json(self.root / f"matcher/fold_{fold}/validation.json") for fold in range(5)]
            full = load_json(self.root / "matcher/full/validation.json")
            selection = select_matcher_variant(validations)
            payload = create_lock(
                "matcher_result", campaign_spec_sha256=self.spec["content_hash"],
                parent_lock=load_json(self.matcher_design_lock), payload={
                    "selected_variant": selection["selected_variant"], "selected_operating_point": "ultra_pure",
                    "threshold": .99, "fold_validation_sha256": [row["content_hash"] for row in validations],
                    "full_matcher_report_sha256": load_json(self.full_matcher)["content_hash"],
                    "validation_report_sha256": full["content_hash"],
                    "meets_initial_precision_target": selection["selector_requirements_met"],
                    "category_eligibility": selection["category_eligibility"],
                    "matching_only_selection": selection["candidates"],
                },
            ); write_immutable_json(self.matcher_result_lock, payload); return [self.matcher_result_lock]
        if task == "weaver_parity":
            output = self.root / "runtime/weaver_parity.json"
            _run(_script(self.repository, "validate_scouting_weaver.py", "--repository", self.repository, "--device", "cuda", "--output", output), repository=self.repository)
            return [output]
        if task == "budget_grid":
            index = _array_id(); batches = (256, 512); rates = (1e-4, 3e-4, 1e-3); passes = (10, 20)
            batch, rate, exposure = [(b, r, p) for b in batches for r in rates for p in passes][index]
            train_rows = load_json(self.audit)["roles"]["train"]["mapped"]
            updates = 2 if self.spec["mode"] == "smoke" else math.ceil(train_rows / batch) * exposure
            output = self.root / f"training/budget_grid/{index:02d}"
            command = _script(
                self.repository, "train_scouting_pmard_teacher.py", "--split-manifest", self.split,
                "--data-root", self.data, "--output-dir", output,
                "--experiment-id", f"budget_b{batch}_lr{rate:g}_p{exposure}", "--alpha", 0,
                "--class-counts", self.audit, "--source-snapshot-sha256", self.source_snapshot_sha256,
                "--total-updates", updates, "--batch-size", batch, "--learning-rate", rate,
                "--seed", 1337, "--device", "cuda",
            )
            if self.spec["mode"] == "smoke": command.extend(("--max-rows-per-role", "4096"))
            _run(command, repository=self.repository); return [output / "training_report.json"]
        if task == "budget_selection":
            paths = [self.root / f"training/budget_grid/{index:02d}/training_report.json" for index in range(12)]
            selection = select_budget([load_json(path) for path in paths]); selected_hash = selection["selected_training_report_sha256"]
            selection["selected_report_path"] = str(next(path for path in paths if load_json(path)["content_hash"] == selected_hash))
            selection = with_content_hash({key: value for key, value in selection.items() if key != "content_hash"})
            output = self.root / "training/budget_selection.json"; write_immutable_json(output, selection); return [output]
        if task == "temperature_grid":
            index = _array_id(); tau = (1, 2, 4)[index]
            selected = load_json(self.root / "training/budget_selection.json")
            base_report = load_json(selected["selected_report_path"]); config = base_report["config"]
            output = self.root / f"training/temperature/tau_{tau}"
            command = _script(
                self.repository, "train_scouting_pmard_student.py", "--split-manifest", self.split,
                "--data-root", self.data, "--output-dir", output, "--arm", "K1", "--alpha", 0,
                "--temperature", tau, "--hlt-teacher-report", selected["selected_report_path"],
                "--class-counts", self.audit, "--source-snapshot-sha256", self.source_snapshot_sha256,
                "--total-updates", config["total_updates"], "--batch-size", config["effective_batch_size"],
                "--learning-rate", config["peak_learning_rate"], "--seed", 1337, "--device", "cuda",
            )
            if self.spec["mode"] == "smoke": command.extend(("--max-rows-per-role", "4096"))
            _run(command, repository=self.repository); return [output / "training_report.json"]
        if task == "training_lock":
            reports = [load_json(self.root / f"training/temperature/tau_{tau}/training_report.json") for tau in (1, 2, 4)]
            chosen = min(reports, key=lambda row: (row["validation"]["cross_entropy"], -row["validation"]["accuracy"], row["experiment_id"]))
            base_selection = load_json(self.root / "training/budget_selection.json"); base = load_json(base_selection["selected_report_path"])
            config = base["config"]
            parity = load_json(self.root / "runtime/weaver_parity.json")
            if parity.get("parity", {}).get("passed") is not True:
                raise RuntimeError("training lock requires passing installed-Weaver parity")
            payload = create_lock(
                "training", campaign_spec_sha256=self.spec["content_hash"],
                parent_lock=load_json(self.matcher_result_lock), payload={
                    "batch_size": config["effective_batch_size"], "peak_learning_rate": config["peak_learning_rate"],
                    "total_updates": config["total_updates"], "temperature": chosen["config"]["loss"]["temperature"],
                    "screen_seed": 1337, "confirmation_seeds": [11, 22, 33, 44, 55],
                    "representation_coefficient": REPRESENTATION_COEFFICIENT,
                    "baseline_report_path": base_selection["selected_report_path"],
                    "weaver_parity_sha256": parity["content_hash"],
                    "temperature_candidate_sha256": [row["content_hash"] for row in reports],
                },
            ); write_immutable_json(self.training_lock, payload); return [self.training_lock]
        return self._run_training_stage(task)

    def _run_training_stage(self, task: str) -> list[Path]:
        locked = self._locked_training(); baseline = Path(locked["baseline_report_path"])
        if task == "teachers":
            index = _array_id(); alpha_rows = (0.0, .05, .1, .25, .5, 1.0)
            if index == 0:
                output = self.root / "training/teachers/T0_reference.json"
                write_immutable_json(output, with_content_hash({
                    "contract": "hlt_classification_pmard_teacher_reference_v1", "schema_version": 1,
                    "training_report_sha256": load_json(baseline)["content_hash"], "training_report_path": str(baseline),
                })); return [output]
            if index == 6:
                output = self.root / "training/teachers/TOFF"
                _run(self._teacher_command(output=output, experiment="TOFF", native_offline=True), repository=self.repository)
                return [output / "training_report.json"]
            alpha = alpha_rows[index]; tag = {0.05: "T05", .1: "T10", .25: "T25", .5: "T50", 1.0: "T100"}[alpha]
            output = self.root / f"training/teachers/{tag}"
            _run(self._teacher_command(output=output, experiment=tag, alpha=alpha), repository=self.repository)
            return [output / "training_report.json"]
        if task == "k2_alpha_sweep":
            alpha = (0.0, .05, .1, .25, .5, 1.0)[_array_id()]
            output = self.root / f"training/k2_alpha/{alpha:g}"
            _run(self._student_command(
                output=output, arm="K2", alpha=alpha, hlt_teacher=baseline,
                privileged_teacher=self._teacher_report(alpha) if alpha else baseline,
            ), repository=self.repository); return [output / "training_report.json"]
        if task == "alpha_selection":
            paths = [self.root / f"training/k2_alpha/{alpha:g}/training_report.json" for alpha in (0.0, .05, .1, .25, .5, 1.0)]
            selection = select_alpha([load_json(path) for path in paths]); selection["selected_report_path"] = str(next(path for path in paths if load_json(path)["content_hash"] == selection["selected_report_sha256"]))
            selection = with_content_hash({key: value for key, value in selection.items() if key != "content_hash"})
            output = self.root / "training/alpha_selection.json"; write_immutable_json(output, selection); return [output]
        alpha_selection = load_json(self.root / "training/alpha_selection.json"); alpha = float(alpha_selection["selected_alpha"])
        privileged = self._teacher_report(alpha); toff = self.root / "training/teachers/TOFF/training_report.json"
        if task == "kd_controls":
            arm = f"K{_array_id()}"; output = self.root / f"training/kd_controls/{arm}"
            _run(self._student_command(
                output=output, arm=arm, alpha=alpha,
                hlt_teacher=baseline if arm not in {"K0", "K4"} else None,
                privileged_teacher=toff if arm == "K6" else privileged if arm in {"K2", "K3", "K4", "K5"} else None,
            ), repository=self.repository); return [output / "training_report.json"]
        if task == "mechanism_controls":
            controls = [
                ("M0", .99, "P4_ONLY", "0,1,2,3,4", 0.0), ("M1", .99, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M2", .99, "P4_ONLY", "0,1,2,3,4", 0.0), ("M3", .99, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M4", .99, "P4_ONLY", "0,1,2,3,4", 0.0), ("M5", .99, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M5", .95, "P4_ONLY", "0,1,2,3,4", 0.0), ("M5", .80, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M5", .99, "MATCH_SHUFFLED", "0,1,2,3,4", 0.0), ("M5", .99, "DIRECTION_ONLY", "0,1,2,3,4", 0.0),
                ("M5", .99, "RESPONSE_ONLY", "0,1,2,3,4", 0.0), ("M5", .99, "WRONG_DIRECTION", "0,1,2,3,4", 0.0),
                ("M5", .99, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M5", .99, "P4_ONLY", "0,1,2", 0.0), ("M5", .99, "P4_ONLY", "0,1,2,3,4", 0.0),
                ("M5", .99, "P4_ONLY", "0,1,2,3,4", .005), ("M5", .99, "P4_ONLY", "0,1,2,3,4", .01),
                ("M5", .99, "P4_ONLY", "0,1,2,3,4", .02), ("M5", .99, "P4_ONLY", "0,1,2,3,4", .05),
                ("M5", .99, "RANDOM_DIRECTION", "0,1,2,3,4", 0.0),
                ("M5", .99, "LOG_ANGULAR", "0,1,2,3,4", 0.0),
                ("M5", .99, "CONFIDENCE_WEIGHTED", "0,1,2,3,4", 0.0),
            ]
            index = _array_id(); variant, threshold, family, categories, corruption = controls[index]
            control_alpha = 0.0 if index == 12 else alpha
            root = self.root / f"training/mechanisms/{index:02d}"
            teacher_dir = root / "teacher"; extra = (
                "--matcher-variant", variant, "--matcher-threshold", threshold,
                "--repair-family", family, "--eligible-categories", categories,
                "--match-corruption-fraction", corruption,
            )
            _run(self._teacher_command(output=teacher_dir, experiment=f"mechanism_{index:02d}_teacher", alpha=control_alpha, extra=extra), repository=self.repository)
            _run(self._student_command(output=root / "student", arm="K2", alpha=control_alpha, hlt_teacher=baseline,
                                       privileged_teacher=teacher_dir / "training_report.json", extra=extra), repository=self.repository)
            return [teacher_dir / "training_report.json", root / "student/training_report.json"]
        if task == "representation":
            index = _array_id()
            if index == 0: return [Path(alpha_selection["selected_report_path"])]
            arm_index = index if index <= 5 else index - 5; arm = f"R{arm_index}"
            control = index > 5; output = self.root / f"training/representation/{index:02d}_{arm}{'_control' if control else ''}"
            extra: list[object] = ["--representation-arm", arm, "--representation-coefficient", 0 if control else REPRESENTATION_COEFFICIENT]
            if control: extra.append("--representation-control")
            _run(self._student_command(output=output, arm="K2", alpha=alpha, hlt_teacher=baseline,
                                       privileged_teacher=privileged, extra=extra), repository=self.repository)
            return [output / "training_report.json"]
        if task in {"generation_1", "generation_2"}:
            generation = int(task.rsplit("_", 1)[1])
            previous = Path(alpha_selection["selected_report_path"]) if generation == 1 else self.root / f"training/generations/{generation - 1}/student/training_report.json"
            root = self.root / f"training/generations/{generation}"
            fine_updates = max(1, locked["total_updates"] // 4); fine_lr = locked["peak_learning_rate"] / 10
            extra = ("--initialize-from", previous, "--anchor-teacher-report", previous, "--temperature", locked["temperature"])
            _run(self._teacher_command(output=root / "teacher", experiment=f"T{generation}", alpha=alpha,
                                       extra=extra, updates=fine_updates, lr=fine_lr), repository=self.repository)
            _run(self._student_command(output=root / "student", arm="K2", alpha=alpha, hlt_teacher=previous,
                                       privileged_teacher=root / "teacher/training_report.json",
                                       extra=("--generation", generation + 1)), repository=self.repository)
            _run(self._student_command(
                output=root / "same_generation_self_kd_control", arm="K1", alpha=0,
                hlt_teacher=previous, privileged_teacher=None,
                extra=("--generation", generation + 1),
            ), repository=self.repository)
            return [root / "teacher/training_report.json", root / "student/training_report.json",
                    root / "same_generation_self_kd_control/training_report.json"]
        if task == "screen_selection":
            candidates: list[tuple[str, Path, str]] = []
            for arm in (f"K{i}" for i in range(7)):
                candidates.append((arm, self.root / f"training/kd_controls/{arm}/training_report.json", "logit"))
            candidates.append(("B1_primary", Path(alpha_selection["selected_report_path"]), "logit"))
            for index in range(22):
                candidates.append((f"mechanism_{index:02d}", self.root / f"training/mechanisms/{index:02d}/student/training_report.json", "mechanism"))
            for index in range(1, 11):
                candidates.append((f"representation_{index:02d}", next((self.root / "training/representation").glob(f"{index:02d}_*/training_report.json")), "representation"))
            candidates.append(("B2", self.root / "training/generations/1/student/training_report.json", "generation"))
            candidates.append(("B3", self.root / "training/generations/2/student/training_report.json", "generation"))
            candidates.append(("B2_self_kd_control", self.root / "training/generations/1/same_generation_self_kd_control/training_report.json", "generation_control"))
            candidates.append(("B3_self_kd_control", self.root / "training/generations/2/same_generation_self_kd_control/training_report.json", "generation_control"))
            rows = [(name, path, category, load_json(path)) for name, path, category in candidates]
            best = min((row for row in rows if row[2] not in {"mechanism"}), key=lambda row: utility_key(row[3]))
            best_utility = float(best[3]["validation"]["macro_mean_log_qcd_rejection_at_50pct_signal"])
            mandatory_names = {"K0", "K1"}
            for category in ("logit", "representation", "generation"):
                category_rows = [row for row in rows if row[2] == category]
                if category_rows: mandatory_names.add(min(category_rows, key=lambda row: utility_key(row[3]))[0])
            promoted = []
            for name, path, category, report in rows:
                utility = float(report["validation"]["macro_mean_log_qcd_rejection_at_50pct_signal"])
                within = best_utility - utility <= .01 * max(abs(best_utility), 1.0e-12)
                if name in mandatory_names or (within and category not in {"mechanism", "generation_control"}):
                    promoted.append({
                        "graph_id": name, "category": category, "screen_report_path": str(path),
                        "screen_report_sha256": report["content_hash"],
                        "scientific_config": report.get("scientific_config", {}),
                    })
            output = self.root / "training/screen_selection.json"
            write_immutable_json(output, with_content_hash({
                "contract": "hlt_classification_pmard_screen_selection_v1", "schema_version": 1,
                "promotion_window_relative": .01, "best_graph_id": best[0],
                "mandatory_graph_ids": sorted(mandatory_names), "promoted": promoted,
                "candidate_report_sha256": [report["content_hash"] for _, _, _, report in rows],
            })); return [output]
        if task == "screen_confirmation_lock":
            selection = load_json(self.root / "training/screen_selection.json")
            output = self.root / "locks/05_screen_confirmation.json"
            payload = create_lock(
                "screen_confirmation", campaign_spec_sha256=self.spec["content_hash"],
                parent_lock=load_json(self.training_lock),
                payload={"selection_sha256": selection["content_hash"],
                         "promoted": selection["promoted"], "all_registered_rows_complete": True},
            ); write_immutable_json(output, payload); return [output]
        if task == "confirmation":
            seed = (11, 22, 33, 44, 55)[_array_id()]
            return [self._run_confirmation(seed)]
        if task == "miniature_summary":
            indexes = [load_json(self.root / f"training/confirmation/seed_{seed}/confirmation_index.json")
                       for seed in (11, 22, 33, 44, 55)]
            output = self.root / "reports/miniature_summary.json"
            write_immutable_json(output, with_content_hash({
                "contract": "hlt_classification_pmard_smoke_summary_v1", "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "confirmation_index_sha256": [row["content_hash"] for row in indexes],
                "final_test_accessed": False, "complete": True,
            })); return [output]
        if task == "finalist_lock":
            return [self._create_finalist_lock()]
        if task == "execution_lock":
            finalist = load_json(self.root / "locks/06_finalist.json")
            output = self.root / "locks/07_execution.json"
            payload = create_lock(
                "execution", campaign_spec_sha256=self.spec["content_hash"],
                parent_lock=finalist, payload={
                    "source_snapshot_sha256": self.source_snapshot_sha256,
                    "split_manifest_sha256": load_json(self.split)["content_hash"],
                    "finalist_selection_sha256": finalist["payload"]["selection_sha256"],
                    "job_dag_sha256": self.spec["content_hash"],
                },
            ); write_immutable_json(output, payload); return [output]
        if task == "final_test":
            return self._run_final_test()
        if task == "aggregate_report":
            report = self._aggregate_final_report()
            return [report, report.parent / "final_primary_metrics.png"]
        raise ValueError(f"PMARD task {task!r} has no workflow implementation")

    def _view_extra(self, scientific: Mapping[str, Any]) -> list[object]:
        matcher = self._matcher_settings()
        return [
            "--matcher-variant", scientific.get("matcher_variant", matcher["selected_variant"]),
            "--matcher-threshold", scientific.get("matcher_threshold", matcher["threshold"]),
            "--repair-family", scientific.get("repair_family", "P4_ONLY"),
            "--eligible-categories", scientific.get("eligible_categories", self._eligible_categories()),
            "--match-corruption-fraction", scientific.get("match_corruption_fraction", 0.0),
        ]

    def _run_confirmation(self, seed: int) -> Path:
        selection = load_json(self.root / "training/screen_selection.json")
        root = self.root / f"training/confirmation/seed_{seed}"
        baseline_dir = root / "B0"
        _run(self._teacher_command(output=baseline_dir, experiment=f"B0_seed{seed}", seed=seed), repository=self.repository)
        baseline = baseline_dir / "training_report.json"
        produced = []
        teacher_cache: dict[str, Path] = {}
        generation_cache: dict[int, Path] = {}
        for row in selection["promoted"]:
            graph_id = row["graph_id"]; scientific = row["scientific_config"]
            arm = scientific.get("arm", "K2"); alpha = float(scientific.get("alpha", 0.0))
            if graph_id == "K0" or arm == "K0":
                produced.append({"graph_id": graph_id, "report_path": str(baseline),
                                 "report_sha256": load_json(baseline)["content_hash"]}); continue
            requested_generation = int(scientific.get("generation", 0))
            if requested_generation > 1 and arm == "K2":
                if 1 not in generation_cache:
                    key = json.dumps({"alpha": alpha, "generation_base": True}, sort_keys=True)
                    teacher_dir = root / "generations/T0"
                    _run(self._teacher_command(
                        output=teacher_dir, experiment=f"T0_seed{seed}", alpha=alpha,
                        extra=self._view_extra(scientific), seed=seed,
                    ), repository=self.repository)
                    teacher_cache[key] = teacher_dir / "training_report.json"
                    b1 = root / "generations/B1"
                    _run(self._student_command(
                        output=b1, arm="K2", alpha=alpha, hlt_teacher=baseline,
                        privileged_teacher=teacher_cache[key], extra=(*self._view_extra(scientific), "--generation", 1), seed=seed,
                    ), repository=self.repository); generation_cache[1] = b1 / "training_report.json"
                while max(generation_cache) < requested_generation:
                    generation = max(generation_cache); previous = generation_cache[generation]
                    companion = root / f"generations/T{generation}"
                    _run(self._teacher_command(
                        output=companion, experiment=f"T{generation}_seed{seed}", alpha=alpha,
                        extra=(*self._view_extra(scientific), "--initialize-from", previous,
                               "--anchor-teacher-report", previous, "--temperature", self._locked_training()["temperature"]),
                        updates=max(1, self._locked_training()["total_updates"] // 4),
                        lr=self._locked_training()["peak_learning_rate"] / 10, seed=seed,
                    ), repository=self.repository)
                    next_generation = generation + 1; student_dir = root / f"generations/B{next_generation}"
                    _run(self._student_command(
                        output=student_dir, arm="K2", alpha=alpha, hlt_teacher=previous,
                        privileged_teacher=companion / "training_report.json",
                        extra=(*self._view_extra(scientific), "--generation", next_generation), seed=seed,
                    ), repository=self.repository); generation_cache[next_generation] = student_dir / "training_report.json"
                report_path = generation_cache[requested_generation]
                produced.append({"graph_id": graph_id, "report_path": str(report_path),
                                 "report_sha256": load_json(report_path)["content_hash"]}); continue
            privilege = None
            if arm == "K6":
                key = "TOFF"
                if key not in teacher_cache:
                    teacher_dir = root / "teachers/TOFF"
                    _run(self._teacher_command(output=teacher_dir, experiment=f"TOFF_seed{seed}", native_offline=True, seed=seed), repository=self.repository)
                    teacher_cache[key] = teacher_dir / "training_report.json"
                privilege = teacher_cache[key]
            elif arm not in {"K0", "K1"}:
                key = json.dumps({"alpha": alpha, **{name: scientific.get(name) for name in ("matcher_variant", "matcher_threshold", "repair_family")}}, sort_keys=True)
                if key not in teacher_cache:
                    teacher_dir = root / "teachers" / f"view_{len(teacher_cache):02d}"
                    _run(self._teacher_command(
                        output=teacher_dir, experiment=f"teacher_seed{seed}_{len(teacher_cache):02d}",
                        alpha=alpha, extra=self._view_extra(scientific), seed=seed,
                    ), repository=self.repository); teacher_cache[key] = teacher_dir / "training_report.json"
                privilege = teacher_cache[key]
            output = root / "students" / graph_id
            extra = self._view_extra(scientific)
            representation = scientific.get("representation_arm", "R0")
            if representation != "R0":
                extra.extend(("--representation-arm", representation,
                              "--representation-coefficient", scientific.get("representation_coefficient", 0)))
                if scientific.get("representation_control"): extra.append("--representation-control")
            _run(self._student_command(
                output=output, arm=arm, alpha=alpha,
                hlt_teacher=baseline if arm not in {"K0", "K4"} else None,
                privileged_teacher=privilege, extra=extra, seed=seed,
            ), repository=self.repository)
            report_path = output / "training_report.json"; produced.append({
                "graph_id": graph_id, "report_path": str(report_path),
                "report_sha256": load_json(report_path)["content_hash"],
            })
        index_path = root / "confirmation_index.json"
        write_immutable_json(index_path, with_content_hash({
            "contract": "hlt_classification_pmard_confirmation_index_v1", "schema_version": 1,
            "seed": seed, "screen_selection_sha256": selection["content_hash"], "graphs": produced,
        })); return index_path

    def _create_finalist_lock(self) -> Path:
        indexes = [load_json(self.root / f"training/confirmation/seed_{seed}/confirmation_index.json") for seed in (11, 22, 33, 44, 55)]
        graph_ids = set.intersection(*(set(row["graph_id"] for row in index["graphs"]) for index in indexes))
        summaries = []
        for graph_id in sorted(graph_ids):
            reports = []
            for index in indexes:
                item = next(row for row in index["graphs"] if row["graph_id"] == graph_id)
                reports.append(load_json(item["report_path"]))
            keys = ("macro_mean_log_qcd_rejection_at_50pct_signal", "macro_ovr_auc", "cross_entropy", "top_label_ece_15_bin", "accuracy")
            means = {key: sum(float(row["validation"][key]) for row in reports) / len(reports) for key in keys}
            summaries.append({"experiment_id": graph_id, "validation": means,
                              "seed_reports": [{"path": next(item["report_path"] for item in index["graphs"] if item["graph_id"] == graph_id),
                                                "sha256": report["content_hash"]} for index, report in zip(indexes, reports, strict=True)]})
        utility = min(summaries, key=utility_key)
        ce = min(summaries, key=lambda row: (row["validation"]["cross_entropy"], -row["validation"]["accuracy"], row["experiment_id"]))
        selected = [utility] + ([] if ce["experiment_id"] == utility["experiment_id"] else [ce])
        k1 = next(row for row in summaries if row["experiment_id"] == "K1")
        evaluation_graphs = selected + ([] if any(row["experiment_id"] == "K1" for row in selected) else [k1])
        selection_path = self.root / "training/finalist_selection.json"
        write_immutable_json(selection_path, with_content_hash({
            "contract": "hlt_classification_pmard_finalist_selection_v1", "schema_version": 1,
            "selected": selected, "evaluation_graphs": evaluation_graphs,
            "required_null_control": "K1", "all_confirmation_summaries": summaries,
        }))
        output = self.root / "locks/06_finalist.json"
        payload = create_lock(
            "finalist", campaign_spec_sha256=self.spec["content_hash"],
            parent_lock=load_json(self.root / "locks/05_screen_confirmation.json"),
            payload={"selection_sha256": load_json(selection_path)["content_hash"],
                     "selected": selected, "evaluation_graphs": evaluation_graphs,
                     "required_null_control": "K1"},
        ); write_immutable_json(output, payload); return output

    def _run_final_test(self) -> list[Path]:
        execution = load_json(self.root / "locks/07_execution.json")
        claim_path = self.root / "locks/final_test_claim.json"
        claim_final_execution(
            claim_path, execution_lock=execution,
            final_test_manifest_sha256=load_json(self.split)["content_hash"],
        )
        selection = load_json(self.root / "training/finalist_selection.json"); outputs = [claim_path]
        for finalist in selection["evaluation_graphs"]:
            for seed_report in finalist["seed_reports"]:
                report_path = Path(seed_report["path"]); seed = load_json(report_path)["config"]["master_seed"]
                output = self.root / f"final_test/{finalist['experiment_id']}/seed_{seed}"
                _run(_script(
                    self.repository, "evaluate_scouting_pmard.py", "--split-manifest", self.split,
                    "--data-root", self.data, "--training-report", report_path,
                    "--role", "final_test", "--output-dir", output,
                    "--execution-lock", self.root / "locks/07_execution.json", "--device", "cuda",
                ), repository=self.repository); outputs.append(output / "evaluation_report.json")
        return outputs

    def _aggregate_final_report(self) -> Path:
        selection = load_json(self.root / "training/finalist_selection.json")
        rows = []
        for finalist in selection["evaluation_graphs"]:
            for seed_report in finalist["seed_reports"]:
                seed = load_json(seed_report["path"])["config"]["master_seed"]
                evaluation = load_json(self.root / f"final_test/{finalist['experiment_id']}/seed_{seed}/evaluation_report.json")
                rows.append({"graph_id": finalist["experiment_id"], "seed": seed,
                             "evaluation_sha256": evaluation["content_hash"], "metrics": evaluation["metrics"]})
        from .statistics import dependence_sensitivity, load_prediction_set, nested_paired_intervals
        by_graph = {graph["experiment_id"]: {} for graph in selection["evaluation_graphs"]}
        for graph in selection["evaluation_graphs"]:
            for seed_report in graph["seed_reports"]:
                seed = load_json(seed_report["path"])["config"]["master_seed"]
                by_graph[graph["experiment_id"]][seed] = load_prediction_set(
                    self.root / f"final_test/{graph['experiment_id']}/seed_{seed}/predictions.npz"
                )
        intervals = {
            finalist["experiment_id"]: nested_paired_intervals(
                by_graph[finalist["experiment_id"]], by_graph["K1"], replicates=10_000,
            )
            for finalist in selection["selected"]
        }
        dependence = {
            finalist["experiment_id"]: dependence_sensitivity(
                by_graph[finalist["experiment_id"]], by_graph["K1"],
            )
            for finalist in selection["selected"]
        }
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        graph_names = [graph["experiment_id"] for graph in selection["evaluation_graphs"]]
        ce_means = [np.mean([row["metrics"]["cross_entropy"] for row in rows if row["graph_id"] == name])
                    for name in graph_names]
        rejection_means = [np.mean([row["metrics"]["macro_mean_log_qcd_rejection_at_50pct_signal"]
                                    for row in rows if row["graph_id"] == name]) for name in graph_names]
        figure, axes = plt.subplots(1, 2, figsize=(max(7, len(graph_names) * 1.4), 3.5))
        axes[0].bar(graph_names, ce_means); axes[0].set_ylabel("cross entropy")
        axes[1].bar(graph_names, rejection_means); axes[1].set_ylabel("macro mean log QCD rejection")
        for axis in axes: axis.tick_params(axis="x", rotation=45)
        figure.tight_layout(); stream = BytesIO(); figure.savefig(stream, format="png", dpi=160); plt.close(figure)
        plot_path = self.root / "reports/final_primary_metrics.png"; atomic_publish_bytes(plot_path, stream.getvalue())
        report = with_content_hash({
            "contract": "hlt_classification_pmard_final_report_v1", "schema_version": 1,
            "finalist_selection_sha256": selection["content_hash"], "rows": rows,
            "paired_nested_bootstrap": intervals,
            "dependence_sensitivity": dependence,
            "plots": [{"file": plot_path.name, "sha256": sha256_file(plot_path)}],
            "positive_evidence_rule": "macro_log_rejection_lower_above_zero_and_ce_upper_below_zero",
        })
        output = self.root / "reports/final_report.json"; write_immutable_json(output, report); return output


def write_task_attestation(
    *, spec: Mapping[str, Any], task: str, array_id: str | None,
    outputs: Sequence[Path], campaign_root: Path,
) -> Path:
    rows = []
    for path in outputs:
        if not path.is_file(): raise FileNotFoundError(f"task output is absent: {path}")
        rows.append({"path": str(path), "sha256": sha256_file(path)})
    payload = with_content_hash({
        "contract": PMARD_TASK_ATTESTATION_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "task": task,
        "array_task_id": array_id, "outputs": rows, "complete": True,
    })
    suffix = "" if array_id is None else f"_{array_id}"
    path = campaign_root / "task_attestations" / f"{task}{suffix}.json"
    write_immutable_json(path, payload); return path


__all__ = ["Workflow", "write_task_attestation"]
