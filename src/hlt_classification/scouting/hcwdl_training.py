"""HCWDL node planning, initialization, loss wiring, and engine adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)

from .engine import PmardTrainingConfig, train_pmard
from .hcwdl_ladder import DOMAINS, GRAPH_SHA256, NODE_REGISTRY, NodeSpec
from .hcwdl_parent_loss import (
    HCWDL_PARENT_BASE_LOSS_CONTRACT,
    HCWDL_PARENT_LOSS_SEMANTICS,
)
from .hcwdl_recipe import validate_recipe
from .targets import EphemeralTeacherTargets
from .training import LossConfiguration, derive_seed


TRAINING_REPORT_CONTRACT = "HCWDL_TRAINING_REPORT/v1"
CHECKPOINT_SELECTION_CONTRACT = "HCWDL_CHECKPOINT_SELECTION/v1"


def _loss_semantics_payload() -> dict[str, Any]:
    semantics = dict(HCWDL_PARENT_LOSS_SEMANTICS)
    return {
        "loss_semantics_contract": HCWDL_PARENT_BASE_LOSS_CONTRACT,
        "loss_semantics": semantics,
        "loss_semantics_sha256": canonical_sha256(semantics),
    }


def validate_hcwdl_training_report(value: Mapping[str, Any]) -> str:
    """Validate a corrected parent rerun without reinterpreting old reports."""

    digest = validate_content_hash(
        value, expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    expected = _loss_semantics_payload()
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError("HCWDL training report loss semantics differ")
    require_sha256(
        value.get("pmard_execution_config_sha256"),
        name="PMARD execution-config SHA-256",
    )
    return digest


def _reject_smoke_fields(value: object, *, path: str) -> None:
    """Reject even null smoke-only fields from full parent artifacts."""

    if isinstance(value, Mapping):
        for raw_name, item in value.items():
            name = str(raw_name)
            if "smoke" in name.lower():
                raise ValueError(f"full HCWDL parent contains smoke field: {path}.{name}")
            _reject_smoke_fields(item, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_smoke_fields(item, path=f"{path}[{index}]")


def validate_hcwdl_parent_prefix_campaign(
    value: Mapping[str, Any], *, executable: bool = True,
) -> str:
    """Require the exact non-smoke v8 authority that ends at finalist lock."""

    from .hcwdl_authorization import (
        AUTOMATIC_ENDPOINT_CONTINUATION,
        PARENT_PREFIX_SCOPE,
    )
    from .hcwdl_campaign import (
        PARENT_PREFIX_CAMPAIGN_CONTRACT,
        validate_campaign_spec,
    )

    if value.get("contract") != PARENT_PREFIX_CAMPAIGN_CONTRACT:
        raise ValueError(
            "HCWDL representation import requires HCWDL_CAMPAIGN_SPEC/v8"
        )
    digest = validate_campaign_spec(value, executable=executable)
    expected = {
        "schema_version": 8,
        "execution_scope": PARENT_PREFIX_SCOPE,
        "terminal_task_id": "finalist_lock",
        "training_passes": 60,
        "validation_every_passes": 1,
        "execution_lock_authorized": False,
        "final_test_access_authorized": False,
        "registered_final_test_tasks": 0,
        "endpoint_continuation": AUTOMATIC_ENDPOINT_CONTINUATION,
    }
    if value.get("mode") == "smoke" or any(
        value.get(name) != expected_value
        for name, expected_value in expected.items()
    ):
        raise PermissionError(
            "HCWDL representation import requires the exact non-smoke "
            "60-pass parent-prefix authority"
        )
    return digest


def _full_parent_execution_config_sha256(
    config: Mapping[str, Any], scientific: Mapping[str, Any],
) -> str:
    semantics = _loss_semantics_payload()
    return canonical_sha256({
        "training": dict(config),
        "scientific": dict(scientific),
        "explicit_loss_semantics": semantics,
    })


def validate_hcwdl_full_parent_engine_report(
    value: Mapping[str, Any], *, train_rows: int,
    recipe: Mapping[str, Any], expected_experiment_id: str | None = None,
    expected_exact_config: Mapping[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> str:
    """Validate one genuine 60-pass parent engine report and checkpoints.

    The generic PMARD report validator intentionally accepts historical and
    smoke reports.  Parent import is a narrower scientific boundary: every
    qualifier, screen, and confirmation report must be a current engine row,
    complete exactly sixty validation passes, and bind both selected and final
    checkpoint bytes to the corrected HCWDL loss implementation.
    """

    import torch

    from .engine import (
        PMARD_TRAINING_REPORT_CONTRACT,
        PMARD_TRAINING_REPORT_VERSION,
        validate_pmard_training_report,
    )

    if isinstance(train_rows, bool) or not isinstance(train_rows, int) or train_rows <= 0:
        raise ValueError("full HCWDL parent train-row count differs")
    recipe_sha256 = validate_recipe(recipe, require_authorized=True)
    if (
        value.get("contract") != PMARD_TRAINING_REPORT_CONTRACT
        or value.get("schema_version") != PMARD_TRAINING_REPORT_VERSION
    ):
        raise ValueError("full HCWDL parent requires the current PMARD report contract")
    digest = validate_pmard_training_report(value)
    _reject_smoke_fields(value, path="report")

    config = value.get("config")
    scientific = value.get("scientific_config")
    if not isinstance(config, Mapping) or not isinstance(scientific, Mapping):
        raise ValueError("full HCWDL parent configuration differs")
    batch = int(recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(np.ceil(train_rows / batch))
    total_updates = 60 * updates_per_pass
    required_config = {
        "total_updates": total_updates,
        "effective_batch_size": batch,
        "microbatch_size": int(recipe["batching"]["microbatch_size"]),
        "gradient_accumulation": int(recipe["batching"]["gradient_accumulation"]),
        "adam_epsilon": float(recipe["optimizer"]["epsilon"]),
        "weight_decay": float(recipe["optimizer"]["weight_decay"]),
        "warmup_fraction": float(recipe["schedule"]["warmup_fraction"]),
        "minimum_lr_fraction": float(recipe["schedule"]["minimum_lr_fraction"]),
        "validation_interval": updates_per_pass,
        "validation_checks": 60,
        "logging_interval": max(1, updates_per_pass // 4),
        "amp_dtype": str(recipe["amp_dtype"]),
        "selection_policy": "hcwdl_macro_auc",
    }
    if any(config.get(name) != expected for name, expected in required_config.items()):
        raise ValueError("full HCWDL parent 60-pass execution configuration differs")
    if expected_exact_config is not None and dict(config) != dict(expected_exact_config):
        raise ValueError("full HCWDL parent exact experiment configuration differs")
    if (
        expected_experiment_id is not None
        and value.get("experiment_id") != expected_experiment_id
    ):
        raise ValueError("full HCWDL parent experiment identity differs")
    expected_semantics = _loss_semantics_payload()
    expected_scientific = {
        "campaign": "HCWDL",
        "training_passes": 60,
        "validation_every_passes": 1,
        "performance_early_stopping": False,
        **expected_semantics,
    }
    if any(
        scientific.get(name) != expected
        for name, expected in expected_scientific.items()
    ):
        raise ValueError("full HCWDL parent scientific configuration differs")
    if any(value.get(name) != expected for name, expected in expected_semantics.items()):
        raise ValueError("full HCWDL parent loss-semantics binding differs")
    execution_sha256 = _full_parent_execution_config_sha256(config, scientific)
    if value.get("execution_config_sha256") != execution_sha256:
        raise ValueError("full HCWDL parent execution-config hash differs")
    if (
        value.get("complete") is not True
        or value.get("updates") != total_updates
        or value.get("performance_early_termination") is not False
    ):
        raise ValueError("full HCWDL parent did not complete its exact update budget")

    history = value.get("validation_history")
    if not isinstance(history, list) or len(history) != 60:
        raise ValueError("full HCWDL parent requires exactly 60 validation records")
    expected_updates = [updates_per_pass * index for index in range(1, 61)]
    if any(not isinstance(row, Mapping) for row in history) or [
        row.get("update") for row in history
    ] != expected_updates:
        raise ValueError("full HCWDL parent validation boundaries differ")
    selection = select_checkpoint(history)
    if (
        value.get("selected_update") != selection["selected_update"]
        or value.get("selected_update") not in expected_updates
    ):
        raise ValueError("full HCWDL parent checkpoint selection differs")
    selected_row = next(
        row for row in history if row["update"] == selection["selected_update"]
    )
    selected_metrics = {
        name: item for name, item in selected_row.items() if name != "update"
    }
    expected_metric_hex = {
        "selected_cross_entropy_hex": float(
            selected_row["cross_entropy"]
        ).hex(),
        "selected_accuracy_hex": float(selected_row["accuracy"]).hex(),
        "selected_macro_ovr_auc_hex": float(
            selected_row["macro_ovr_auc"]
        ).hex(),
        "selected_macro_mean_log_qcd_rejection_at_50pct_signal_hex": float(
            selected_row["macro_mean_log_qcd_rejection_at_50pct_signal"]
        ).hex(),
    }
    if value.get("validation") != selected_metrics or any(
        value.get(name) != item for name, item in expected_metric_hex.items()
    ):
        raise ValueError("full HCWDL parent selected validation metrics differ")
    if (
        value.get("selected_checkpoint") != "selected_model.pt"
        or value.get("final_checkpoint") != "final_model.pt"
    ):
        raise ValueError("full HCWDL parent selected/final checkpoint names differ")
    selected_sha256 = require_sha256(
        value.get("selected_checkpoint_sha256"), name="full parent selected checkpoint",
    )
    final_sha256 = require_sha256(
        value.get("final_checkpoint_sha256"), name="full parent final checkpoint",
    )

    if report_path is not None:
        registered = Path(report_path)
        if not registered.is_file() or registered.is_symlink():
            raise FileNotFoundError("full HCWDL parent engine report is absent or a symlink")
        registered = registered.resolve()
        if load_json(registered) != dict(value):
            raise ValueError("full HCWDL parent engine report bytes differ from validation input")

        def checkpoint_payload(name: str, expected_sha256: str) -> Mapping[str, Any]:
            if Path(name).name != name:
                raise ValueError("full HCWDL parent checkpoint path escapes its report directory")
            path = registered.parent / name
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"full HCWDL parent checkpoint is absent: {path}")
            if sha256_file(path) != expected_sha256:
                raise ValueError("full HCWDL parent checkpoint byte hash differs")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError("full HCWDL parent checkpoint payload differs")
            _reject_smoke_fields(payload, path=f"checkpoint.{name}")
            if (
                payload.get("config") != dict(config)
                or payload.get("scientific_config") != dict(scientific)
                or payload.get("execution_config_sha256") != execution_sha256
                or any(
                    payload.get(field) != expected
                    for field, expected in expected_semantics.items()
                )
            ):
                raise ValueError("full HCWDL parent checkpoint authority differs")
            return payload

        selected = checkpoint_payload("selected_model.pt", selected_sha256)
        final = checkpoint_payload("final_model.pt", final_sha256)
        if selected.get("selected_update") != value.get("selected_update"):
            raise ValueError("full HCWDL selected checkpoint update differs")
        if final.get("final_update") != total_updates:
            raise ValueError("full HCWDL final checkpoint is not the completed update")
    return digest


def validate_hcwdl_full_parent_wrapper_report(
    value: Mapping[str, Any], *, training_report_path: str | Path,
    train_rows: int, recipe: Mapping[str, Any], expected_node_id: str,
    expected_replicate_seed: int = 1337,
) -> dict[str, Any]:
    """Cross-bind one HCWDL wrapper to a strict full engine execution."""

    wrapper_path = Path(training_report_path)
    if not wrapper_path.is_file() or wrapper_path.is_symlink():
        raise FileNotFoundError("full HCWDL parent wrapper is absent or a symlink")
    wrapper_path = wrapper_path.resolve()
    if load_json(wrapper_path) != dict(value):
        raise ValueError("full HCWDL parent wrapper bytes differ from validation input")
    wrapper_sha256 = validate_hcwdl_training_report(value)
    _reject_smoke_fields(value, path="wrapper")
    engine_path = wrapper_path.parent / "training_report.json"
    engine = load_json(engine_path)
    engine_sha256 = validate_hcwdl_full_parent_engine_report(
        engine, train_rows=train_rows, recipe=recipe,
        expected_experiment_id=expected_node_id,
        expected_exact_config=asdict(node_training_config(
            expected_node_id, recipe, train_rows=train_rows,
            replicate_seed=expected_replicate_seed,
        )),
        report_path=engine_path,
    )
    recipe_sha256 = validate_recipe(recipe, require_authorized=True)
    history = engine["validation_history"]
    if (
        value.get("node_id") != expected_node_id
        or value.get("complete") is not True
        or value.get("recipe_sha256") != recipe_sha256
        or not isinstance(value.get("parents"), Mapping)
        or value["parents"].get("recipe") != recipe_sha256
        or value.get("parents") != engine.get("parents")
        or value.get("pmard_engine_report_sha256") != engine_sha256
        or value.get("pmard_execution_config_sha256")
        != engine.get("execution_config_sha256")
        or value.get("selected_checkpoint_sha256")
        != engine.get("selected_checkpoint_sha256")
        or value.get("final_checkpoint_sha256")
        != engine.get("final_checkpoint_sha256")
        or value.get("selection") != select_checkpoint(history)
    ):
        raise ValueError(f"full HCWDL parent wrapper differs: {expected_node_id}")
    return {
        "wrapper_sha256": wrapper_sha256,
        "engine_sha256": engine_sha256,
        "engine_path": engine_path.resolve(),
        "engine": dict(engine),
        "execution_config_sha256": engine["execution_config_sha256"],
        "selected_checkpoint_sha256": engine["selected_checkpoint_sha256"],
        "final_checkpoint_sha256": engine["final_checkpoint_sha256"],
        "completed_updates": engine["updates"],
        "validation_record_count": len(history),
    }


def checkpoint_selection_key(metrics: Mapping[str, object], update: int) -> tuple[object, ...]:
    required = (
        "macro_ovr_auc", "cross_entropy",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    )
    values = [float(metrics[name]) for name in required]
    if not np.isfinite(values).all():
        raise FloatingPointError("HCWDL checkpoint metric is nonfinite")
    return (-values[0], values[1], -values[2], int(update))


def select_checkpoint(records: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("HCWDL checkpoint selection requires validation records")
    if any("update" not in row for row in rows):
        raise ValueError("HCWDL validation record lacks update")
    selected = min(rows, key=lambda row: checkpoint_selection_key(row, int(row["update"])))
    ordered = sorted(rows, key=lambda row: checkpoint_selection_key(row, int(row["update"])))
    return {
        "contract": CHECKPOINT_SELECTION_CONTRACT,
        "schema_version": 1,
        "selected_update": int(selected["update"]),
        "selected_macro_ovr_auc_hex": float(selected["macro_ovr_auc"]).hex(),
        "selected_cross_entropy_hex": float(selected["cross_entropy"]).hex(),
        "selected_logr50_hex": float(
            selected["macro_mean_log_qcd_rejection_at_50pct_signal"]
        ).hex(),
        "ordered_updates": [int(row["update"]) for row in ordered],
    }


def _loss_for_node(node: NodeSpec, recipe: Mapping[str, Any]) -> LossConfiguration:
    if node.loss_kind == "ce":
        return LossConfiguration(
            arm=f"HCWDL_{node.node_id}_CE", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0, privileged_temperature=1.0,
        )
    if node.loss_kind == "ce_kd":
        coefficient = recipe["single_teacher_coefficients"]
        teacher = node.teachers[0]
        hlt = float(coefficient["teacher_kd"]) if teacher.domain == "hlt" else 0.0
        privileged = float(coefficient["teacher_kd"]) if teacher.domain != "hlt" else 0.0
        temperature = float(
            recipe["predecessor_temperature"]
            if teacher.domain == "hlt"
            else recipe["single_privileged_temperature"]
        )
        return LossConfiguration.for_mixture(
            arm=f"HCWDL_{node.node_id}_SINGLE", ce=float(coefficient["ce"]),
            hlt_kd=hlt, privileged_kd=privileged,
            hlt_temperature=temperature, privileged_temperature=temperature,
        )
    coefficient = recipe["dual_teacher_coefficients"]
    return LossConfiguration.for_mixture(
        arm=f"HCWDL_{node.node_id}_DUAL", ce=float(coefficient["ce"]),
        hlt_kd=float(coefficient["predecessor_kd"]),
        privileged_kd=float(coefficient["privileged_kd"]),
        hlt_temperature=float(recipe["predecessor_temperature"]),
        privileged_temperature=float(recipe["privileged_temperature"]),
    )


def node_training_config(
    node_id: str,
    recipe: Mapping[str, Any],
    *,
    train_rows: int,
    replicate_seed: int,
    require_authorized_recipe: bool = True,
) -> PmardTrainingConfig:
    validate_recipe(recipe, require_authorized=require_authorized_recipe)
    if node_id not in NODE_REGISTRY or train_rows <= 0:
        raise ValueError("HCWDL node or train rows differ")
    node = NODE_REGISTRY[node_id]
    batch = int(recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(np.ceil(train_rows / batch))
    if len(node.teachers) == 2:
        peak_learning_rate = float(recipe["dual_teacher_peak_learning_rate"])
    elif node.stage == "root":
        lr_role = "cold_root"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    elif node.initialization == "warm":
        lr_role = "warm_child"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    else:
        lr_role = "cold_child"
        peak_learning_rate = float(recipe["optimizer"]["peak_learning_rates"][lr_role])
    model_input = str(DOMAINS[node.student_domain]["input"])
    return PmardTrainingConfig(
        experiment_id=node.node_id,
        loss=_loss_for_node(node, recipe),
        total_updates=60 * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=peak_learning_rate,
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass,
        validation_checks=60,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=derive_seed(replicate_seed, f"hcwdl/{node.node_id}"),
        amp_dtype=str(recipe["amp_dtype"]),
        model_input=model_input,
        selection_policy="hcwdl_macro_auc",
    )


def qualifier_training_config(
    qualifier_id: str, recipe: Mapping[str, Any], *, train_rows: int,
    replicate_seed: int,
) -> PmardTrainingConfig:
    """Reconstruct the exact full endpoint-qualifier execution config."""

    input_keys = {
        "T0": "hlt", "TFS": "privileged", "THC": "privileged",
        "TSOFT": "privileged", "TSHELL": "privileged", "TOFF": "toff",
    }
    if qualifier_id not in input_keys or train_rows <= 0:
        raise ValueError("HCWDL qualifier or train rows differ")
    validate_recipe(recipe, require_authorized=True)
    batch = int(recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(np.ceil(train_rows / batch))
    shared_seed = derive_seed(
        int(replicate_seed), "hcwdl/qualification/shared_trajectory_v1",
    )
    return PmardTrainingConfig(
        experiment_id=qualifier_id,
        loss=LossConfiguration(
            arm=f"HCWDL_{qualifier_id}_CE", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0,
            privileged_temperature=1.0,
        ),
        total_updates=60 * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=float(
            recipe["optimizer"]["peak_learning_rates"]["cold_root"]
        ),
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass,
        validation_checks=60,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=shared_seed,
        amp_dtype=str(recipe["amp_dtype"]),
        model_input=input_keys[qualifier_id],
        selection_policy="hcwdl_macro_auc",
    )


def confirmation_control_training_config(
    control_id: str, recipe: Mapping[str, Any], *, train_rows: int,
    replicate_seed: int,
) -> PmardTrainingConfig:
    """Reconstruct the exact full null-control execution config."""

    validate_recipe(recipe, require_authorized=True)
    batch = int(recipe["batching"]["effective_batch_size"])
    updates_per_pass = int(np.ceil(train_rows / batch))
    seed = derive_seed(int(replicate_seed), f"hcwdl/control/{control_id}")
    if control_id == "NULL_WARM_LABEL_ONLY":
        loss = LossConfiguration(
            arm="HCWDL_NULL_WARM_LABEL_ONLY", ce=1.0, hlt_kd=0.0,
            privileged_kd=0.0, temperature=1.0,
            privileged_temperature=1.0,
        )
        learning_rate_role = "warm_child"
    elif control_id in {"NULL_M1_SELF_KD", "NULL_M6_PREDECESSOR_ONLY"}:
        coefficients = (
            recipe["single_teacher_coefficients"]
            if control_id == "NULL_M1_SELF_KD"
            else recipe["controls"]["predecessor_only_coefficients"]
        )
        kd_name = (
            "teacher_kd" if control_id == "NULL_M1_SELF_KD"
            else "predecessor_kd"
        )
        loss = LossConfiguration.for_mixture(
            arm=f"HCWDL_{control_id}", ce=float(coefficients["ce"]),
            hlt_kd=float(coefficients[kd_name]), privileged_kd=0.0,
            hlt_temperature=float(recipe["predecessor_temperature"]),
            privileged_temperature=float(recipe["privileged_temperature"]),
        )
        learning_rate_role = "cold_child"
    else:
        raise ValueError("unknown HCWDL confirmation control")
    return PmardTrainingConfig(
        experiment_id=control_id,
        loss=loss,
        total_updates=60 * updates_per_pass,
        effective_batch_size=batch,
        microbatch_size=int(recipe["batching"]["microbatch_size"]),
        gradient_accumulation=int(recipe["batching"]["gradient_accumulation"]),
        adam_epsilon=float(recipe["optimizer"]["epsilon"]),
        peak_learning_rate=float(
            recipe["optimizer"]["peak_learning_rates"][learning_rate_role]
        ),
        weight_decay=float(recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(recipe["schedule"]["minimum_lr_fraction"]),
        validation_interval=updates_per_pass,
        validation_checks=60,
        logging_interval=max(1, updates_per_pass // 4),
        master_seed=seed,
        amp_dtype=str(recipe["amp_dtype"]),
        model_input="hlt",
        selection_policy="hcwdl_macro_auc",
    )


def initialize_node_model(
    node_id: str,
    *,
    model_factory: Callable[[], object],
    replicate_seed: int,
    warm_checkpoint: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> object:
    import torch

    if node_id not in NODE_REGISTRY:
        raise ValueError("unknown HCWDL node")
    node = NODE_REGISTRY[node_id]
    seed = derive_seed(replicate_seed, f"hcwdl/init/{node_id}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_factory()
    if node.initialization == "fresh":
        if warm_checkpoint is not None or expected_checkpoint_sha256 is not None:
            raise ValueError("fresh HCWDL node cannot load a warm checkpoint")
        return model
    if warm_checkpoint is None or expected_checkpoint_sha256 is None:
        raise ValueError("warm HCWDL node requires its selected parent checkpoint")
    expected = require_sha256(expected_checkpoint_sha256, name="warm checkpoint SHA-256")
    if sha256_file(warm_checkpoint) != expected:
        raise ValueError("warm checkpoint byte hash differs")
    payload = torch.load(warm_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("warm checkpoint payload differs")
    model.load_state_dict(payload["model"], strict=True)
    return model


def train_hcwdl_node(
    *,
    node_id: str,
    recipe: Mapping[str, Any],
    train_rows: int,
    replicate_seed: int,
    model_factory: Callable[[], object],
    train_batches: Callable[[int], Iterable[Mapping[str, object]]],
    validation_batches: Callable[[], Iterable[Mapping[str, object]]],
    class_weights: np.ndarray,
    output_dir: str | Path,
    parents: Mapping[str, str],
    device: str,
    hlt_teacher_targets: EphemeralTeacherTargets | None = None,
    privileged_teacher_targets: EphemeralTeacherTargets | None = None,
    warm_checkpoint: str | Path | None = None,
    warm_checkpoint_sha256: str | None = None,
    resume: bool = True,
    stop_after_update: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    recipe_sha256 = validate_recipe(recipe, require_authorized=True)
    node = NODE_REGISTRY[node_id]
    model = initialize_node_model(
        node_id, model_factory=model_factory, replicate_seed=replicate_seed,
        warm_checkpoint=warm_checkpoint,
        expected_checkpoint_sha256=warm_checkpoint_sha256,
    )
    config = node_training_config(
        node_id, recipe, train_rows=train_rows, replicate_seed=replicate_seed,
    )
    if smoke:
        config = replace(
            config, total_updates=2, validation_interval=2,
            validation_checks=1, logging_interval=1,
        )
    scientific = {
        "campaign": "HCWDL",
        "graph_sha256": GRAPH_SHA256,
        "node": node.payload(),
        "recipe_sha256": recipe_sha256,
        "performance_early_stopping": False,
        **_loss_semantics_payload(),
    }
    scientific.update(
        {"smoke_updates": 2}
        if smoke else {"training_passes": 60, "validation_every_passes": 1}
    )
    validated_parents = {name: require_sha256(value, name=f"HCWDL parent {name}") for name, value in parents.items()}
    validated_parents["recipe"] = recipe_sha256
    report = train_pmard(
        model=model, train_batches=train_batches,
        validation_batches=validation_batches, config=config,
        class_weights=np.asarray(class_weights, np.float32),
        output_dir=output_dir, scientific_config=scientific,
        parents=validated_parents, device=device,
        hlt_teacher_targets=hlt_teacher_targets,
        privileged_teacher_targets=privileged_teacher_targets,
        loss_semantics_contract=HCWDL_PARENT_BASE_LOSS_CONTRACT,
        resume=resume, stop_after_update=stop_after_update,
    )
    expected_checks = 1 if smoke else 60
    if len(report["validation_history"]) != expected_checks:
        raise RuntimeError(
            f"HCWDL node did not publish exactly {expected_checks} validation records"
        )
    selection = select_checkpoint(report["validation_history"])
    if selection["selected_update"] != report["selected_update"]:
        raise RuntimeError("HCWDL engine and independent checkpoint selection differ")
    output = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT,
        "schema_version": 1,
        "node_id": node_id,
        "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": recipe_sha256,
        "parents": validated_parents,
        "pmard_engine_report_sha256": report["content_hash"],
        "pmard_execution_config_sha256": report["execution_config_sha256"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "final_checkpoint_sha256": report["final_checkpoint_sha256"],
        "selection": selection,
        "complete": True,
        **_loss_semantics_payload(),
    })
    validate_hcwdl_training_report(output)
    write_immutable_json(Path(output_dir) / "hcwdl_training_report.json", output)
    return output


__all__ = [
    "CHECKPOINT_SELECTION_CONTRACT", "TRAINING_REPORT_CONTRACT",
    "checkpoint_selection_key", "confirmation_control_training_config",
    "initialize_node_model", "node_training_config", "qualifier_training_config",
    "select_checkpoint", "train_hcwdl_node",
    "validate_hcwdl_full_parent_engine_report",
    "validate_hcwdl_full_parent_wrapper_report",
    "validate_hcwdl_parent_prefix_campaign", "validate_hcwdl_training_report",
]
