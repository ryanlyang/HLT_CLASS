"""Dedicated HCWDL matching-free representation-KD node training.

The runtime in this module is intentionally separate from the historical
PMARD representation arms.  A training update consumes one authenticated
student-view batch and one already materialized compact target-bank join.
That one immediate-predecessor bank supplies both logits and representation
targets.  Teacher particles and teacher models are therefore absent while
the student optimizer is live.

Production callers must provide the immutable graph/recipe/target lineage and
the missing repository-specific HLT batch adapter.  The adapter protocol is
small and explicit: every batch carries ``features``, ``vectors``, ``mask``,
``visible_indices``, ``family_codes``, ``labels`` and ``identity_digests``.
This file never derives an identity from labels or particle positions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
from io import BytesIO
import gc
import inspect
import math
import pickle
from pathlib import Path
import random
import time
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes,
    canonical_sha256,
    load_json,
    require_sha256,
    sha256_bytes,
    sha256_file,
    validate_content_hash,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.training.checkpoints import (
    capture_model_runtime_state,
    capture_rng_state,
    restore_model_runtime_state,
    restore_rng_state,
)

from .evaluation import classification_metrics
from .hcwdl_parent_loss import hcwdl_base_loss, hcwdl_base_loss_rows
from .hcwdl_recipe import validate_recipe as validate_parent_recipe
from .hcwdl_representation_calibration import (
    CALIBRATION_BATCHES,
    CALIBRATION_SELECTION_CONTRACT,
    CalibrationComponentRows,
    CalibrationForwardResult,
    GradientCalibrationResult,
    calibrate_representation_components,
    early_backbone_parameters,
    validate_calibration_selection_artifact,
)
from .hcwdl_representation_contracts import (
    ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
    CHECKPOINT_SELECTION_CONTRACT,
    DEPLOYABLE_EXTRACTION_CONTRACT,
    DIAGNOSTIC_BATCH_CONTRACT,
    GRADIENT_CALIBRATION_CONTRACT,
    GRADIENT_CALIBRATION_MANIFEST_CONTRACT,
    TARGET_MANIFEST_CONTRACT,
    TRAINING_REPORT_CONTRACT,
    SELECTED_TRAINING_CHECKPOINT_CONTRACT,
    FINAL_TRAINING_CHECKPOINT_CONTRACT,
    build_versioned_artifact,
    logical_array_sha256,
    logical_array_sha256_from_byte_hash,
    validate_versioned_artifact,
)
from .hcwdl_representation_artifacts import publish_binary_envelope
from .hcwdl_representation_graph import (
    CONTROL_REGISTRY,
    NODE_REGISTRY,
    RREL_STRATEGY,
    RSET_STRATEGY,
    RepresentationControlSpec,
    RepresentationNodeSpec,
)
from .hcwdl_representation_graph_registry import (
    registered_graph_sha256 as _registered_graph_sha256,
)
from .hcwdl_representation_kernels import SpectralKernelResources
from .hcwdl_representation_losses import (
    ORTHOGONALITY_COEFFICIENT,
    RHO_REPRESENTATION,
    JetLossResult,
    RelationLossResult,
    SetLossResult,
    class_weighted_eligible_mean,
    effective_pass_for_update,
    jet_representation_loss,
    jet_set_ramp,
    native_offline_set_representation_loss,
    ordinary_set_representation_loss,
    projection_diagnostics,
    projection_orthogonality,
    relation_ramp,
    relation_representation_loss,
    scheduled_representation_loss,
)
from .hcwdl_representation_recipe import (
    PARENT_RECIPE_CONTRACT,
    validate_representation_recipe,
)
from .hcwdl_representation_reporting import checkpoint_key, select_checkpoint
from .hcwdl_representation_resume import (
    REQUIRED_LINEAGE_KEYS,
    build_state_inventory,
    load_highest_valid_resume,
    publish_resume_generation,
    scan_resume_generations,
)
from .hcwdl_representation_targets import identity_order_sha256, identity_set_sha256
from .training import LossConfiguration, derive_seed


PREDECESSOR_LOGIT_BANK_CONTRACT: Final = "HCWDL_REP_PREDECESSOR_LOGIT_RAM/v1"
BATCH_PROTOCOL: Final = "HCWDL_REPRESENTATION_HLT_BATCH/v1"
EXECUTION_MODES: Final = ("scientific", "smoke", "synthetic_test")
PAIRED_RNG_STREAMS_CONTRACT: Final = "HCWDL_REP_PAIRED_RNG_STREAMS/v1"


class RepresentationTrainingInterrupted(RuntimeError):
    """Raised only after a complete crash-safe resume generation is committed."""


@dataclass(frozen=True)
class NodeExecution:
    """Normalized execution semantics for one primary node or terminal control."""

    execution_id: str
    graph_spec: RepresentationNodeSpec | RepresentationControlSpec
    strategy: str
    track: str
    rung: int
    student_domain: str
    deployable: bool
    parent_counterpart: str
    initialization: str
    initialization_parent: str | None
    predecessor_logit_teacher: str | None
    representation_logit_teacher: str
    representation_teacher_domain: str
    target_bank_identity: str
    is_control: bool
    jet_only: bool
    relation_enabled: bool
    shuffled_representation_targets: bool
    control_counterpart: str | None

    @property
    def short_strategy(self) -> str:
        return "RSET" if self.strategy == RSET_STRATEGY else "RREL"

    @property
    def teacher_latent_domain(self) -> str:
        return (
            "native_offline"
            if self.representation_teacher_domain == "toff"
            else "ordinary"
        )

    @property
    def active_components(self) -> tuple[str, ...]:
        if self.jet_only:
            return ("jet",)
        if self.relation_enabled:
            return ("jet", "set", "relation")
        return ("jet", "set")


def resolve_node_execution(execution_id: str) -> NodeExecution:
    """Resolve routing from the frozen registries, never from name suffixes."""

    if execution_id in NODE_REGISTRY:
        node = NODE_REGISTRY[execution_id]
        return NodeExecution(
            execution_id=execution_id,
            graph_spec=node,
            strategy=node.strategy,
            track=node.track,
            rung=node.rung,
            student_domain=node.student_domain,
            deployable=node.deployable,
            parent_counterpart=node.parent_counterpart,
            initialization=node.initialization,
            initialization_parent=node.initialization_parent,
            predecessor_logit_teacher=node.predecessor_logit_teacher,
            representation_logit_teacher=node.representation_logit_teacher,
            representation_teacher_domain=node.representation_teacher_domain,
            target_bank_identity=node.target_bank_identity,
            is_control=False,
            jet_only=False,
            relation_enabled=node.strategy == RREL_STRATEGY,
            shuffled_representation_targets=False,
            control_counterpart=None,
        )
    if execution_id in CONTROL_REGISTRY:
        control = CONTROL_REGISTRY[execution_id]
        allocation = dict(control.component_allocation)
        return NodeExecution(
            execution_id=execution_id,
            graph_spec=control,
            strategy=control.strategy,
            track=control.track,
            rung=control.rung,
            student_domain=control.student_domain,
            deployable=True,
            parent_counterpart=control.parent_counterpart,
            initialization=control.initialization,
            initialization_parent=control.initialization_parent,
            predecessor_logit_teacher=control.predecessor_logit_teacher,
            representation_logit_teacher=control.representation_logit_teacher,
            representation_teacher_domain=control.representation_teacher_domain,
            target_bank_identity=control.target_bank_identity,
            is_control=True,
            jet_only=allocation == {"jet": 1.0, "set": 0.0, "relation": 0.0},
            relation_enabled=allocation["relation"] > 0,
            shuffled_representation_targets=control.shuffled_representation_targets,
            control_counterpart=control.paired_primary_node,
        )
    # Supplemental graph identities remain outside the dense-descent graph
    # contract.  The lazy import avoids a module cycle while allowing the
    # shared, heavily tested representation engine to execute their exact
    # frozen semantics.
    from .hcwdl_homotopy_representation_graph import NODE_REGISTRY as U_RKD_REGISTRY

    if execution_id in U_RKD_REGISTRY:
        node = U_RKD_REGISTRY[execution_id]
        return NodeExecution(
            execution_id=execution_id,
            graph_spec=node,
            strategy=node.strategy,
            track=node.track,
            rung=node.transition_index,
            student_domain=node.student_domain,
            deployable=node.deployable,
            parent_counterpart=node.parent_counterpart,
            initialization=node.initialization,
            initialization_parent=node.initialization_parent,
            predecessor_logit_teacher=None,
            representation_logit_teacher=node.teacher.node_id,
            representation_teacher_domain=node.teacher.domain,
            target_bank_identity=node.target_bank_identity,
            is_control=False,
            jet_only=False,
            relation_enabled=node.strategy == RREL_STRATEGY,
            shuffled_representation_targets=False,
            control_counterpart=None,
        )
    from .hcwdl_direct_offline_kd_graph import (
        REPRESENTATION_NODE_REGISTRY as DIRECT_REGISTRY,
    )

    if execution_id in DIRECT_REGISTRY:
        node = DIRECT_REGISTRY[execution_id]
        return NodeExecution(
            execution_id=execution_id,
            graph_spec=node,
            strategy=node.strategy,
            track=node.track,
            rung=1,
            student_domain="hlt",
            deployable=True,
            parent_counterpart="HLT_DIRECT_PAIR",
            initialization="fresh",
            initialization_parent=None,
            predecessor_logit_teacher=None,
            representation_logit_teacher="TOFF_CE",
            representation_teacher_domain="toff",
            target_bank_identity="TOFF_CE",
            is_control=False,
            jet_only=False,
            relation_enabled=node.strategy == RREL_STRATEGY,
            shuffled_representation_targets=False,
            control_counterpart=None,
        )
    raise KeyError(f"unregistered HCWDL-RKD execution {execution_id!r}")


def paired_rng_streams(execution_id: str, replicate_seed: int) -> dict[str, Any]:
    """Derive the parent-counterpart random streams frozen by Section 18.2.

    Data/model streams deliberately use the parent logit node's domains, not
    the new representation node ID.  Consequently RSET, RREL, their M5
    controls, and the compatible logit-only execution receive the same row
    order and stochastic forward draws for a fixed replicate seed.  Only the
    auxiliary projection-head construction is node-specific.
    """

    execution = resolve_node_execution(execution_id)
    if isinstance(replicate_seed, bool) or int(replicate_seed) < 0:
        raise ValueError("HCWDL-RKD replicate seed differs")
    seed = int(replicate_seed)
    if execution_id in {"HLT_RSET", "HLT_RREL"}:
        alias = "direct_hlt_pair_v1"
        training_master = derive_seed(seed, "hcwdl/direct_hlt_pair_v1")
        domains = {
            "sampler": "hcwdl_direct/sampler/direct_hlt_pair_v1",
            "validation_order": "hcwdl_direct/sampler/direct_hlt_pair_v1",
            "repair": "hcwdl_direct/no_repair",
            "backbone_initialization": "hcwdl/init/direct_hlt_pair_v1",
            "counterpart_training_master": "hcwdl/direct_hlt_pair_v1",
            "training_stochastic": "training_dropout_and_augmentation",
            "representation_projection": (
                f"hcwdl_direct/representation_projection/{execution_id}"
            ),
        }
        streams = {name: derive_seed(seed, domain) for name, domain in domains.items()}
        streams["counterpart_training_master"] = training_master
        streams["training_stochastic"] = derive_seed(
            training_master, domains["training_stochastic"],
        )
        return with_content_hash({
            "contract": PAIRED_RNG_STREAMS_CONTRACT,
            "schema_version": 1,
            "execution_id": execution_id,
            "parent_logit_counterpart_node_id": "HLT_DIRECT_PAIR",
            "replicate_seed": seed,
            "domains": domains,
            "streams": streams,
            "seed_alias": alias,
            "rff_consumes_training_rng": False,
            "calibration_restores_training_rng": True,
        })
    if execution_id.startswith(("F_RSET_", "F_RREL_")):
        from .hcwdl_homotopy_representation_graph import NODE_REGISTRY as U_RKD_REGISTRY

        node = U_RKD_REGISTRY[execution_id]
        alias = node.seed_alias
        domains = {
            "sampler": f"hcwdl_uj/sampler/{alias}",
            "validation_order": f"hcwdl_uj/sampler/{alias}",
            "repair": "hcwdl_uj/repair/shared_v1",
            "backbone_initialization": f"hcwdl_uj/init/{alias}",
            "counterpart_training_master": f"hcwdl_uj/training/{alias}",
            "training_stochastic": f"hcwdl_uj/training_stochastic/{alias}",
            "representation_projection": (
                f"hcwdl_u_rkd/representation_projection/{execution_id}"
            ),
        }
        streams = {
            name: derive_seed(seed, domain) for name, domain in domains.items()
        }
        return with_content_hash({
            "contract": PAIRED_RNG_STREAMS_CONTRACT,
            "schema_version": 1,
            "execution_id": execution_id,
            "parent_logit_counterpart_node_id": node.parent_counterpart,
            "replicate_seed": seed,
            "domains": domains,
            "streams": streams,
            "seed_alias": alias,
            "rff_consumes_training_rng": False,
            "calibration_restores_training_rng": True,
        })
    counterpart = execution.parent_counterpart
    master = derive_seed(seed, f"hcwdl/{counterpart}")
    domains = {
        "sampler": "hcwdl/sampler",
        "validation_order": "hcwdl/sampler",
        "repair": "hcwdl/repair/hlt",
        "backbone_initialization": f"hcwdl/init/{counterpart}",
        "counterpart_training_master": f"hcwdl/{counterpart}",
        "training_stochastic": "training_dropout_and_augmentation",
        "representation_projection": (
            f"hcwdl_rkd/representation_projection/{execution_id}"
        ),
    }
    streams = {
        "sampler": derive_seed(seed, domains["sampler"]),
        "validation_order": derive_seed(seed, domains["validation_order"]),
        "repair": derive_seed(seed, domains["repair"]),
        "backbone_initialization": derive_seed(
            seed, domains["backbone_initialization"],
        ),
        "counterpart_training_master": master,
        "training_stochastic": derive_seed(
            master, domains["training_stochastic"],
        ),
        "representation_projection": derive_seed(
            seed, domains["representation_projection"],
        ),
    }
    return with_content_hash({
        "contract": PAIRED_RNG_STREAMS_CONTRACT,
        "schema_version": 1,
        "execution_id": execution_id,
        "parent_logit_counterpart_node_id": counterpart,
        "replicate_seed": seed,
        "domains": domains,
        "streams": streams,
        "rff_consumes_training_rng": False,
        "calibration_restores_training_rng": True,
    })


def validate_paired_rng_streams(
    value: Mapping[str, Any], *, execution_id: str, replicate_seed: int,
) -> str:
    digest = validate_content_hash(
        value,
        expected_contract=PAIRED_RNG_STREAMS_CONTRACT,
        expected_schema_version=1,
    )
    if dict(value) != paired_rng_streams(execution_id, replicate_seed):
        raise ValueError("HCWDL-RKD paired RNG stream derivation differs")
    return digest


def _seed_paired_training_rng(value: Mapping[str, Any]) -> None:
    """Install the counterpart's stochastic stream before optimizer work."""

    import torch

    training_seed = int(value["streams"]["training_stochastic"])
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)


@dataclass(frozen=True)
class RepresentationTrainingConfiguration:
    """Optimization values inherited from the authenticated parent recipe."""

    execution_id: str
    mode: str
    replicate_seed: int
    train_rows: int
    effective_batch_size: int
    training_passes: int
    maximum_optimizer_updates: int | None
    peak_learning_rate: float
    adam_betas: tuple[float, float]
    adam_epsilon: float
    weight_decay: float
    warmup_fraction: float
    minimum_lr_fraction: float
    amp_dtype: str
    logging_interval_updates: int

    def __post_init__(self) -> None:
        resolve_node_execution(self.execution_id)
        if self.mode not in EXECUTION_MODES:
            raise ValueError("unknown HCWDL-RKD execution mode")
        if self.train_rows <= 0 or self.effective_batch_size <= 0:
            raise ValueError("HCWDL-RKD train rows/batch size must be positive")
        if self.training_passes <= 0 or self.peak_learning_rate <= 0:
            raise ValueError("HCWDL-RKD duration/learning rate must be positive")
        if self.mode == "scientific" and (
            self.training_passes != 60 or self.maximum_optimizer_updates is not None
        ):
            raise ValueError("scientific HCWDL-RKD nodes require exactly 60 passes")
        if self.mode == "smoke" and self.maximum_optimizer_updates != 2:
            raise ValueError("HCWDL-RKD smoke requires exactly two optimizer updates")
        if self.maximum_optimizer_updates is not None and self.maximum_optimizer_updates <= 0:
            raise ValueError("optimizer-update bound must be positive")
        if self.amp_dtype not in {"none", "bfloat16"}:
            raise ValueError("unsupported HCWDL-RKD autocast dtype")
        if self.logging_interval_updates <= 0:
            raise ValueError("HCWDL-RKD logging interval must be positive")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.minimum_lr_fraction <= 1:
            raise ValueError("HCWDL-RKD learning-rate schedule differs")

    @property
    def updates_per_pass(self) -> int:
        return math.ceil(self.train_rows / self.effective_batch_size)

    @property
    def scientific_total_updates(self) -> int:
        return self.training_passes * self.updates_per_pass

    @property
    def active_total_updates(self) -> int:
        return (
            self.scientific_total_updates
            if self.maximum_optimizer_updates is None
            else self.maximum_optimizer_updates
        )


def representation_training_configuration(
    execution_id: str,
    parent_recipe: Mapping[str, Any],
    *,
    train_rows: int,
    replicate_seed: int,
    mode: str = "scientific",
    synthetic_passes: int = 1,
) -> RepresentationTrainingConfiguration:
    """Derive every optimization value from the exact ``HCWDL_RECIPE/v4`` parent.

    ``synthetic_test`` is an explicitly non-authorizing local fixture mode. It
    is the only mode that can shorten the pass count without claiming a
    scientific execution.
    """

    require_authorized = mode in {"scientific", "smoke"}
    validate_parent_recipe(
        parent_recipe,
        require_authorized=require_authorized,
        expected_profile="primary_ladder" if require_authorized else None,
    )
    if parent_recipe.get("contract") != PARENT_RECIPE_CONTRACT:
        raise ValueError(
            "HCWDL-RKD requires the unweighted HCWDL_RECIPE/v4 parent"
        )
    if not np.array_equal(
        np.asarray(parent_recipe.get("class_weights"), dtype=np.float32),
        np.ones(15, dtype=np.float32),
    ):
        raise ValueError("HCWDL-RKD parent recipe must bind fifteen exact ones")
    execution = resolve_node_execution(execution_id)
    batching = parent_recipe["batching"]
    if int(batching["gradient_accumulation"]) != 1:
        raise ValueError("the frozen HCWDL-RKD one-forward step requires accumulation one")
    if int(batching["microbatch_size"]) != int(batching["effective_batch_size"]):
        raise ValueError("the frozen HCWDL-RKD microbatch must equal effective batch")
    # Every dense-descent node has one exact teacher: the target bank produced
    # by its immediate richer predecessor (TOFF for the D100 roots).  The old
    # ascent-only dual-teacher learning rate is therefore never applicable.
    lr_role = "warm_child" if execution.initialization == "warm" else "cold_child"
    peak = float(parent_recipe["optimizer"]["peak_learning_rates"][lr_role])
    passes = 60 if mode != "synthetic_test" else int(synthetic_passes)
    maximum = 2 if mode == "smoke" else None
    updates_per_pass = math.ceil(train_rows / int(batching["effective_batch_size"]))
    return RepresentationTrainingConfiguration(
        execution_id=execution_id,
        mode=mode,
        replicate_seed=int(replicate_seed),
        train_rows=int(train_rows),
        effective_batch_size=int(batching["effective_batch_size"]),
        training_passes=passes,
        maximum_optimizer_updates=maximum,
        peak_learning_rate=peak,
        adam_betas=tuple(float(value) for value in parent_recipe["optimizer"]["betas"]),
        adam_epsilon=float(parent_recipe["optimizer"]["epsilon"]),
        weight_decay=float(parent_recipe["optimizer"]["weight_decay"]),
        warmup_fraction=float(parent_recipe["schedule"]["warmup_fraction"]),
        minimum_lr_fraction=float(parent_recipe["schedule"]["minimum_lr_fraction"]),
        amp_dtype=str(parent_recipe["amp_dtype"]),
        logging_interval_updates=max(1, updates_per_pass // 4),
    )


def _default_deployable_factory():
    from hlt_classification.models.scouting_particle_transformer import (
        build_scouting_particle_transformer,
    )

    return build_scouting_particle_transformer()


def _default_wrapper_factory(**kwargs):
    from hlt_classification.models.hcwdl_representation import (
        HCWDLRepresentationStudent,
    )

    return HCWDLRepresentationStudent(**kwargs)


def initialize_representation_student(
    execution_id: str,
    *,
    replicate_seed: int,
    warm_checkpoint: str | Path | None = None,
    warm_checkpoint_sha256: str | None = None,
    deployable_factory: Callable[[], Any] = _default_deployable_factory,
    wrapper_factory: Callable[..., Any] = _default_wrapper_factory,
    warm_loader: Callable[[str | Path, str], Any] | None = None,
):
    """Create the paired cold backbone or strict warm deployable plus fresh heads."""

    import torch

    execution = resolve_node_execution(execution_id)
    rng_streams = paired_rng_streams(execution_id, replicate_seed)
    backbone_seed = int(rng_streams["streams"]["backbone_initialization"])
    projection_seed = int(rng_streams["streams"]["representation_projection"])
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(backbone_seed)
        if execution.initialization == "fresh":
            if warm_checkpoint is not None or warm_checkpoint_sha256 is not None:
                raise ValueError("cold HCWDL-RKD node cannot load warm state")
            deployable = deployable_factory()
        else:
            if warm_checkpoint is None or warm_checkpoint_sha256 is None:
                raise ValueError("warm HCWDL-RKD node requires extracted deployable state")
            expected = require_sha256(
                warm_checkpoint_sha256, name="warm deployable checkpoint SHA-256",
            )
            if warm_loader is None:
                from hlt_classification.models.hcwdl_representation import (
                    load_hcwdl_deployable_checkpoint,
                )

                deployable = load_hcwdl_deployable_checkpoint(
                    warm_checkpoint, expected_sha256=expected,
                )
            else:
                deployable = warm_loader(warm_checkpoint, expected)
        torch.manual_seed(projection_seed)
        model = wrapper_factory(
            strategy=execution.short_strategy,
            teacher_latent_domain=execution.teacher_latent_domain,
            jet_only=execution.jet_only,
            deployable_model=deployable,
        )
        heads = model.representation_heads
        heads.reset_identity()
    expected_heads = (
        ("jet",)
        if execution.jet_only
        else ("jet", "token")
        if execution.teacher_latent_domain == "ordinary"
        else ("jet", "token_charged", "token_neutral")
    )
    observed_heads = tuple(name for name, _ in heads.projection_items())
    if observed_heads != expected_heads:
        raise RuntimeError("HCWDL-RKD fresh representation-head topology differs")
    for _, projection in heads.projection_items():
        identity = torch.eye(128, dtype=projection.weight.dtype, device=projection.weight.device)
        if not torch.equal(projection.weight.detach(), identity):
            raise RuntimeError("HCWDL-RKD representation head is not exactly identity initialized")
    return model


@dataclass(frozen=True)
class HLTBatch:
    features: Any
    vectors: Any
    mask: Any
    visible_indices: Any
    family_codes: Any
    labels: Any
    identity_digests: np.ndarray
    family_reason_codes: Any | None = None


def _field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        if name not in container:
            raise ValueError(f"HCWDL-RKD HLT batch lacks {name}")
        return container[name]
    if not hasattr(container, name):
        raise ValueError(f"HCWDL-RKD HLT view lacks {name}")
    return getattr(container, name)


def normalize_hlt_batch(batch: Mapping[str, Any]) -> HLTBatch:
    """Validate the strict HLT-only batch boundary used by training and caches."""

    if not isinstance(batch, Mapping):
        raise TypeError("HCWDL-RKD batch must be a mapping")
    if "hlt" in batch:
        expected_batch = {"hlt", "labels", "identity_digests"}
        if set(batch) != expected_batch:
            raise ValueError(
                "HCWDL-RKD nested batch fields differ: "
                f"missing={sorted(expected_batch - set(batch))}, "
                f"unexpected={sorted(set(batch) - expected_batch)}"
            )
        view = batch["hlt"]
        required_view = {
            "features", "vectors", "mask", "visible_indices", "family_codes",
        }
        allowed_view = required_view | {"family_reason_codes"}
        if isinstance(view, Mapping) and (
            not required_view <= set(view) or set(view) - allowed_view
        ):
            raise ValueError(
                "HCWDL-RKD HLT view fields differ: "
                f"missing={sorted(required_view - set(view))}, "
                f"unexpected={sorted(set(view) - allowed_view)}"
            )
    else:
        required_batch = {
            "features", "vectors", "mask", "visible_indices", "family_codes",
            "labels", "identity_digests",
        }
        allowed_batch = required_batch | {"family_reason_codes"}
        if not required_batch <= set(batch) or set(batch) - allowed_batch:
            raise ValueError(
                "HCWDL-RKD batch fields differ: "
                f"missing={sorted(required_batch - set(batch))}, "
                f"unexpected={sorted(set(batch) - allowed_batch)}"
            )
        view = batch
    features = _field(view, "features")
    vectors = _field(view, "vectors")
    mask = _field(view, "mask")
    visible_indices = _field(view, "visible_indices")
    family_codes = _field(view, "family_codes")
    family_reason_codes = (
        view.get("family_reason_codes")
        if isinstance(view, Mapping)
        else getattr(view, "family_reason_codes", None)
    )
    labels = batch.get("labels")
    identities = np.asarray(batch.get("identity_digests"))
    if labels is None:
        raise ValueError("HCWDL-RKD HLT batch lacks labels")
    if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1] != 32:
        raise ValueError("HCWDL-RKD identities must be uint8 [batch,32]")
    if len({bytes(row) for row in identities}) != len(identities):
        raise ValueError("HCWDL-RKD batch repeats a canonical identity")
    shape = np.shape(features)
    if len(shape) != 3 or shape[1] != 21 or shape[0] != len(identities):
        raise ValueError("HCWDL-RKD feature batch must be [batch,21,tokens]")
    if np.shape(vectors) != (shape[0], 4, shape[2]):
        raise ValueError("HCWDL-RKD vector batch shape differs")
    if np.shape(mask) not in {(shape[0], 1, shape[2]), (shape[0], shape[2])}:
        raise ValueError("HCWDL-RKD mask batch shape differs")
    if np.shape(visible_indices) != (shape[0], shape[2]):
        raise ValueError("HCWDL-RKD visible-index shape differs")
    if np.shape(family_codes) != (shape[0], shape[2]):
        raise ValueError("HCWDL-RKD family-code shape differs")
    if family_reason_codes is not None and np.shape(family_reason_codes) != (
        shape[0], shape[2]
    ):
        raise ValueError("HCWDL-RKD family-reason-code shape differs")
    if np.shape(labels) != (shape[0],):
        raise ValueError("HCWDL-RKD label shape differs")
    return HLTBatch(
        features, vectors, mask, visible_indices, family_codes, labels,
        np.ascontiguousarray(identities), family_reason_codes,
    )


def _validate_target_bank_binding(
    target_bank,
    *,
    execution: NodeExecution,
    lineage: Mapping[str, str],
    train_rows: int,
    replicate_seed: int,
) -> Mapping[str, Any]:
    """Bind the live RAM bank to this exact authorized execution."""

    manifest = getattr(target_bank, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError("HCWDL-RKD target bank lacks an immutable manifest")
    supplemental = execution.execution_id.startswith(("F_RSET_", "F_RREL_"))
    direct = execution.execution_id in {"HLT_RSET", "HLT_RREL"}
    if supplemental:
        from .hcwdl_homotopy_representation_targets import validate_target_manifest

        validate_target_manifest(manifest)
    elif direct:
        from .hcwdl_direct_offline_kd_targets import validate_target_manifest

        validate_target_manifest(manifest)
    else:
        validate_versioned_artifact(
            manifest, expected_contract=TARGET_MANIFEST_CONTRACT,
        )
    payload = manifest["payload"]
    if manifest["parents"].get("target_generation") != lineage["target_generation"]:
        raise ValueError("HCWDL-RKD target manifest generation lineage differs")
    if payload.get("logical_target_sha256") != lineage["target_logical"]:
        raise ValueError("HCWDL-RKD target manifest logical lineage differs")
    if payload.get("logical_bank_id") != execution.target_bank_identity:
        raise ValueError("HCWDL-RKD target bank identity differs")
    if int(payload.get("rows", -1)) != train_rows:
        raise ValueError("HCWDL-RKD target bank train-row count differs")
    require_sha256(payload.get("identity_set_sha256"), name="target identity-set SHA-256")
    consumers = payload.get("authorized_consumers")
    if not isinstance(consumers, Sequence):
        raise ValueError("HCWDL-RKD target manifest consumer registry is absent")
    expected = {
        "node_id": execution.execution_id,
        "strategy": execution.short_strategy,
        "track": execution.track,
    }
    if not supplemental and not direct:
        expected["execution_id"] = lineage["execution"]
    matches = [
        row for row in consumers
        if isinstance(row, Mapping)
        and all(row.get(name) == value for name, value in expected.items())
    ]
    if len(matches) != 1:
        raise PermissionError("HCWDL-RKD target bank does not authorize this execution")
    if int(matches[0].get("seed", -1)) != int(replicate_seed):
        raise PermissionError(
            "HCWDL-RKD target consumer seed differs from the physical execution"
        )
    return payload


def _batch_tensors(batch: HLTBatch, device):
    import torch

    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    vectors = torch.as_tensor(batch.vectors, dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch.mask, dtype=torch.bool, device=device)
    if mask.ndim == 2:
        mask = mask[:, None, :]
    visible = torch.as_tensor(batch.visible_indices, dtype=torch.long, device=device)
    family = torch.as_tensor(batch.family_codes, dtype=torch.int8, device=device)
    labels = torch.as_tensor(batch.labels, dtype=torch.long, device=device)
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= 15):
        raise ValueError("HCWDL-RKD labels lie outside 15 classes")
    return features, vectors, mask, visible, family, labels


@dataclass(frozen=True)
class InMemoryPredecessorLogits:
    identities: np.ndarray
    logits: np.ndarray
    logical_sha256: str
    _lookup: Mapping[bytes, int]

    def join(self, identity_digests: np.ndarray) -> np.ndarray:
        identities = np.asarray(identity_digests)
        if identities.dtype != np.uint8 or identities.ndim != 2 or identities.shape[1] != 32:
            raise ValueError("predecessor-logit join identities differ")
        keys = [bytes(row) for row in identities]
        if len(keys) != len(set(keys)):
            raise ValueError("predecessor-logit join repeats identities")
        try:
            indexes = np.asarray([self._lookup[key] for key in keys], dtype=np.int64)
        except KeyError as error:
            raise KeyError("predecessor-logit RAM join is incomplete") from error
        return np.ascontiguousarray(self.logits[indexes])


def build_predecessor_logit_bank(
    predecessor_model,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: str,
    expected_rows: int,
) -> InMemoryPredecessorLogits:
    """Run one frozen FP32 HLT pass and return a canonical RAM-only logit bank."""

    import torch

    target_device = torch.device(device)
    predecessor_model.to(target_device)
    predecessor_model.eval()
    identity_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for raw in batches:
            batch = normalize_hlt_batch(raw)
            features, vectors, mask, _, _, _ = _batch_tensors(batch, target_device)
            with torch.autocast(device_type=target_device.type, enabled=False):
                logits = predecessor_model(features, vectors, mask).float()
            if logits.shape != (len(batch.identity_digests), 15) or not torch.isfinite(logits).all():
                raise FloatingPointError("predecessor HLT logits are invalid")
            identity_parts.append(batch.identity_digests.copy())
            logit_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
    if not identity_parts:
        raise ValueError("predecessor-logit pass produced no rows")
    identities = np.ascontiguousarray(np.concatenate(identity_parts, axis=0))
    logits = np.ascontiguousarray(np.concatenate(logit_parts, axis=0), dtype=np.float32)
    if len(identities) != expected_rows:
        raise ValueError("predecessor-logit pass row count differs")
    keys = [bytes(row) for row in identities]
    if len(set(keys)) != len(keys):
        raise ValueError("predecessor-logit pass repeats identities")
    order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)
    identities = np.ascontiguousarray(identities[order])
    logits = np.ascontiguousarray(logits[order])
    lookup = {bytes(row): index for index, row in enumerate(identities)}
    logical = canonical_sha256({
        "contract": PREDECESSOR_LOGIT_BANK_CONTRACT,
        "identity_sha256": logical_array_sha256("identity_digests", identities),
        "logits_sha256": logical_array_sha256("logits", logits),
        "rows": len(identities),
    })
    return InMemoryPredecessorLogits(identities, logits, logical, lookup)


def _train_batch_stream(
    factory: Callable[..., Iterable[Mapping[str, Any]]],
    *,
    pass_index: int,
    start_batch: int,
) -> Iterable[Mapping[str, Any]]:
    """Open a canonical stream at its cursor without replaying prior batches."""

    try:
        inspect.signature(factory).bind(pass_index, start_batch)
    except (TypeError, ValueError):
        if start_batch:
            raise TypeError(
                "mid-pass exact resume requires train_batches(pass_index, start_batch)"
            ) from None
        return factory(pass_index)
    return factory(pass_index, start_batch)


def _release_frozen_model(model) -> None:
    try:
        model.to("cpu")
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            torch = None


def node_base_loss_configuration(execution: NodeExecution) -> LossConfiguration:
    """Return the single-immediate-teacher base objective for every rung."""

    temperature = 1.0
    if execution.execution_id.startswith(("F_RSET_", "F_RREL_")):
        from .hcwdl_homotopy_representation_graph import NODE_REGISTRY as U_RKD_REGISTRY

        temperature = U_RKD_REGISTRY[execution.execution_id].temperature
    elif execution.execution_id in {"HLT_RSET", "HLT_RREL"}:
        temperature = 2.0
    return LossConfiguration.for_mixture(
        arm=f"HCWDL_RKD_{execution.execution_id}_DENSE_DESCENT",
        ce=0.25,
        hlt_kd=0.75,
        privileged_kd=0.0,
        hlt_temperature=temperature,
        privileged_temperature=temperature,
    )


def _target_tensors(
    target_bank,
    identities: np.ndarray,
    *,
    device,
    execution: NodeExecution,
    shuffled_representation_joiner: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None,
) -> dict[str, Any]:
    import torch

    correct = dict(target_bank.join(identities))
    if execution.shuffled_representation_targets:
        if shuffled_representation_joiner is None:
            raise ValueError("shuffled control requires an authenticated representation-only joiner")
        shuffled = dict(shuffled_representation_joiner(identities))
        forbidden = {"logits", "identity_digest", "label", "source_file_id", "source_entry"}
        if forbidden & set(shuffled):
            raise ValueError("shuffle adapter attempted to replace logits/identity/labels")
        representation_keys = set(correct) - {"logits"}
        if set(shuffled) != representation_keys:
            raise ValueError("shuffled representation target columns differ")
        correct.update(shuffled)
    if "logits" not in correct:
        raise ValueError("privileged target bank lacks logits")
    tensors: dict[str, Any] = {}
    for name, raw in correct.items():
        array = np.asarray(raw)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise FloatingPointError(f"target bank column {name!r} is nonfinite")
        tensor = torch.as_tensor(array, device=device)
        if tensor.dtype.is_floating_point:
            tensor = tensor.float()
        tensor = tensor.detach()
        if tensor.requires_grad:
            raise RuntimeError("teacher target unexpectedly requires gradients")
        tensors[name] = tensor
    return tensors


@dataclass(frozen=True)
class RawRepresentationComponents:
    jet: JetLossResult | None
    set: SetLossResult | None
    relation: RelationLossResult | None
    losses: Mapping[str, Any]
    rows: Mapping[str, CalibrationComponentRows]


def _ordinary_eligibility(value, *, batch: int, trailing: tuple[int, ...]):
    import torch

    result = torch.as_tensor(value, dtype=torch.bool, device=value.device)
    expected = (batch, *trailing)
    if result.shape == (batch, 1, *trailing):
        result = result[:, 0]
    if result.shape != expected:
        raise ValueError("ordinary target eligibility shape differs")
    return result


def _timed_component_call(
    name: str,
    *,
    reference,
    callback: Callable[[str, float], None] | None,
    operation: Callable[[], Any],
):
    """Time one exact loss component only for the bounded diagnostic worker.

    The ordinary training path supplies no callback and therefore performs no
    synchronization or extra timing work.  The diagnostic path synchronizes
    around the existing operation so its wall-clock partition includes every
    queued CUDA kernel attributable to that component.
    """

    if callback is None:
        return operation()
    import torch

    device = torch.as_tensor(reference).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    callback(name, time.perf_counter() - started)
    return result


def _raw_representation_components(
    *,
    execution: NodeExecution,
    model,
    surfaces,
    targets: Mapping[str, Any],
    labels,
    class_weights,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    components: Sequence[str],
    timing_callback: Callable[[str, float], None] | None = None,
) -> RawRepresentationComponents:
    """Compute selected raw components entirely in FP32."""

    import torch

    requested = tuple(components)
    if len(set(requested)) != len(requested) or not set(requested) <= set(execution.active_components):
        raise ValueError("requested representation component registry differs")
    heads = dict(model.representation_heads.projection_items())
    tokens = surfaces.particle_block_2.float()
    vectors = surfaces.vectors.float()
    mask = surfaces.particle_mask.bool()
    visible = surfaces.visible_indices.long()
    family = surfaces.family_codes.to(torch.int8)
    teacher_jet = targets.get("jet_penultimate")
    jet_result = None
    set_result = None
    relation_result = None
    losses: dict[str, Any] = {}
    rows: dict[str, CalibrationComponentRows] = {}
    if "jet" in requested:
        if teacher_jet is None:
            raise ValueError("target bank lacks pooled jet representation")
        jet_result = _timed_component_call(
            "jet_representation_loss", reference=tokens,
            callback=timing_callback,
            operation=lambda: jet_representation_loss(
                surfaces.jet_penultimate.float(), teacher_jet.float(), heads["jet"],
                labels=labels, class_weights=class_weights,
            ),
        )
        losses["jet"] = jet_result.loss
        rows["jet"] = CalibrationComponentRows(
            per_jet=0.75 * jet_result.direct_rows + 0.25 * jet_result.gram,
            eligible=torch.ones(len(labels), dtype=torch.bool, device=labels.device),
            support={"eligible_rows": len(labels)},
            loss=jet_result.loss,
        )
    if "set" in requested:
        if execution.teacher_latent_domain == "ordinary":
            target = targets.get("token_kernel_mean")
            eligible = targets.get("token_family_eligibility")
            if target is None or eligible is None or "token" not in heads:
                raise ValueError("ordinary set target/head is absent")
            eligible = _ordinary_eligibility(eligible, batch=len(labels), trailing=())
            set_result = _timed_component_call(
                "set_representation_loss", reference=tokens,
                callback=timing_callback,
                operation=lambda: ordinary_set_representation_loss(
                    tokens, vectors, mask, target, eligible, heads["token"],
                    token_resources, labels=labels, class_weights=class_weights,
                ),
            )
        else:
            required = (
                "token_kernel_mean_charged", "token_kernel_mean_neutral",
                "token_family_eligibility",
            )
            if any(name not in targets for name in required):
                raise ValueError("TOFF set target is absent")
            target = torch.stack(
                (targets["token_kernel_mean_charged"], targets["token_kernel_mean_neutral"]),
                dim=1,
            )
            set_result = _timed_component_call(
                "set_representation_loss", reference=tokens,
                callback=timing_callback,
                operation=lambda: native_offline_set_representation_loss(
                    tokens, vectors, mask, family, target,
                    targets["token_family_eligibility"].bool(),
                    (heads["token_charged"], heads["token_neutral"]),
                    token_resources, labels=labels, class_weights=class_weights,
                ),
            )
        losses["set"] = set_result.reduction.loss
        rows["set"] = CalibrationComponentRows(
            per_jet=set_result.reduction.per_jet,
            eligible=set_result.reduction.eligible,
            support={"eligible_rows": set_result.reduction.eligible_count},
            loss=set_result.reduction.loss,
        )
    if "relation" in requested:
        if execution.teacher_latent_domain == "ordinary":
            target = targets.get("relation_kernel_mean")
            eligible = targets.get("relation_eligibility")
            if target is None or eligible is None:
                raise ValueError("ordinary relation target is absent")
            # Relation KD is deliberately basis invariant.  It compares
            # cosine relations in the raw contextual particle-block-2 state;
            # the set projection is not part of this objective.
            target = target.float()
            if target.shape == (len(labels), 3, relation_resources.total_features):
                target = target[:, None, :, :]
            eligible = _ordinary_eligibility(eligible, batch=len(labels), trailing=(3,))
            eligible = eligible[:, None, :]
            relation_result = _timed_component_call(
                "relation_representation_loss", reference=tokens,
                callback=timing_callback,
                operation=lambda: relation_representation_loss(
                    tokens, vectors, mask, visible, target, eligible,
                    relation_resources, labels=labels, class_weights=class_weights,
                ),
            )
        else:
            required = (
                "relation_kernel_mean_charged", "relation_kernel_mean_neutral",
                "relation_eligibility",
            )
            if any(name not in targets for name in required):
                raise ValueError("TOFF relation target is absent")
            target = torch.stack(
                (
                    targets["relation_kernel_mean_charged"],
                    targets["relation_kernel_mean_neutral"],
                ),
                dim=1,
            )
            relation_result = _timed_component_call(
                "relation_representation_loss", reference=tokens,
                callback=timing_callback,
                operation=lambda: relation_representation_loss(
                    tokens, vectors, mask, visible, target,
                    targets["relation_eligibility"].bool(), relation_resources,
                    labels=labels, class_weights=class_weights,
                    family_codes=family,
                ),
            )
        losses["relation"] = relation_result.reduction.loss
        rows["relation"] = CalibrationComponentRows(
            per_jet=relation_result.reduction.per_jet,
            eligible=relation_result.reduction.eligible,
            support={"eligible_rows": relation_result.reduction.eligible_count},
            loss=relation_result.reduction.loss,
        )
    return RawRepresentationComponents(jet_result, set_result, relation_result, losses, rows)


@dataclass(frozen=True)
class NodeLossResult:
    total: Any
    base: Mapping[str, Any]
    representation_total: Any
    scheduled: Any
    raw_components: RawRepresentationComponents | None
    reporting_components: Mapping[str, Any]


def _control_scheduled_loss(
    execution: NodeExecution,
    *,
    effective_pass: float,
    scaled: Mapping[str, Any],
    orthogonality,
):
    """Apply the two registered component-control schedules."""

    import torch
    from .hcwdl_representation_losses import ScheduledRepresentationLoss

    js = jet_set_ramp(effective_pass)
    if execution.jet_only:
        jet_coefficient, set_coefficient, relation_coefficient = js, 0.0, 0.0
        scientific = jet_coefficient * scaled["jet"]
    else:
        jet_coefficient, set_coefficient, relation_coefficient = 0.4 * js, 0.6 * js, 0.0
        scientific = jet_coefficient * scaled["jet"] + set_coefficient * scaled["set"]
    orthogonal = js * ORTHOGONALITY_COEFFICIENT * orthogonality.float()
    total = RHO_REPRESENTATION * (scientific + orthogonal)
    if not all(torch.isfinite(value) for value in (scientific, orthogonal, total)):
        raise FloatingPointError("controlled representation schedule is nonfinite")
    return ScheduledRepresentationLoss(
        total=total,
        scientific=scientific,
        orthogonality=orthogonal,
        jet_coefficient=jet_coefficient,
        set_coefficient=set_coefficient,
        relation_coefficient=relation_coefficient,
        ramp_jet_set=js,
        ramp_relation=relation_ramp(effective_pass),
    )


def compute_node_loss(
    *,
    execution: NodeExecution,
    model,
    surfaces,
    labels,
    class_weights,
    privileged_targets: Mapping[str, Any],
    predecessor_logits,
    calibration_scales: Mapping[str, float],
    effective_pass: float,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    force_components: bool = False,
    timing_callback: Callable[[str, float], None] | None = None,
) -> NodeLossResult:
    """Compute the locked base objective plus the exact scheduled auxiliary."""

    import torch

    configuration = node_base_loss_configuration(execution)
    hlt_logits = privileged_targets["logits"].float()
    privileged_for_base = None
    if predecessor_logits is not None:
        raise ValueError("dense descent forbids a second predecessor-logit teacher")
    base = _timed_component_call(
        "base_logit_loss", reference=surfaces.logits,
        callback=timing_callback,
        operation=lambda: hcwdl_base_loss(
            surfaces.logits.float(), labels,
            class_weights=class_weights,
            configuration=configuration,
            hlt_teacher_logits=hlt_logits,
            privileged_teacher_logits=privileged_for_base,
        ),
    )
    js = jet_set_ramp(effective_pass)
    rel = relation_ramp(effective_pass)
    required: list[str] = []
    if force_components or js > 0:
        required.extend(name for name in execution.active_components if name != "relation")
    if "relation" in execution.active_components and (force_components or rel > 0):
        required.append("relation")
    if not required:
        zero = base["total"] * 0.0
        return NodeLossResult(
            total=base["total"], base=base, representation_total=zero,
            scheduled=None, raw_components=None,
            reporting_components={
                "representation_total": zero,
                "ramp_jet_set": zero,
                "ramp_relation": zero,
                "effective_jet_coefficient": zero,
                "effective_set_coefficient": zero,
                "effective_relation_coefficient": zero,
            },
        )
    for name in required:
        if name not in calibration_scales:
            raise ValueError(f"calibration scale for {name!r} is absent")
        scale = float(calibration_scales[name])
        if not math.isfinite(scale) or scale < 0:
            raise FloatingPointError("calibration scale is invalid")
    raw = _raw_representation_components(
        execution=execution, model=model, surfaces=surfaces,
        targets=privileged_targets, labels=labels, class_weights=class_weights,
        token_resources=token_resources, relation_resources=relation_resources,
        components=required, timing_callback=timing_callback,
    )
    scaled = {
        name: raw.losses[name] * float(calibration_scales[name]) for name in required
    }
    orthogonality = projection_orthogonality(
        dict(model.representation_heads.projection_items()),
    )
    if execution.is_control and (
        execution.jet_only or not execution.relation_enabled
    ):
        scheduled = _control_scheduled_loss(
            execution, effective_pass=effective_pass, scaled=scaled,
            orthogonality=orthogonality,
        )
    else:
        scheduled = scheduled_representation_loss(
            strategy=execution.short_strategy,
            effective_pass=effective_pass,
            scaled_jet=scaled["jet"],
            scaled_set=scaled["set"],
            scaled_relation=scaled.get("relation"),
            orthogonality=orthogonality,
        )
    total = base["total"] + scheduled.total
    if not torch.isfinite(total):
        raise FloatingPointError("HCWDL-RKD total loss is nonfinite")
    reporting: dict[str, Any] = {
        "representation_total": scheduled.total,
        "representation_scientific": scheduled.scientific,
        "representation_orthogonality": scheduled.orthogonality,
        "raw_orthogonality": orthogonality,
        "ramp_jet_set": torch.as_tensor(
            scheduled.ramp_jet_set, device=total.device,
        ),
        "ramp_relation": torch.as_tensor(
            scheduled.ramp_relation, device=total.device,
        ),
        "effective_jet_coefficient": torch.as_tensor(
            scheduled.jet_coefficient, device=total.device,
        ),
        "effective_set_coefficient": torch.as_tensor(
            scheduled.set_coefficient, device=total.device,
        ),
        "effective_relation_coefficient": torch.as_tensor(
            scheduled.relation_coefficient, device=total.device,
        ),
    }
    reporting.update({f"raw_{name}": value for name, value in raw.losses.items()})
    reporting.update({f"calibrated_{name}": value for name, value in scaled.items()})
    return NodeLossResult(total, base, scheduled.total, scheduled, raw, reporting)


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        # Optimizer state dictionaries intentionally use integer parameter
        # identifiers.  Converting every key to text makes the serialized
        # state look valid while silently preventing ``Optimizer.load_state_dict``
        # from reconnecting those moments to the integer IDs in param_groups.
        # Preserve the exact key type; the resume inventory already validates
        # that keys are strings or integers.
        return {name: _cpu_tree(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return copy.deepcopy(value)


class _IntervalMeans:
    def __init__(self, device) -> None:
        self.device = device
        self.example_count = 0
        self.batch_count = 0
        self.sums: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def add(self, values: Mapping[str, Any], examples: int) -> None:
        import torch

        if examples <= 0:
            raise ValueError("interval accumulator received an empty batch")
        self.example_count += int(examples)
        self.batch_count += 1
        for name, value in values.items():
            scalar = torch.as_tensor(value, device=self.device).detach().float()
            if scalar.ndim != 0 or not torch.isfinite(scalar):
                raise FloatingPointError("interval loss value is invalid")
            contribution = scalar.to(torch.float64) * examples
            self.sums[name] = self.sums.get(
                name, torch.zeros((), dtype=torch.float64, device=self.device),
            ) + contribution

    def flush(self, *, update: int, partial: bool = False) -> dict[str, Any] | None:
        if self.example_count == 0:
            return None
        row = {
            "through_update": int(update),
            "examples": self.example_count,
            "batches": self.batch_count,
            "partial": bool(partial),
            "means": {
                name: float((value / self.example_count).cpu())
                for name, value in sorted(self.sums.items())
            },
        }
        self.history.append(row)
        self.example_count = 0
        self.batch_count = 0
        self.sums = {}
        return row

    def state_dict(self) -> dict[str, Any]:
        return {
            "example_count": self.example_count,
            "batch_count": self.batch_count,
            "sums": _cpu_tree(self.sums),
            "history": copy.deepcopy(self.history),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        import torch

        if set(state) != {"example_count", "batch_count", "sums", "history"}:
            raise ValueError("interval-aggregate resume fields differ")
        self.example_count = int(state["example_count"])
        self.batch_count = int(state["batch_count"])
        self.sums = {
            str(name): torch.as_tensor(value, device=self.device, dtype=torch.float64)
            for name, value in state["sums"].items()
        }
        self.history = copy.deepcopy(list(state["history"]))


def _optimizer_for(model, config: RepresentationTrainingConfiguration):
    import torch

    exclusions = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name in exclusions else decay).append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("HCWDL-RKD optimizer has no trainable parameters")
    return torch.optim.AdamW(
        groups,
        lr=config.peak_learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
    )


def _learning_rate(config: RepresentationTrainingConfiguration, update: int) -> float:
    total = config.active_total_updates
    warmup = max(1, round(total * config.warmup_fraction))
    if update < warmup:
        return config.peak_learning_rate * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return config.peak_learning_rate * (
        config.minimum_lr_fraction + (1 - config.minimum_lr_fraction) * cosine
    )


def _head_state(model) -> dict[str, Any]:
    return {
        name: _cpu_tree(projection.state_dict())
        for name, projection in model.representation_heads.projection_items()
    }


def _load_head_state(model, state: Mapping[str, Any]) -> None:
    projections = dict(model.representation_heads.projection_items())
    if set(projections) != set(state):
        raise ValueError("resume representation-head topology differs")
    for name, projection in projections.items():
        result = projection.load_state_dict(state[name], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("resume representation-head state differs")


def _scheduler_state(config: RepresentationTrainingConfiguration, update: int) -> dict[str, Any]:
    return {
        "kind": "HCWDL_WARMUP_COSINE_CLOSED_FORM/v1",
        "completed_updates": int(update),
        "active_total_updates": config.active_total_updates,
        "peak_learning_rate": config.peak_learning_rate,
        "warmup_fraction": config.warmup_fraction,
        "minimum_lr_fraction": config.minimum_lr_fraction,
    }


def _state_dict_logical_sha256(state: Mapping[str, Any]) -> str:
    """Hash an exact tensor state without depending on ``torch.save`` bytes."""

    import torch

    rows: dict[str, str] = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("state-dict names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError("state-dict values must be tensors")
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        rows[name] = logical_array_sha256_from_byte_hash(
            name=name,
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            c_order_byte_sha256=sha256_bytes(raw),
            byte_length=len(raw),
        )
    return canonical_sha256(rows)


def _default_calibration_state(
    *,
    diagnostic_batch_sha256: str | None,
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "artifact_hashes": {},
        "diagnostic_batch_sha256": diagnostic_batch_sha256,
        "selection_sha256": (
            None if selection is None else selection["content_hash"]
        ),
        "ordered_selection_sha256": (
            None if selection is None else selection["ordered_selection_sha256"]
        ),
        "ordered_calibration_identity_sha256": (
            None
            if selection is None
            else selection["canonical_identity_order_sha256"]
        ),
        "calibration_rows": (
            None if selection is None else int(selection["actual_rows"])
        ),
        "components": {
            name: {
                "status": "pending",
                "scale": 0.0,
                "history": [],
            }
            for name in ("jet", "set", "relation")
        },
    }


def _calibration_scales(
    calibration: Mapping[str, Any], execution: NodeExecution, effective_pass: float,
) -> dict[str, float]:
    values = {
        name: float(calibration["components"][name]["scale"])
        for name in execution.active_components
    }
    required = []
    if jet_set_ramp(effective_pass) > 0:
        required.extend(name for name in execution.active_components if name != "relation")
    if "relation" in execution.active_components and relation_ramp(effective_pass) > 0:
        required.append("relation")
    for name in required:
        if calibration["components"][name]["status"] not in {
            "active", "inactive_valid_support",
        }:
            raise RuntimeError(f"representation component {name!r} was not calibrated")
    return values


def _state_snapshot(
    *,
    model,
    optimizer,
    config: RepresentationTrainingConfiguration,
    completed_pass: int,
    completed_update: int,
    next_canonical_batch: int,
    interval: _IntervalMeans,
    validation_history: Sequence[Mapping[str, Any]],
    selection_state: Mapping[str, Any],
    calibration: Mapping[str, Any],
    target_bindings: Mapping[str, str],
    rng_streams: Mapping[str, Any],
    producer_runtime_signature: Mapping[str, Any],
    sampler_external_state: Mapping[str, Any],
    pass_identity_digests: np.ndarray,
) -> dict[str, Any]:
    runtime = capture_model_runtime_state(model)
    return {
        "deployable_model": _cpu_tree(model.deployable_model.state_dict()),
        "representation_heads": _head_state(model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": _scheduler_state(config, completed_update),
        "sampler": {
            "protocol": "canonical_pass_batch_index/v1",
            "external": _cpu_tree(sampler_external_state),
            "pass_identity_digests": np.ascontiguousarray(
                pass_identity_digests, dtype=np.uint8,
            ).copy(),
        },
        "trimmer": copy.deepcopy(runtime["trimmers"]),
        "rng": _cpu_tree(capture_rng_state()),
        "rng_streams": copy.deepcopy(dict(rng_streams)),
        "model_runtime": _cpu_tree(runtime),
        "cursor": {
            "completed_pass": int(completed_pass),
            "completed_update": int(completed_update),
            "next_canonical_batch": int(next_canonical_batch),
        },
        "interval_aggregates": interval.state_dict(),
        "validation_history": copy.deepcopy(list(validation_history)),
        "selection_state": copy.deepcopy(dict(selection_state)),
        "calibration": _cpu_tree(calibration),
        "target_bindings": dict(target_bindings),
        "producer_runtime_signature": copy.deepcopy(dict(producer_runtime_signature)),
    }


def _restore_state(
    state: Mapping[str, Any],
    *,
    model,
    optimizer,
    config: RepresentationTrainingConfiguration,
    interval: _IntervalMeans,
    sampler_external_restore: Callable[[Mapping[str, Any]], None] | None,
    expected_rng_streams: Mapping[str, Any],
) -> tuple[
    int, int, int, list[dict[str, Any]], dict[str, Any], dict[str, Any], np.ndarray,
]:
    result = model.deployable_model.load_state_dict(state["deployable_model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("resume deployable state differs")
    _load_head_state(model, state["representation_heads"])
    optimizer.load_state_dict(state["optimizer"])
    expected_scheduler = _scheduler_state(config, int(state["cursor"]["completed_update"]))
    if state["scheduler"] != expected_scheduler:
        raise ValueError("resume scheduler state differs")
    restore_model_runtime_state(model, state["model_runtime"])
    if state["trimmer"] != state["model_runtime"]["trimmers"]:
        raise ValueError("resume trimmer namespace differs from model runtime")
    restore_rng_state(state["rng"])
    if state.get("rng_streams") != expected_rng_streams:
        raise ValueError("resume paired RNG stream contract differs")
    interval.load_state_dict(state["interval_aggregates"])
    sampler = state["sampler"]
    if sampler.get("protocol") != "canonical_pass_batch_index/v1":
        raise ValueError("resume sampler protocol differs")
    pass_identities = np.asarray(sampler.get("pass_identity_digests"))
    if (
        pass_identities.dtype != np.uint8
        or pass_identities.ndim != 2
        or pass_identities.shape[1] != 32
        or len({bytes(row) for row in pass_identities}) != len(pass_identities)
    ):
        raise ValueError("resume in-pass identity registry differs")
    if sampler_external_restore is not None:
        sampler_external_restore(copy.deepcopy(sampler["external"]))
    cursor = state["cursor"]
    selection = copy.deepcopy(dict(state["selection_state"]))
    if selection.get("checkpoint_path") is not None:
        path = Path(selection["checkpoint_path"])
        if not path.is_file() or sha256_file(path) != require_sha256(
            selection["checkpoint_sha256"], name="selected training checkpoint SHA-256",
        ):
            raise ValueError("resume selected training checkpoint differs")
    return (
        int(cursor["completed_pass"]),
        int(cursor["completed_update"]),
        int(cursor["next_canonical_batch"]),
        copy.deepcopy(list(state["validation_history"])),
        selection,
        copy.deepcopy(dict(state["calibration"])),
        np.ascontiguousarray(pass_identities).copy(),
    )


def _validate_runtime_lineage(
    resume_lineage: Mapping[str, Any], producer_runtime_signature: Mapping[str, Any],
    *, expected_graph_sha256: str,
) -> dict[str, str]:
    if set(resume_lineage) != REQUIRED_LINEAGE_KEYS:
        raise ValueError("HCWDL-RKD resume lineage fields differ")
    normalized = {
        name: require_sha256(value, name=f"resume lineage {name}")
        for name, value in resume_lineage.items()
    }
    expected_graph_sha256 = require_sha256(
        expected_graph_sha256, name="registered execution graph SHA-256",
    )
    if normalized["ascent_graph"] != expected_graph_sha256:
        raise ValueError("HCWDL-RKD resume graph hash differs")
    if producer_runtime_signature.get("content_hash") != normalized[
        "producer_runtime_signature"
    ]:
        raise ValueError("producer runtime signature lineage differs")
    return normalized


def _prune_checkpoint_candidates_for_resume(
    candidate_root: str | Path,
    *,
    resume_root: str | Path,
    lineage: Mapping[str, str],
) -> tuple[Path, ...]:
    """Retain candidates referenced by every still-valid resume generation."""

    retained = scan_resume_generations(
        resume_root, expected_lineage=lineage,
    ).valid_generations
    referenced = {
        Path(path).resolve()
        for generation in retained
        for path in [generation.state["selection_state"].get("checkpoint_path")]
        if path is not None
    }
    removed: list[Path] = []
    for candidate in Path(candidate_root).glob("*.pt"):
        if candidate.resolve() not in referenced:
            candidate.unlink()
            removed.append(candidate)
    return tuple(removed)


def _validation(
    model,
    batches: Iterable[Mapping[str, Any]],
    *,
    device,
    amp_dtype: str,
) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    import torch

    prior_mode = model.training
    runtime = capture_model_runtime_state(model)
    rng = capture_rng_state()
    logits_parts = []
    label_parts = []
    parity_inputs = None
    model.eval()
    try:
        with torch.inference_mode():
            for raw in batches:
                batch = normalize_hlt_batch(raw)
                features, vectors, mask, _, _, labels = _batch_tensors(batch, device)
                if parity_inputs is None:
                    parity_inputs = (
                        features.detach().clone(), vectors.detach().clone(), mask.detach().clone(),
                    )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_dtype == "bfloat16",
                ):
                    logits = model(features, vectors, mask)
                logits = logits.float()
                if logits.shape != (len(labels), 15) or not torch.isfinite(logits).all():
                    raise FloatingPointError("validation logits are invalid")
                logits_parts.append(logits.cpu().numpy())
                label_parts.append(labels.cpu().numpy())
    finally:
        restore_model_runtime_state(model, runtime)
        restore_rng_state(rng)
        model.train(prior_mode)
    if not logits_parts or parity_inputs is None:
        raise ValueError("validation stream is empty")
    metrics = classification_metrics(
        np.concatenate(logits_parts, axis=0), np.concatenate(label_parts, axis=0),
    )
    required = (
        "cross_entropy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
    )
    if any(metrics.get(name) is None or not math.isfinite(float(metrics[name])) for name in required):
        raise FloatingPointError("required validation metrics are nonfinite")
    return metrics, parity_inputs


class _NoMemoPickler(pickle.Pickler):
    """Canonical acyclic checkpoint pickler independent of object aliasing.

    Exact resume may reconstruct equal dictionaries with different incidental
    Python object sharing (for example, repeated metric strings loaded from a
    prior pickle).  Standard pickle memoizes those identities, making equal
    checkpoint trees serialize to different bytes.  Selected checkpoints are
    acyclic, so disabling object memoization removes that non-scientific byte
    difference while Torch's persistent storage IDs continue to encode tensor
    storage correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast = True


class _CanonicalPickleModule:
    Pickler = _NoMemoPickler
    Unpickler = pickle.Unpickler
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def _torch_bytes(value: Any) -> bytes:
    import torch

    stream = BytesIO()
    torch.save(value, stream, pickle_module=_CanonicalPickleModule)
    return stream.getvalue()


def _publish_selected_training_checkpoint(
    output_dir: Path,
    *,
    execution: NodeExecution,
    row: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Path, str]:
    checkpoint_id = str(row["checkpoint_id"])
    payload = {
        "contract": SELECTED_TRAINING_CHECKPOINT_CONTRACT,
        "schema_version": 1,
        "execution_id": execution.execution_id,
        "checkpoint_id": checkpoint_id,
        "completed_pass": int(row["completed_pass"]),
        "completed_update": int(row["update"]),
        "validation": copy.deepcopy(dict(row["validation"])),
        "state": _cpu_tree(state),
    }
    data = _torch_bytes(payload)
    # Boundary candidates are private staging artifacts.  Only the winning
    # training state is exposed later through the committed selected envelope.
    path = (
        output_dir / "checkpoints" / "selected" / "staging" / "candidates"
        / f"{checkpoint_id}.pt"
    )
    atomic_publish_bytes(path, data)
    return path, sha256_bytes(data)


def _load_selected_training_checkpoint(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    import torch

    if sha256_file(path) != require_sha256(expected_sha256, name="selected checkpoint SHA-256"):
        raise ValueError("selected training checkpoint bytes differ")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get("contract") != SELECTED_TRAINING_CHECKPOINT_CONTRACT:
        raise ValueError("selected training checkpoint contract differs")
    if value.get("schema_version") != 1 or not isinstance(value.get("state"), Mapping):
        raise ValueError("selected training checkpoint payload differs")
    return value


def _calibration_result_payload(result: GradientCalibrationResult) -> dict[str, Any]:
    return {
        "contract": result.contract,
        "components": {
            name: asdict(component) for name, component in sorted(result.components.items())
        },
        "parameter_names": list(result.parameter_names),
        "parameter_shapes": [list(shape) for shape in result.parameter_shapes],
        "parameter_scalar_count": result.parameter_scalar_count,
        "forward_calls": result.forward_calls,
    }


def _materialize_diagnostic_batch(
    provider: Callable[[], Mapping[str, Any]],
    *,
    execution: NodeExecution,
    mode: str,
    lineage: Mapping[str, str],
    representation_recipe_sha256: str,
    output: Path,
    calibration_directory: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Freeze and authenticate the independent first calibration microbatch."""

    raw = normalize_hlt_batch(provider())
    rows = len(raw.identity_digests)
    if rows <= 0 or rows > 256:
        raise ValueError("HCWDL-RKD diagnostic microbatch must contain 1..256 rows")
    if mode == "scientific" and rows != 256:
        raise ValueError("scientific HCWDL-RKD diagnostic microbatch requires 256 rows")
    frozen = {
        "features": np.asarray(raw.features).copy(),
        "vectors": np.asarray(raw.vectors).copy(),
        "mask": np.asarray(raw.mask).copy(),
        "visible_indices": np.asarray(raw.visible_indices).copy(),
        "family_codes": np.asarray(raw.family_codes).copy(),
        "labels": np.asarray(raw.labels).copy(),
        "identity_digests": np.asarray(raw.identity_digests).copy(),
    }
    if raw.family_reason_codes is not None:
        frozen["family_reason_codes"] = np.asarray(
            raw.family_reason_codes,
        ).copy()
    array_hashes = {
        name: logical_array_sha256(name, value)
        for name, value in sorted(frozen.items())
    }
    parents = {
        "execution": lineage["execution"],
        "representation_recipe": representation_recipe_sha256,
        "target_generation": lineage["target_generation"],
        "target_logical": lineage["target_logical"],
    }
    artifact = build_versioned_artifact(
        DIAGNOSTIC_BATCH_CONTRACT,
        parents=parents,
        payload={
            "execution_id": execution.execution_id,
            "batch_protocol": BATCH_PROTOCOL,
            "selection": "first_canonical_calibration_microbatch",
            "train_only": True,
            "rows": rows,
            "ordered_identity_sha256": identity_order_sha256(
                frozen["identity_digests"],
            ),
            "identity_set_sha256": identity_set_sha256(
                frozen["identity_digests"],
            ),
            "array_logical_sha256": array_hashes,
        },
    )
    path = (
        output / "calibration"
        if calibration_directory is None
        else Path(calibration_directory)
    ) / "diagnostic_batch.json"
    write_immutable_json(path, artifact)
    published = load_json(path)
    validate_versioned_artifact(
        published,
        expected_contract=DIAGNOSTIC_BATCH_CONTRACT,
        expected_parents=parents,
        required_payload_keys=(
            "execution_id", "batch_protocol", "selection", "train_only",
            "rows", "ordered_identity_sha256", "identity_set_sha256",
            "array_logical_sha256",
        ),
    )
    if published != artifact:
        raise ValueError("HCWDL-RKD diagnostic-batch artifact differs")
    return frozen, artifact


def _tree_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for diagnostic non-mutation assertions."""

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _gradient_tensors(loss, named_parameters, *, retain_graph: bool):
    import torch

    parameters = tuple(parameter for _, parameter in named_parameters)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, create_graph=False,
        allow_unused=False,
    )
    if len(gradients) != len(parameters):
        raise RuntimeError("diagnostic gradient support differs")
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None or gradient.shape != parameter.shape:
            raise RuntimeError("diagnostic gradient is disconnected")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("diagnostic gradient is nonfinite")
    return gradients


def _gradient_rms(gradients) -> float:
    import torch

    numerator = torch.zeros((), dtype=torch.float64, device=gradients[0].device)
    count = 0
    for gradient in gradients:
        numerator = numerator + gradient.detach().double().square().sum()
        count += gradient.numel()
    value = torch.sqrt(numerator / count)
    if not torch.isfinite(value):
        raise FloatingPointError("diagnostic gradient RMS is nonfinite")
    return float(value.cpu())


def _gradient_comparison(base_gradients, representation_gradients) -> dict[str, Any]:
    import torch

    base_squared = torch.zeros(
        (), dtype=torch.float64, device=base_gradients[0].device,
    )
    representation_squared = torch.zeros_like(base_squared)
    dot = torch.zeros_like(base_squared)
    for base, representation in zip(
        base_gradients, representation_gradients, strict=True,
    ):
        left = base.detach().double()
        right = representation.detach().double()
        base_squared = base_squared + left.square().sum()
        representation_squared = representation_squared + right.square().sum()
        dot = dot + (left * right).sum()
    base_norm = torch.sqrt(base_squared)
    representation_norm = torch.sqrt(representation_squared)
    if not all(torch.isfinite(value) for value in (base_norm, representation_norm, dot)):
        raise FloatingPointError("diagnostic gradient comparison is nonfinite")
    base_value = float(base_norm.cpu())
    representation_value = float(representation_norm.cpu())
    ratio = None if base_value == 0 else representation_value / base_value
    cosine = (
        None
        if base_value == 0 or representation_value == 0
        else float((dot / (base_norm * representation_norm)).cpu())
    )
    return {
        "base_gradient_norm": base_value,
        "representation_gradient_norm": representation_value,
        "representation_to_base_ratio": ratio,
        "gradient_cosine": cosine,
        "status": (
            "zero_base_gradient" if base_value == 0 else
            "zero_representation_gradient" if representation_value == 0 else
            "active"
        ),
    }


def _diagnostic_scientific_loss(
    execution: NodeExecution,
    *,
    effective_pass: float,
    scaled: Mapping[str, Any],
    zero,
):
    js = jet_set_ramp(effective_pass)
    rel = relation_ramp(effective_pass)
    if execution.jet_only:
        return js * scaled.get("jet", zero), {
            "jet": js, "set": 0.0, "relation": 0.0,
        }
    if execution.relation_enabled:
        common = js - 0.25 * rel
        return (
            0.4 * common * scaled.get("jet", zero)
            + 0.6 * common * scaled.get("set", zero)
            + 0.25 * rel * scaled.get("relation", zero)
        ), {
            "jet": 0.4 * common,
            "set": 0.6 * common,
            "relation": 0.25 * rel,
        }
    return (
        0.4 * js * scaled.get("jet", zero)
        + 0.6 * js * scaled.get("set", zero)
    ), {"jet": 0.4 * js, "set": 0.6 * js, "relation": 0.0}


def _projection_diagnostic_payload(model) -> list[dict[str, Any]]:
    """Serialize finite weights even when their report-only condition is infinite."""

    rows = []
    for value in projection_diagnostics(
        dict(model.representation_heads.projection_items()),
    ):
        condition = float(value.condition_number)
        rows.append({
            "name": value.name,
            "singular_values": list(value.singular_values),
            "minimum_singular_value": min(value.singular_values),
            "maximum_singular_value": max(value.singular_values),
            "condition_number": condition if math.isfinite(condition) else None,
            "condition_number_status": (
                "finite" if math.isfinite(condition) else "infinite_zero_minimum"
            ),
            "poorly_conditioned": bool(value.poorly_conditioned),
            "orthogonality_loss": float(value.orthogonality_loss),
        })
    return rows


def run_representation_diagnostic(
    *,
    execution: NodeExecution,
    model,
    optimizer,
    batch: Mapping[str, Any],
    completed_pass: int,
    completed_update: int,
    device,
    amp_dtype: str,
    class_weights,
    target_bank,
    predecessor_bank: InMemoryPredecessorLogits | None,
    shuffled_representation_joiner,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    calibration: Mapping[str, Any],
    parameter_selector: Callable[[Any], Sequence[tuple[str, Any]]] = early_backbone_parameters,
    external_snapshot: Callable[[], object] | None = None,
    external_restore: Callable[[object], None] | None = None,
) -> dict[str, Any]:
    """Run the immutable one-forward post-validation diagnostic non-mutatively."""

    import torch

    if (external_snapshot is None) != (external_restore is None):
        raise ValueError("diagnostic external snapshot/restore must be supplied together")
    normalized = normalize_hlt_batch(batch)
    prior_model_state = _cpu_tree(model.state_dict())
    prior_runtime = capture_model_runtime_state(model)
    prior_modes = {name: module.training for name, module in model.named_modules()}
    prior_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    prior_optimizer = _cpu_tree(optimizer.state_dict())
    prior_rng = _cpu_tree(capture_rng_state())
    prior_external = (
        None if external_snapshot is None else _cpu_tree(external_snapshot())
    )
    seed_payload = {
        "contract": "HCWDL_REP_DIAGNOSTIC/v1",
        "execution_id": execution.execution_id,
        "completed_update": int(completed_update),
    }
    seed_sha256 = canonical_sha256(seed_payload)
    seed = int(seed_sha256[:16], 16) % (2**63 - 1)
    result: dict[str, Any] | None = None
    try:
        model.eval()
        cuda_devices = (
            [] if device.type != "cuda" else [
                torch.cuda.current_device() if device.index is None else device.index
            ]
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            features, vectors, mask, visible, family, labels = _batch_tensors(
                normalized, device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_dtype == "bfloat16",
            ):
                surfaces = model.forward_hcwdl_surfaces(
                    features, vectors, mask, visible, family,
                )
            targets = _target_tensors(
                target_bank, normalized.identity_digests, device=device,
                execution=execution,
                shuffled_representation_joiner=shuffled_representation_joiner,
            )
            predecessor = (
                None if predecessor_bank is None else torch.as_tensor(
                    predecessor_bank.join(normalized.identity_digests),
                    device=device, dtype=torch.float32,
                ).detach()
            )
            configuration = node_base_loss_configuration(execution)
            if predecessor is not None:
                raise ValueError(
                    "dense descent diagnostic forbids a second predecessor-logit teacher"
                )
            hlt_logits = targets["logits"].float()
            privileged_for_base = None
            base = hcwdl_base_loss(
                surfaces.logits.float(), labels, class_weights=class_weights,
                configuration=configuration, hlt_teacher_logits=hlt_logits,
                privileged_teacher_logits=privileged_for_base,
            )
            base_rows = hcwdl_base_loss_rows(
                surfaces.logits.float(), labels, class_weights=class_weights,
                configuration=configuration, hlt_teacher_logits=hlt_logits,
                privileged_teacher_logits=privileged_for_base,
            )["total_rows"]
            calibrated_names = tuple(
                name for name in execution.active_components
                if calibration["components"][name]["status"] in {
                    "active", "inactive_valid_support",
                }
            )
            raw = (
                None if not calibrated_names else _raw_representation_components(
                    execution=execution, model=model, surfaces=surfaces,
                    targets=targets, labels=labels, class_weights=class_weights,
                    token_resources=token_resources,
                    relation_resources=relation_resources,
                    components=calibrated_names,
                )
            )
            named_parameters = tuple(parameter_selector(model))
            if not named_parameters or len({name for name, _ in named_parameters}) != len(
                named_parameters
            ):
                raise ValueError("diagnostic parameter support is empty or duplicated")
            component_rows: dict[str, Any] = {}
            scaled: dict[str, Any] = {}
            for name in ("jet", "set", "relation"):
                if name not in execution.active_components:
                    component_rows[name] = {
                        "status": "not_part_of_strategy", "raw_loss": None,
                        "scale": None, "scaled_loss": None,
                        "base_gradient_rms_on_support": None,
                        "raw_gradient_rms": None,
                        "scaled_gradient_rms": None,
                    }
                    continue
                component_state = calibration["components"][name]
                if component_state["status"] == "pending":
                    component_rows[name] = {
                        "status": "not_yet_calibrated", "raw_loss": None,
                        "scale": None, "scaled_loss": None,
                        "base_gradient_rms_on_support": None,
                        "raw_gradient_rms": None,
                        "scaled_gradient_rms": None,
                    }
                    continue
                if raw is None or name not in raw.rows:
                    raise RuntimeError("diagnostic calibrated component is absent")
                rows = raw.rows[name]
                eligible = torch.as_tensor(
                    rows.eligible, device=device, dtype=torch.bool,
                )
                scale = float(component_state["scale"])
                raw_loss = raw.losses[name]
                scaled_loss = scale * raw_loss
                scaled[name] = scaled_loss
                if not bool(eligible.any()):
                    component_rows[name] = {
                        "status": "no_eligible_rows", "raw_loss": None,
                        "scale": scale, "scaled_loss": None,
                        "base_gradient_rms_on_support": None,
                        "raw_gradient_rms": None,
                        "scaled_gradient_rms": None,
                    }
                    continue
                matched_base = class_weighted_eligible_mean(
                    base_rows, labels, class_weights, eligible,
                ).loss
                base_gradient = _gradient_tensors(
                    matched_base, named_parameters, retain_graph=True,
                )
                raw_gradient = _gradient_tensors(
                    raw_loss, named_parameters, retain_graph=True,
                )
                scaled_gradient = tuple(scale * value for value in raw_gradient)
                component_rows[name] = {
                    "status": component_state["status"],
                    "raw_loss": float(raw_loss.detach().cpu()),
                    "scale": scale,
                    "scaled_loss": float(scaled_loss.detach().cpu()),
                    "eligible_rows": int(eligible.sum().item()),
                    "base_gradient_rms_on_support": _gradient_rms(base_gradient),
                    "raw_gradient_rms": _gradient_rms(raw_gradient),
                    "scaled_gradient_rms": _gradient_rms(scaled_gradient),
                }
            zero = base["total"] * 0.0
            scientific, coefficients = _diagnostic_scientific_loss(
                execution, effective_pass=float(completed_pass), scaled=scaled,
                zero=zero,
            )
            for name, coefficient in coefficients.items():
                if coefficient > 0 and name in execution.active_components and name not in scaled:
                    raise RuntimeError(
                        f"active diagnostic component {name!r} was not calibrated"
                    )
            orthogonality = projection_orthogonality(
                dict(model.representation_heads.projection_items()),
            )
            representation_total = RHO_REPRESENTATION * (
                scientific
                + jet_set_ramp(float(completed_pass))
                * ORTHOGONALITY_COEFFICIENT
                * orthogonality.float()
            )
            if calibrated_names:
                base_gradient = _gradient_tensors(
                    base["total"], named_parameters, retain_graph=True,
                )
                representation_gradient = _gradient_tensors(
                    representation_total, named_parameters, retain_graph=False,
                )
                gradient_comparison = _gradient_comparison(
                    base_gradient, representation_gradient,
                )
            else:
                gradient_comparison = {
                    "base_gradient_norm": None,
                    "representation_gradient_norm": None,
                    "representation_to_base_ratio": None,
                    "gradient_cosine": None,
                    "status": "not_yet_calibrated",
                }
            labels_cpu = np.asarray(normalized.labels, dtype=np.int64)
            relation_summary = None
            jet_summary = None
            if raw is not None and raw.jet is not None:
                jet_summary = {
                    "direct": float(raw.jet.direct.detach().cpu()),
                    "gram": float(raw.jet.gram.detach().cpu()),
                    "gram_pair_weights": (
                        raw.jet.gram_pair_weights.detach().cpu().tolist()
                    ),
                    "gram_squared_errors": (
                        raw.jet.gram_squared_errors.detach().cpu().tolist()
                    ),
                }
            if raw is not None and raw.relation is not None:
                sketches = raw.relation.student_sketches
                relation_eligibility = raw.relation.family_stratum_eligible.detach().cpu().numpy()
                relation_by_class: dict[str, dict[str, Any]] = {}
                weights_cpu = np.asarray(class_weights.detach().cpu(), dtype=np.float64)
                for class_index in sorted(set(labels_cpu.tolist())):
                    class_rows = labels_cpu == class_index
                    entries: dict[str, Any] = {}
                    for family_index in range(relation_eligibility.shape[1]):
                        for stratum_index in range(relation_eligibility.shape[2]):
                            active = relation_eligibility[
                                class_rows, family_index, stratum_index
                            ].astype(bool)
                            row_weights = np.full(
                                int(active.sum()), weights_cpu[class_index],
                                dtype=np.float64,
                            )
                            denominator = float(row_weights.sum())
                            ess = (
                                0.0 if not len(row_weights) else
                                denominator**2 / float((row_weights**2).sum())
                            )
                            entries[f"{family_index}:{stratum_index}"] = {
                                "eligible_rows": int(active.sum()),
                                "denominator_weight": denominator,
                                "effective_sample_size": ess,
                            }
                    relation_by_class[str(class_index)] = entries
                relation_summary = {
                    "pair_counts": sketches.pair_counts.detach().cpu().tolist(),
                    "effective_sample_sizes": (
                        sketches.effective_sample_sizes.detach().cpu().tolist()
                    ),
                    "jointly_active": (
                        raw.relation.family_stratum_eligible.detach().cpu().tolist()
                    ),
                    "eligibility_by_class_family_stratum": relation_by_class,
                }
            set_summary = None
            if raw is not None and raw.set is not None:
                set_eligibility = raw.set.family_eligible.detach().cpu().numpy()
                set_by_class: dict[str, dict[str, Any]] = {}
                weights_cpu = np.asarray(class_weights.detach().cpu(), dtype=np.float64)
                for class_index in sorted(set(labels_cpu.tolist())):
                    class_rows = labels_cpu == class_index
                    entries: dict[str, Any] = {}
                    for family_index in range(set_eligibility.shape[1]):
                        active = set_eligibility[class_rows, family_index].astype(bool)
                        row_weights = np.full(
                            int(active.sum()), weights_cpu[class_index], dtype=np.float64,
                        )
                        denominator = float(row_weights.sum())
                        ess = (
                            0.0 if not len(row_weights) else
                            denominator**2 / float((row_weights**2).sum())
                        )
                        entries[str(family_index)] = {
                            "eligible_rows": int(active.sum()),
                            "denominator_weight": denominator,
                            "effective_sample_size": ess,
                        }
                    set_by_class[str(class_index)] = entries
                set_summary = {
                    "eligible_rows": raw.set.reduction.eligible_count,
                    "denominator": float(raw.set.reduction.denominator.detach().cpu()),
                    "active_family_count": (
                        raw.set.active_family_count.detach().cpu().tolist()
                    ),
                    "eligibility_by_class_family": set_by_class,
                }
            family_values, family_counts = np.unique(
                np.asarray(normalized.family_codes)[
                    np.asarray(normalized.mask).reshape(
                        len(normalized.identity_digests), -1,
                    ).astype(bool)
                ],
                return_counts=True,
            )
            visible_mask = np.asarray(normalized.mask).reshape(
                len(normalized.identity_digests), -1,
            ).astype(bool)
            family_cpu = np.asarray(normalized.family_codes, dtype=np.int8)
            family_counts_by_class: dict[str, dict[str, int]] = {}
            reason_counts_by_class: dict[str, dict[str, int]] | None = (
                {} if normalized.family_reason_codes is not None else None
            )
            reason_cpu = (
                None if normalized.family_reason_codes is None else np.asarray(
                    normalized.family_reason_codes, dtype=np.int8,
                )
            )
            vectors_cpu = np.asarray(normalized.vectors, dtype=np.float64)
            pt_cpu = np.sqrt(vectors_cpu[:, 0] ** 2 + vectors_cpu[:, 1] ** 2)
            set_population_by_class_family: dict[str, dict[str, Any]] = {}
            for class_index in sorted(set(labels_cpu.tolist())):
                selected_rows = labels_cpu == class_index
                selected_visible = visible_mask[selected_rows]
                selected_family = family_cpu[selected_rows][selected_visible]
                names, counts = np.unique(selected_family, return_counts=True)
                family_counts_by_class[str(class_index)] = {
                    str(int(name)): int(count)
                    for name, count in zip(names, counts, strict=True)
                }
                population: dict[str, Any] = {}
                selected_pt = pt_cpu[selected_rows]
                selected_family_grid = family_cpu[selected_rows]
                for family_code in sorted(set(selected_family.tolist())):
                    family_selected = selected_visible & (
                        selected_family_grid == family_code
                    )
                    population[str(int(family_code))] = {
                        "token_count": int(family_selected.sum()),
                        "scalar_pt_sum": float(selected_pt[family_selected].sum()),
                    }
                set_population_by_class_family[str(class_index)] = population
                if reason_counts_by_class is not None and reason_cpu is not None:
                    selected_reason = reason_cpu[selected_rows][selected_visible]
                    names, counts = np.unique(selected_reason, return_counts=True)
                    reason_counts_by_class[str(class_index)] = {
                        str(int(name)): int(count)
                        for name, count in zip(names, counts, strict=True)
                    }
            result = {
                "contract": "HCWDL_REPRESENTATION_DIAGNOSTIC_RESULT/v1",
                "completed_pass": int(completed_pass),
                "completed_update": int(completed_update),
                "seed_payload_sha256": seed_sha256,
                "ordered_identity_sha256": identity_order_sha256(
                    normalized.identity_digests,
                ),
                "student_forward_calls": 1,
                "forward_dtype": amp_dtype,
                "loss_dtype": "float32",
                "finite": True,
                "components": component_rows,
                "effective_coefficients": coefficients,
                "rho_representation": RHO_REPRESENTATION,
                "gradient_comparison": gradient_comparison,
                "projection_diagnostics": _projection_diagnostic_payload(model),
                "set_support": set_summary,
                "jet_support": jet_summary,
                "relation_support": relation_summary,
                "visible_family_counts": {
                    str(int(name)): int(count)
                    for name, count in zip(family_values, family_counts, strict=True)
                },
                "visible_family_counts_by_class": family_counts_by_class,
                "family_reason_counts_by_class": reason_counts_by_class,
                "set_population_by_class_family": set_population_by_class_family,
                "teacher_target_join_rows": len(normalized.identity_digests),
                "teacher_target_join_bytes": sum(
                    value.numel() * value.element_size()
                    for value in targets.values()
                ),
            }
    finally:
        model.load_state_dict(prior_model_state, strict=True)
        restore_model_runtime_state(model, prior_runtime)
        for name, module in model.named_modules():
            module.training = prior_modes[name]
        for name, parameter in model.named_parameters():
            gradient = prior_gradients[name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
        optimizer.load_state_dict(prior_optimizer)
        restore_rng_state(prior_rng)
        if external_restore is not None:
            external_restore(copy.deepcopy(prior_external))
    if result is None:
        raise RuntimeError("HCWDL-RKD diagnostic did not produce a result")
    if not _tree_equal(_cpu_tree(model.state_dict()), prior_model_state):
        raise RuntimeError("diagnostic mutated model state")
    if not _tree_equal(_cpu_tree(optimizer.state_dict()), prior_optimizer):
        raise RuntimeError("diagnostic mutated optimizer state")
    if not _tree_equal(_cpu_tree(capture_rng_state()), prior_rng):
        raise RuntimeError("diagnostic mutated RNG state")
    if external_snapshot is not None and not _tree_equal(
        _cpu_tree(external_snapshot()), prior_external,
    ):
        raise RuntimeError("diagnostic mutated external cursor/logging state")
    return result


def _run_calibration(
    *,
    phase: str,
    component_names: Sequence[str],
    execution: NodeExecution,
    model,
    optimizer,
    batches: Iterable[Mapping[str, Any]],
    device,
    amp_dtype: str,
    class_weights,
    target_bank,
    predecessor_bank: InMemoryPredecessorLogits | None,
    shuffled_representation_joiner,
    token_resources,
    relation_resources,
    expected_batches: int,
    minimum_valid_batches: int,
    external_snapshot,
    external_restore,
) -> GradientCalibrationResult:
    import torch

    configuration = node_base_loss_configuration(execution)

    def student_forward(raw):
        batch = normalize_hlt_batch(raw)
        features, vectors, mask, visible, family, labels = _batch_tensors(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_dtype == "bfloat16",
        ):
            surfaces = model.forward_hcwdl_surfaces(
                features, vectors, mask, visible, family,
            )
        targets = _target_tensors(
            target_bank, batch.identity_digests, device=device, execution=execution,
            shuffled_representation_joiner=shuffled_representation_joiner,
        )
        predecessor = (
            None
            if predecessor_bank is None
            else torch.as_tensor(
                predecessor_bank.join(batch.identity_digests), device=device,
                dtype=torch.float32,
            ).detach()
        )
        return batch, labels, surfaces, targets, predecessor

    def losses_from_forward(_raw, forward):
        _batch, labels, surfaces, targets, predecessor = forward
        if predecessor is not None:
            raise ValueError(
                "dense descent calibration forbids a second predecessor-logit teacher"
            )
        hlt_logits = targets["logits"].float()
        privileged_logits = None
        base_rows = hcwdl_base_loss_rows(
            surfaces.logits.float(), labels, class_weights=class_weights,
            configuration=configuration, hlt_teacher_logits=hlt_logits,
            privileged_teacher_logits=privileged_logits,
        )
        raw_components = _raw_representation_components(
            execution=execution, model=model, surfaces=surfaces, targets=targets,
            labels=labels, class_weights=class_weights,
            token_resources=token_resources, relation_resources=relation_resources,
            components=component_names,
        )
        return CalibrationForwardResult(
            base_rows=base_rows["total_rows"], labels=labels,
            class_weights=class_weights, components=raw_components.rows,
        )

    return calibrate_representation_components(
        model=model,
        batches=batches,
        student_forward=student_forward,
        losses_from_forward=losses_from_forward,
        component_names=component_names,
        optimizer=optimizer,
        expected_batches=expected_batches,
        minimum_valid_batches=minimum_valid_batches,
        external_snapshot=external_snapshot,
        external_restore=external_restore,
    )


def _apply_calibration_result(
    state: dict[str, Any],
    *,
    phase: str,
    result: GradientCalibrationResult,
    artifact_hash: str,
    completed_pass: int,
    completed_update: int,
) -> None:
    state["artifact_hashes"][phase] = require_sha256(
        artifact_hash, name=f"calibration artifact {phase}",
    )
    for name, component in result.components.items():
        row = asdict(component)
        row.update({
            "phase": phase,
            "completed_pass": completed_pass,
            "completed_update": completed_update,
            "artifact_sha256": artifact_hash,
        })
        state["components"][name] = {
            "status": component.status,
            "scale": component.scale,
            "history": [*state["components"][name]["history"], row],
        }


def _publish_calibration_manifest(
    output: Path,
    *,
    execution: NodeExecution,
    lineage: Mapping[str, str],
    calibration: dict[str, Any],
    completed_pass: int,
    completed_update: int,
    calibration_directory: Path | None = None,
) -> dict[str, Any] | None:
    """Publish the one- or two-phase immutable calibration manifest exactly once."""

    required_phases = (
        ("jet_set", "relation")
        if "relation" in execution.active_components
        else ("jet_set",)
    )
    if not all(name in calibration["artifact_hashes"] for name in required_phases):
        return None
    existing_hash = calibration["artifact_hashes"].get("manifest")
    phase_hashes = {
        name: calibration["artifact_hashes"][name] for name in required_phases
    }
    parents = {
        "execution": lineage["execution"],
        "representation_recipe": lineage["representation_recipe"],
        "target_generation": lineage["target_generation"],
        "target_logical": lineage["target_logical"],
        **{f"phase_{name}": digest for name, digest in phase_hashes.items()},
    }
    if calibration.get("selection_sha256") is not None:
        parents["calibration_selection"] = require_sha256(
            calibration["selection_sha256"], name="calibration selection",
        )
    path = (
        output / "calibration"
        if calibration_directory is None
        else Path(calibration_directory)
    ) / "manifest.json"
    if existing_hash is not None:
        published = load_json(path)
        digest = validate_versioned_artifact(
            published,
            expected_contract=GRADIENT_CALIBRATION_MANIFEST_CONTRACT,
            expected_parents=parents,
            required_payload_keys=(
                "execution_id", "strategy", "required_phase_order",
                "completed_pass", "completed_update", "components",
                "ordered_selection_sha256",
                "ordered_calibration_identity_sha256", "calibration_rows",
            ),
        )
        if digest != existing_hash:
            raise ValueError("calibration manifest hash differs on resume")
        return published
    artifact = build_versioned_artifact(
        GRADIENT_CALIBRATION_MANIFEST_CONTRACT,
        parents=parents,
        payload={
            "execution_id": execution.execution_id,
            "strategy": execution.short_strategy,
            "required_phase_order": list(required_phases),
            "completed_pass": int(completed_pass),
            "completed_update": int(completed_update),
            "ordered_selection_sha256": calibration.get(
                "ordered_selection_sha256"
            ),
            "ordered_calibration_identity_sha256": calibration.get(
                "ordered_calibration_identity_sha256"
            ),
            "calibration_rows": calibration.get("calibration_rows"),
            "components": {
                name: {
                    "status": calibration["components"][name]["status"],
                    "scale": calibration["components"][name]["scale"],
                    "scale_hex": float(
                        calibration["components"][name]["scale"]
                    ).hex(),
                }
                for name in execution.active_components
            },
        },
    )
    write_immutable_json(path, artifact)
    published = load_json(path)
    validate_versioned_artifact(
        published,
        expected_contract=GRADIENT_CALIBRATION_MANIFEST_CONTRACT,
        expected_parents=parents,
        required_payload_keys=(
            "execution_id", "strategy", "required_phase_order",
            "completed_pass", "completed_update", "components",
            "ordered_selection_sha256",
            "ordered_calibration_identity_sha256", "calibration_rows",
        ),
    )
    if published != artifact:
        raise ValueError("calibration manifest artifact differs")
    calibration["artifact_hashes"]["manifest"] = artifact["content_hash"]
    return artifact


def _default_extractor(
    model,
    *,
    checkpoint_path: Path,
    selected_training_checkpoint_sha256: str,
    architecture_attestation_sha256: str,
    parity_inputs,
) -> Mapping[str, Any]:
    from hlt_classification.models.hcwdl_representation import (
        publish_hcwdl_deployable_extraction,
    )

    result = publish_hcwdl_deployable_extraction(
        model,
        checkpoint_path=checkpoint_path,
        selected_training_checkpoint_sha256=selected_training_checkpoint_sha256,
        architecture_attestation_sha256=architecture_attestation_sha256,
        parity_inputs=parity_inputs,
    )
    return {
        "checkpoint_path": str(result.checkpoint_path),
        "checkpoint_sha256": result.checkpoint_sha256,
        "report_path": str(result.report_path),
        "report_sha256": result.report["content_hash"],
        "strict_hlt_only": True,
    }


def _publish_terminal_checkpoint_envelopes(
    *,
    output: Path,
    execution: NodeExecution,
    lineage: Mapping[str, str],
    architecture_attestation_sha256: str,
    selected_path: Path,
    selected_sha256: str,
    final_state: Mapping[str, Any],
    extraction: Mapping[str, Any],
    registered_output_row: Mapping[str, Any] | None,
    publication_owner: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Commit selected/final binary states and the exact extraction record.

    Training may write boundary candidates and an extractor-owned temporary
    file, but neither becomes a reusable artifact.  This function is the sole
    publication boundary for terminal binary state.
    """

    selected_bytes = selected_path.read_bytes()
    if sha256_bytes(selected_bytes) != require_sha256(
        selected_sha256, name="selected training checkpoint SHA-256",
    ):
        raise ValueError("selected checkpoint changed before terminal commit")
    deployable_path = Path(str(extraction.get("checkpoint_path", "")))
    if not deployable_path.is_file():
        raise FileNotFoundError("deployable extractor did not publish its temporary state")
    deployable_bytes = deployable_path.read_bytes()
    deployable_sha256 = sha256_bytes(deployable_bytes)
    if deployable_sha256 != require_sha256(
        extraction.get("checkpoint_sha256"), name="deployable checkpoint SHA-256",
    ):
        raise ValueError("deployable extraction bytes differ")

    output_row = dict(registered_output_row or {
        "task_kind": "representation_training",
        "node_id": execution.execution_id,
        "registered_execution_id": lineage["execution"],
    })
    owner = dict(publication_owner or {
        "registered_execution_id": lineage["execution"],
        "representation_recipe_sha256": lineage["representation_recipe"],
    })
    common_parents = {
        "execution": lineage["execution"],
        "representation_recipe": lineage["representation_recipe"],
        "target_generation": lineage["target_generation"],
        "target_logical": lineage["target_logical"],
        "architecture_attestation": require_sha256(
            architecture_attestation_sha256,
            name="architecture attestation SHA-256",
        ),
    }
    selected_envelope = publish_binary_envelope(
        output / "checkpoints" / "selected",
        artifact_contract=SELECTED_TRAINING_CHECKPOINT_CONTRACT,
        producer_task_id=str(output_row.get("task_key", execution.execution_id)),
        schema={
            "kind": "selected_training_and_deployable_state",
            "schema_version": 1,
        },
        immutable_parent_hashes={
            **common_parents,
            "selected_training_state": selected_sha256,
            "deployable_state": deployable_sha256,
        },
        registered_output_row=output_row,
        campaign_or_recovery_owner=owner,
        payloads={
            "training_state.pt": selected_bytes,
            "deployable_state.pt": deployable_bytes,
        },
        member_metadata={
            "training_state.pt": {"logical_sha256": selected_sha256},
            "deployable_state.pt": {"logical_sha256": deployable_sha256},
        },
        sidecar_payload={
            "node_id": execution.execution_id,
            "registered_execution_id": lineage["execution"],
            "student_domain": execution.student_domain,
            "deployment_authorized": execution.deployable,
            "strict_hlt_only_deployable": execution.deployable,
        },
    )
    final_payload = {
        "contract": FINAL_TRAINING_CHECKPOINT_CONTRACT,
        "schema_version": 1,
        "execution_id": execution.execution_id,
        "registered_execution_id": lineage["execution"],
        "state": _cpu_tree(final_state),
    }
    final_bytes = _torch_bytes(final_payload)
    final_sha256 = sha256_bytes(final_bytes)
    final_envelope = publish_binary_envelope(
        output / "checkpoints" / "final",
        artifact_contract=FINAL_TRAINING_CHECKPOINT_CONTRACT,
        producer_task_id=str(output_row.get("task_key", execution.execution_id)),
        schema={"kind": "terminal_training_state", "schema_version": 1},
        immutable_parent_hashes={
            **common_parents,
            "final_training_state": final_sha256,
        },
        registered_output_row=output_row,
        campaign_or_recovery_owner=owner,
        payloads={"training_state.pt": final_bytes},
        member_metadata={
            "training_state.pt": {"logical_sha256": final_sha256},
        },
        sidecar_payload={
            "node_id": execution.execution_id,
            "registered_execution_id": lineage["execution"],
            "terminal_state_is_not_selected_state": True,
        },
    )
    extraction_artifact = build_versioned_artifact(
        DEPLOYABLE_EXTRACTION_CONTRACT,
        parents={
            **common_parents,
            "selected_envelope": selected_envelope.commit["content_hash"],
            "selected_training_state": selected_sha256,
            "deployable_state": deployable_sha256,
            "extractor_report": require_sha256(
                extraction.get("report_sha256"), name="extractor report SHA-256",
            ),
        },
        payload={
            "node_id": execution.execution_id,
            "registered_execution_id": lineage["execution"],
            "selected_envelope_id": selected_envelope.envelope_id,
            "selected_training_state_path": str(
                selected_envelope.directory / "training_state.pt"
            ),
            "deployable_state_path": str(
                selected_envelope.directory / "deployable_state.pt"
            ),
            "deployable_state_sha256": deployable_sha256,
            "student_domain": execution.student_domain,
            "deployment_authorized": execution.deployable,
            "strict_hlt_only": execution.deployable,
            "training_only_heads_excluded": True,
        },
    )
    extraction_path = output / "deployable_extraction.json"
    write_immutable_json(extraction_path, extraction_artifact)
    # Extractor products were staged only to prove strict public-model parity;
    # the committed envelope is the sole reusable binary publication.
    try:
        deployable_path.relative_to(output)
        deployable_is_internal = True
    except ValueError:
        deployable_is_internal = False
    if deployable_is_internal:
        deployable_path.unlink(missing_ok=True)
        deployable_path.with_suffix(deployable_path.suffix + ".json").unlink(
            missing_ok=True
        )
    published_extraction = {
        **dict(extraction),
        "checkpoint_path": str(selected_envelope.directory / "deployable_state.pt"),
        "checkpoint_sha256": deployable_sha256,
        "report_path": str(extraction_path),
        "report_sha256": extraction_artifact["content_hash"],
        "selected_envelope_id": selected_envelope.envelope_id,
        "selected_envelope_sha256": selected_envelope.commit["content_hash"],
        "final_envelope_id": final_envelope.envelope_id,
        "final_envelope_sha256": final_envelope.commit["content_hash"],
        "final_training_checkpoint_sha256": final_sha256,
        "student_domain": execution.student_domain,
        "deployment_authorized": execution.deployable,
        "strict_hlt_only": execution.deployable,
    }
    envelope_report = {
        "selected": {
            "envelope_id": selected_envelope.envelope_id,
            "commit_sha256": selected_envelope.commit["content_hash"],
            "training_state_path": str(
                selected_envelope.directory / "training_state.pt"
            ),
        },
        "final": {
            "envelope_id": final_envelope.envelope_id,
            "commit_sha256": final_envelope.commit["content_hash"],
            "training_state_path": str(final_envelope.directory / "training_state.pt"),
            "training_state_sha256": final_sha256,
        },
    }
    return published_extraction, envelope_report


def exercise_full_representation_loss(
    model,
    *,
    execution_id: str,
    batch: Mapping[str, Any],
    target_bank,
    predecessor_bank: InMemoryPredecessorLogits | None,
    class_weights: np.ndarray,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    device: str,
    shuffled_representation_joiner=None,
    require_canonical_early_backbone: bool = False,
) -> dict[str, Any]:
    """Non-scientific deep-copy probe with both ramps forced to one.

    No optimizer/scheduler step occurs and the caller's model/RNG state is
    unchanged.  Production support calibration remains mandatory separately.
    """

    import torch

    execution = resolve_node_execution(execution_id)
    rng = capture_rng_state()
    runtime = capture_model_runtime_state(model)
    probe = copy.deepcopy(model).to(device)
    try:
        normalized = normalize_hlt_batch(batch)
        features, vectors, mask, visible, family, labels = _batch_tensors(
            normalized, torch.device(device),
        )
        probe.train()
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            surfaces = probe.forward_hcwdl_surfaces(
                features, vectors, mask, visible, family,
            )
        targets = _target_tensors(
            target_bank, normalized.identity_digests, device=torch.device(device),
            execution=execution,
            shuffled_representation_joiner=shuffled_representation_joiner,
        )
        predecessor = (
            None if predecessor_bank is None else torch.as_tensor(
                predecessor_bank.join(normalized.identity_digests),
                device=device, dtype=torch.float32,
            ).detach()
        )
        loss = compute_node_loss(
            execution=execution, model=probe, surfaces=surfaces, labels=labels,
            class_weights=torch.as_tensor(class_weights, device=device),
            privileged_targets=targets, predecessor_logits=predecessor,
            calibration_scales={name: 1.0 for name in execution.active_components},
            effective_pass=8.0, token_resources=token_resources,
            relation_resources=relation_resources, force_components=True,
        )
        if loss.raw_components is None:
            raise RuntimeError("full-loss probe did not materialize active components")
        try:
            early_parameters = early_backbone_parameters(probe)
        except ValueError:
            if require_canonical_early_backbone:
                raise
            # Lightweight unit-test deployables intentionally omit the full
            # ParT prefix topology.  They may exercise the generic loss API,
            # but authority-bound production below requires the canonical
            # early-backbone selector and never enters this fallback.
            early_parameters = tuple(
                (name, parameter)
                for name, parameter in probe.named_parameters()
                if name.startswith("deployable_model.")
                and parameter.requires_grad
            )
            if not early_parameters:
                raise ValueError("full-loss probe backbone support is empty")
        component_gradient_norms: dict[str, float] = {}
        for component in execution.active_components:
            component_loss = loss.raw_components.losses.get(component)
            if component_loss is None:
                raise RuntimeError(
                    f"full-loss probe lacks active component {component!r}"
                )
            if not component_loss.requires_grad:
                if require_canonical_early_backbone:
                    raise RuntimeError(
                        f"full-loss probe component {component!r} has no live "
                        "canonical gradient support"
                    )
                component_gradient_norms[component] = 0.0
                continue
            gradients = torch.autograd.grad(
                component_loss,
                tuple(parameter for _, parameter in early_parameters),
                retain_graph=True,
                create_graph=False,
                allow_unused=not require_canonical_early_backbone,
            )
            connected_gradients = tuple(
                gradient for gradient in gradients if gradient is not None
            )
            if (
                len(gradients) != len(early_parameters)
                or not connected_gradients
                or (
                    require_canonical_early_backbone
                    and len(connected_gradients) != len(early_parameters)
                )
                or any(
                    not torch.isfinite(gradient).all()
                    for gradient in connected_gradients
                )
            ):
                raise FloatingPointError(
                    f"full-loss probe component {component!r} has invalid "
                    "early-backbone gradients"
                )
            squared = sum(
                gradient.detach().double().square().sum()
                for gradient in connected_gradients
            )
            norm = float(torch.sqrt(squared).cpu())
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(
                    f"full-loss probe component {component!r} is disconnected "
                    "from the early backbone"
                )
            component_gradient_norms[component] = norm
        probe.zero_grad(set_to_none=True)
        loss.total.backward()
        head_norms = {}
        for name, projection in probe.representation_heads.projection_items():
            gradient = projection.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"full-loss probe head {name!r} has invalid gradient")
            head_norms[name] = float(gradient.float().norm().cpu())
        if any(value <= 0 for value in head_norms.values()):
            raise RuntimeError("full-loss probe has a disconnected active projection")
        early_squared = torch.zeros((), dtype=torch.float64, device=device)
        for name, parameter in early_parameters:
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"full-loss probe early-backbone parameter {name!r} has "
                    "an invalid total-loss gradient"
                )
            early_squared = early_squared + gradient.detach().double().square().sum()
        early_norm = float(torch.sqrt(early_squared).cpu())
        total_loss = float(loss.total.detach().cpu())
        representation_loss = float(loss.representation_total.detach().cpu())
        if (
            not math.isfinite(early_norm) or early_norm <= 0
            or not math.isfinite(total_loss) or total_loss <= 0
            or not math.isfinite(representation_loss) or representation_loss <= 0
        ):
            raise FloatingPointError("full-loss probe scalar evidence is invalid")
        return {
            "execution_id": execution_id,
            "scientific_authorization": False,
            "effective_pass_forced": 8.0,
            "active_components": list(execution.active_components),
            "total_loss": total_loss,
            "representation_loss": representation_loss,
            "head_gradient_norms": head_norms,
            "active_component_early_backbone_gradient_norms": (
                component_gradient_norms
            ),
            "early_backbone_gradient_norm": early_norm,
            "finite": True,
            "optimizer_step_performed": False,
        }
    finally:
        del probe
        restore_model_runtime_state(model, runtime)
        restore_rng_state(rng)


def _build_acceptance_real_batch_full_loss_record(
    *, binding: Mapping[str, Any], probe: Mapping[str, Any],
    execution: NodeExecution, registered_execution_id: str,
    diagnostic_batch_sha256: str, target_generation_sha256: str,
    target_logical_sha256: str, target_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind a nonmutating full-loss exercise to one authority-private batch."""

    required_binding = {
        "authority_sha256", "action_id", "action_spec_sha256", "source_commit",
        "representation_recipe_sha256", "train_rows", "validation_rows",
        "replicate_seed", "maximum_optimizer_updates",
        "target_generation_sha256", "target_logical_sha256",
        "target_manifest_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding:
        raise ValueError("acceptance full-loss binding fields differ")
    for name in (
        "authority_sha256", "action_spec_sha256",
        "representation_recipe_sha256", "target_generation_sha256",
        "target_logical_sha256", "target_manifest_sha256",
    ):
        require_sha256(binding[name], name=f"acceptance full-loss {name}")
    source_commit = str(binding["source_commit"])
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("acceptance full-loss source commit differs")
    action_id = str(binding["action_id"])
    if action_id not in {
        "rset_m1c_two_update", "rset_m1w_two_update",
        "rrel_m1c_two_update", "rrel_m1w_two_update",
    }:
        raise ValueError("acceptance full-loss action identity differs")
    exact_counts = {
        "train_rows": 512,
        "validation_rows": 256,
        "replicate_seed": 1337,
        "maximum_optimizer_updates": 2,
    }
    if any(binding.get(name) != expected for name, expected in exact_counts.items()):
        raise PermissionError("acceptance full-loss bounded counts differ")
    expected_targets = {
        "target_generation_sha256": target_generation_sha256,
        "target_logical_sha256": target_logical_sha256,
        "target_manifest_sha256": target_manifest_sha256,
    }
    if any(binding.get(name) != expected for name, expected in expected_targets.items()):
        raise PermissionError("acceptance full-loss target binding differs")
    if probe.get("execution_id") != execution.execution_id:
        raise ValueError("acceptance full-loss execution identity differs")
    active = list(execution.active_components)
    if probe.get("active_components") != active:
        raise ValueError("acceptance full-loss active components differ")
    component_norms = probe.get(
        "active_component_early_backbone_gradient_norms"
    )
    head_norms = probe.get("head_gradient_norms")
    if (
        not isinstance(component_norms, Mapping)
        or set(component_norms) != set(active)
        or not isinstance(head_norms, Mapping)
        or not head_norms
    ):
        raise ValueError("acceptance full-loss gradient registry differs")
    scalar_values = [
        probe.get("total_loss"), probe.get("representation_loss"),
        probe.get("early_backbone_gradient_norm"),
        *component_norms.values(), *head_norms.values(),
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0
        for value in scalar_values
    ):
        raise FloatingPointError("acceptance full-loss scalar evidence is invalid")
    if (
        probe.get("finite") is not True
        or probe.get("optimizer_step_performed") is not False
        or probe.get("scientific_authorization") is not False
        or float(probe.get("effective_pass_forced", -1.0)) != 8.0
    ):
        raise PermissionError("acceptance full-loss nonmutating semantics differ")
    return with_content_hash({
        "contract": ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
        "schema_version": 1,
        "authority_sha256": binding["authority_sha256"],
        "action_id": action_id,
        "action_spec_sha256": binding["action_spec_sha256"],
        "source_commit": source_commit,
        "representation_recipe_sha256": binding[
            "representation_recipe_sha256"
        ],
        "execution_id": execution.execution_id,
        "registered_execution_id": require_sha256(
            registered_execution_id,
            name="acceptance full-loss registered execution",
        ),
        "diagnostic_batch_sha256": require_sha256(
            diagnostic_batch_sha256,
            name="acceptance full-loss diagnostic batch",
        ),
        "target_generation_sha256": require_sha256(
            target_generation_sha256,
            name="acceptance full-loss target generation",
        ),
        "target_logical_sha256": require_sha256(
            target_logical_sha256,
            name="acceptance full-loss target logical identity",
        ),
        "target_manifest_sha256": require_sha256(
            target_manifest_sha256,
            name="acceptance full-loss target manifest",
        ),
        "diagnostic_rows": 256,
        **exact_counts,
        "active_components": active,
        "total_loss": float(probe["total_loss"]),
        "representation_loss": float(probe["representation_loss"]),
        "head_gradient_norms": {
            str(name): float(value) for name, value in sorted(head_norms.items())
        },
        "active_component_early_backbone_gradient_norms": {
            name: float(component_norms[name]) for name in active
        },
        "early_backbone_gradient_norm": float(
            probe["early_backbone_gradient_norm"]
        ),
        "effective_pass_forced": 8.0,
        "real_bounded_training_batch": True,
        "model_and_rng_restored": True,
        "finite": True,
        "optimizer_step_performed": False,
        "scientific_authorization": False,
        "final_role_accessed": False,
    })


def validate_acceptance_real_batch_full_loss_record(
    value: Mapping[str, Any], *, expected_authority_sha256: str | None = None,
    expected_action_id: str | None = None,
    expected_execution_id: str | None = None,
    expected_recipe_sha256: str | None = None,
    expected_diagnostic_batch_sha256: str | None = None,
) -> str:
    """Validate one separate authority-private real-batch loss artifact."""

    digest = validate_content_hash(
        value, expected_contract=ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "contract", "schema_version", "authority_sha256", "action_id",
        "action_spec_sha256", "source_commit", "representation_recipe_sha256",
        "execution_id", "registered_execution_id", "diagnostic_batch_sha256",
        "target_generation_sha256", "target_logical_sha256",
        "target_manifest_sha256",
        "diagnostic_rows", "train_rows", "validation_rows", "replicate_seed",
        "maximum_optimizer_updates", "active_components", "total_loss",
        "representation_loss", "head_gradient_norms",
        "active_component_early_backbone_gradient_norms",
        "early_backbone_gradient_norm", "effective_pass_forced",
        "real_bounded_training_batch", "model_and_rng_restored", "finite",
        "optimizer_step_performed", "scientific_authorization",
        "final_role_accessed", "content_hash",
    }
    if set(value) != required:
        raise ValueError("acceptance real-batch full-loss record fields differ")
    execution_id = str(value["execution_id"])
    execution = resolve_node_execution(execution_id)
    if value["active_components"] != list(execution.active_components):
        raise ValueError("acceptance real-batch active components differ")
    component_norms = value["active_component_early_backbone_gradient_norms"]
    head_norms = value["head_gradient_norms"]
    if (
        not isinstance(component_norms, Mapping)
        or set(component_norms) != set(execution.active_components)
        or not isinstance(head_norms, Mapping) or not head_norms
    ):
        raise ValueError("acceptance real-batch gradient registry differs")
    scalar_values = [
        value["total_loss"], value["representation_loss"],
        value["early_backbone_gradient_norm"],
        *component_norms.values(), *head_norms.values(),
    ]
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        or not math.isfinite(float(item)) or float(item) <= 0
        for item in scalar_values
    ):
        raise FloatingPointError("acceptance real-batch full-loss values are invalid")
    expected = {
        "diagnostic_rows": 256, "train_rows": 512,
        "validation_rows": 256, "replicate_seed": 1337,
        "maximum_optimizer_updates": 2, "effective_pass_forced": 8.0,
        "real_bounded_training_batch": True, "model_and_rng_restored": True,
        "finite": True, "optimizer_step_performed": False,
        "scientific_authorization": False, "final_role_accessed": False,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError("acceptance real-batch full-loss semantics differ")
    for name in (
        "authority_sha256", "action_spec_sha256",
        "representation_recipe_sha256", "registered_execution_id",
        "diagnostic_batch_sha256", "target_generation_sha256",
        "target_logical_sha256", "target_manifest_sha256",
    ):
        require_sha256(value[name], name=f"acceptance real-batch {name}")
    source_commit = str(value["source_commit"])
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("acceptance real-batch source commit differs")
    comparisons = (
        ("authority_sha256", expected_authority_sha256),
        ("action_id", expected_action_id),
        ("execution_id", expected_execution_id),
        ("representation_recipe_sha256", expected_recipe_sha256),
        ("diagnostic_batch_sha256", expected_diagnostic_batch_sha256),
    )
    for name, expected_value in comparisons:
        if expected_value is not None and value[name] != expected_value:
            raise PermissionError(f"acceptance real-batch {name} differs")
    return digest


def _commit_then_wait_for_preemption(
    *, preemption_requested: Callable[[], bool],
    commit_resume: Callable[[], None], preemption_wait: Callable[[], None],
) -> None:
    """Accept only a signal delivered after the exact committed boundary."""

    if preemption_requested():
        raise RuntimeError(
            "preemption arrived before the committed update-one boundary"
        )
    commit_resume()
    preemption_wait()
    if not preemption_requested():
        raise RuntimeError("preemption wait returned without a delivered signal")


def train_hcwdl_representation_node(
    *,
    execution_id: str,
    parent_recipe: Mapping[str, Any],
    representation_recipe: Mapping[str, Any],
    recipe_compatibility: Mapping[str, Any] | None = None,
    campaign_sha256: str,
    train_rows: int,
    replicate_seed: int,
    train_batches: Callable[[int, int], Iterable[Mapping[str, Any]]],
    validation_batches: Callable[[], Iterable[Mapping[str, Any]]],
    target_bank,
    target_cache_diagnostics: Mapping[str, Any] | None = None,
    token_resources: SpectralKernelResources,
    relation_resources: SpectralKernelResources,
    output_dir: str | Path,
    calibration_dir: str | Path | None = None,
    resume_lineage: Mapping[str, Any],
    producer_runtime_signature: Mapping[str, Any],
    architecture_attestation_sha256: str,
    device: str = "cuda",
    mode: str = "scientific",
    synthetic_passes: int = 1,
    warm_checkpoint: str | Path | None = None,
    warm_checkpoint_sha256: str | None = None,
    deployable_factory: Callable[[], Any] = _default_deployable_factory,
    wrapper_factory: Callable[..., Any] = _default_wrapper_factory,
    warm_loader: Callable[[str | Path, str], Any] | None = None,
    predecessor_model_loader: Callable[[str], Any] | None = None,
    predecessor_batches: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
    shuffled_representation_joiner: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    shuffle_map_sha256: str | None = None,
    calibration_batches: Callable[[str], Iterable[Mapping[str, Any]]] | None = None,
    calibration_selection: Mapping[str, Any] | None = None,
    calibration_expected_batches: int = CALIBRATION_BATCHES,
    calibration_minimum_valid_batches: int = 12,
    calibration_external_snapshot: Callable[[], object] | None = None,
    calibration_external_restore: Callable[[object], None] | None = None,
    diagnostic_batches: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
    diagnostic_external_snapshot: Callable[[], object] | None = None,
    diagnostic_external_restore: Callable[[object], None] | None = None,
    diagnostic_parameter_selector: Callable[
        [Any], Sequence[tuple[str, Any]]
    ] = early_backbone_parameters,
    acceptance_full_loss_binding: Mapping[str, Any] | None = None,
    sampler_external_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    sampler_external_restore: Callable[[Mapping[str, Any]], None] | None = None,
    preemption_requested: Callable[[], bool] | None = None,
    preemption_wait_after_update: int | None = None,
    preemption_wait: Callable[[], None] | None = None,
    stop_after_update: int | None = None,
    extractor: Callable[..., Mapping[str, Any]] = _default_extractor,
    registered_output_row: Mapping[str, Any] | None = None,
    publication_owner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one exact primary/control node and publish its deployable extraction."""

    import torch

    execution = resolve_node_execution(execution_id)
    campaign_sha256 = require_sha256(
        campaign_sha256, name="HCWDL-RKD campaign SHA-256",
    )
    representation_recipe_sha256 = validate_representation_recipe(
        representation_recipe,
    )
    parent_recipe_sha256 = validate_parent_recipe(
        parent_recipe,
        require_authorized=mode in {"scientific", "smoke"},
        expected_profile=(
            "primary_ladder" if mode in {"scientific", "smoke"} else None
        ),
    )
    if parent_recipe.get("contract") != PARENT_RECIPE_CONTRACT:
        raise ValueError(
            "HCWDL-RKD requires the unweighted HCWDL_RECIPE/v4 parent"
        )
    if not np.array_equal(
        np.asarray(parent_recipe.get("class_weights"), dtype=np.float32),
        np.ones(15, dtype=np.float32),
    ):
        raise ValueError("HCWDL-RKD parent recipe must bind fifteen exact ones")
    if representation_recipe["parents"]["parent_recipe"] != parent_recipe_sha256:
        if recipe_compatibility is None:
            raise ValueError("representation overlay binds a different parent recipe")
        from .hcwdl_homotopy_representation_prerequisites import (
            validate_recipe_compatibility,
        )
        validate_recipe_compatibility(
            recipe_compatibility, execution_recipe=parent_recipe,
            representation_recipe=representation_recipe,
        )
    elif recipe_compatibility is not None:
        from .hcwdl_homotopy_representation_prerequisites import (
            validate_recipe_compatibility,
        )
        validate_recipe_compatibility(
            recipe_compatibility, execution_recipe=parent_recipe,
            representation_recipe=representation_recipe,
        )
    expected_graph_sha256 = _registered_graph_sha256(execution_id)
    lineage = _validate_runtime_lineage(
        resume_lineage, producer_runtime_signature,
        expected_graph_sha256=expected_graph_sha256,
    )
    if lineage["representation_recipe"] != representation_recipe_sha256:
        raise ValueError("resume lineage binds a different representation recipe")
    require_sha256(
        architecture_attestation_sha256, name="architecture attestation SHA-256",
    )
    target_manifest_payload = _validate_target_bank_binding(
        target_bank, execution=execution, lineage=lineage, train_rows=train_rows,
        replicate_seed=replicate_seed,
    )
    cache_diagnostics = dict(target_cache_diagnostics or {
        "construction_seconds": 0.0,
        "load_seconds": 0.0,
        "generation_sha256": lineage["target_generation"],
        "logical_sha256": lineage["target_logical"],
        "manifest_sha256": target_bank.manifest["content_hash"],
        "source": "already_materialized_test_bank",
    })
    for name in ("construction_seconds", "load_seconds"):
        value = float(cache_diagnostics.get(name, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"HCWDL-RKD target-cache {name} differs")
        cache_diagnostics[name] = value
    expected_cache_hashes = {
        "generation_sha256": lineage["target_generation"],
        "logical_sha256": lineage["target_logical"],
        "manifest_sha256": target_bank.manifest["content_hash"],
    }
    if any(cache_diagnostics.get(name) != value for name, value in expected_cache_hashes.items()):
        raise ValueError("HCWDL-RKD target-cache diagnostic lineage differs")
    if execution.shuffled_representation_targets:
        require_sha256(shuffle_map_sha256, name="representation shuffle-map SHA-256")
        if shuffled_representation_joiner is None:
            raise ValueError("shuffled control lacks its authenticated join adapter")
    elif shuffled_representation_joiner is not None or shuffle_map_sha256 is not None:
        raise ValueError("unshuffled execution cannot receive a shuffle adapter")
    if mode in {"scientific", "smoke"} and diagnostic_batches is None:
        raise ValueError(
            "scientific/smoke HCWDL-RKD execution requires a fixed diagnostic batch"
        )
    if (diagnostic_external_snapshot is None) != (
        diagnostic_external_restore is None
    ):
        raise ValueError("diagnostic external snapshot/restore must be supplied together")
    if mode != "synthetic_test" and diagnostic_parameter_selector is not early_backbone_parameters:
        raise ValueError("scientific diagnostic parameter support cannot be overridden")
    if acceptance_full_loss_binding is not None and mode != "smoke":
        raise PermissionError("acceptance full-loss evidence requires smoke mode")
    if (preemption_wait_after_update is None) != (preemption_wait is None):
        raise ValueError("preemption wait boundary/callback must be supplied together")
    if preemption_wait_after_update is not None and (
        mode != "smoke"
        or preemption_wait_after_update != 1
        or preemption_requested is None
    ):
        raise PermissionError(
            "the exact signal wait barrier is restricted to smoke update one"
        )
    if calibration_selection is None:
        if mode in {"scientific", "smoke"}:
            raise ValueError(
                "scientific/smoke HCWDL-RKD execution lacks calibration selection"
            )
    else:
        validate_calibration_selection_artifact(
            calibration_selection,
            expected_campaign_sha256=campaign_sha256,
            expected_parent_logit_counterpart_node_id=execution.parent_counterpart,
        )
        if int(calibration_selection["actual_rows"]) != min(4096, int(train_rows)):
            raise ValueError("HCWDL-RKD calibration-selection row count differs")

    config = representation_training_configuration(
        execution_id, parent_recipe, train_rows=train_rows,
        replicate_seed=replicate_seed, mode=mode,
        synthetic_passes=synthetic_passes,
    )
    target_payload = target_bank.manifest.get("payload", {})
    if target_payload.get("logical_target_sha256") != lineage["target_logical"]:
        raise ValueError("materialized target bank logical hash differs from resume lineage")
    rng_streams = paired_rng_streams(execution_id, replicate_seed)
    validate_paired_rng_streams(
        rng_streams, execution_id=execution_id, replicate_seed=replicate_seed,
    )
    _seed_paired_training_rng(rng_streams)
    model = initialize_representation_student(
        execution_id, replicate_seed=replicate_seed,
        warm_checkpoint=warm_checkpoint,
        warm_checkpoint_sha256=warm_checkpoint_sha256,
        deployable_factory=deployable_factory,
        wrapper_factory=wrapper_factory,
        warm_loader=warm_loader,
    )
    target_device = torch.device(device)
    model.to(target_device)

    predecessor_bank = None
    predecessor_cache_construction_seconds = 0.0
    if execution.predecessor_logit_teacher is not None:
        if predecessor_model_loader is None:
            raise ValueError("dual-teacher execution requires a predecessor model loader")
        predecessor_rng = capture_rng_state()
        predecessor_sampler_state = (
            None if sampler_external_snapshot is None
            else copy.deepcopy(sampler_external_snapshot())
        )
        predecessor_model = None
        try:
            # Model factories construct a fresh randomly initialized public
            # architecture before strict-loading the authenticated state.  The
            # construction itself must not advance the paired student's
            # Python/NumPy/Torch/CUDA streams.
            predecessor_model = predecessor_model_loader(
                execution.predecessor_logit_teacher
            )
            predecessor_started = time.perf_counter()
            predecessor_bank = build_predecessor_logit_bank(
                predecessor_model,
                (
                    predecessor_batches()
                    if predecessor_batches is not None
                    else _train_batch_stream(
                        train_batches, pass_index=0, start_batch=0,
                    )
                ),
                device=device,
                expected_rows=config.train_rows,
            )
            predecessor_cache_construction_seconds = time.perf_counter() - predecessor_started
        finally:
            if predecessor_model is not None:
                _release_frozen_model(predecessor_model)
            predecessor_model = None
            restore_rng_state(predecessor_rng)
            if predecessor_sampler_state is not None:
                if sampler_external_restore is None:
                    raise ValueError(
                        "predecessor pass snapshot lacks matching sampler restore"
                    )
                sampler_external_restore(predecessor_sampler_state)

    class_weights = torch.as_tensor(
        np.asarray(parent_recipe["class_weights"], dtype=np.float32),
        device=target_device,
    )
    optimizer = _optimizer_for(model, config)
    interval = _IntervalMeans(target_device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration_output = (
        output / "calibration"
        if calibration_dir is None
        else Path(calibration_dir)
    )
    if calibration_selection is not None:
        write_immutable_json(
            calibration_output / "selection.json", calibration_selection,
        )
        published_selection = load_json(calibration_output / "selection.json")
        if published_selection != dict(calibration_selection):
            raise ValueError("published calibration-selection artifact differs")
        validate_calibration_selection_artifact(
            published_selection,
            expected_campaign_sha256=campaign_sha256,
            expected_parent_logit_counterpart_node_id=execution.parent_counterpart,
        )
    diagnostic_materialized = None
    diagnostic_artifact = None
    if diagnostic_batches is not None:
        def take_fixed_diagnostic_batch() -> Mapping[str, Any]:
            iterator = iter(diagnostic_batches())
            try:
                value = next(iterator)
            except StopIteration as error:
                raise ValueError("HCWDL-RKD fixed diagnostic batch is empty") from error
            try:
                next(iterator)
            except StopIteration:
                return value
            raise ValueError("HCWDL-RKD diagnostic factory must yield exactly one batch")

        diagnostic_materialized, diagnostic_artifact = _materialize_diagnostic_batch(
            take_fixed_diagnostic_batch,
            execution=execution,
            mode=mode,
            lineage=lineage,
            representation_recipe_sha256=representation_recipe_sha256,
            output=output,
            calibration_directory=calibration_output,
        )
    acceptance_full_loss = None
    if acceptance_full_loss_binding is not None:
        if diagnostic_materialized is None or diagnostic_artifact is None:
            raise ValueError("acceptance full-loss evidence lacks its diagnostic batch")
        probe = exercise_full_representation_loss(
            model,
            execution_id=execution_id,
            batch=diagnostic_materialized,
            target_bank=target_bank,
            predecessor_bank=predecessor_bank,
            class_weights=np.asarray(
                parent_recipe["class_weights"], dtype=np.float32,
            ),
            token_resources=token_resources,
            relation_resources=relation_resources,
            device=device,
            shuffled_representation_joiner=shuffled_representation_joiner,
            require_canonical_early_backbone=True,
        )
        acceptance_full_loss = _build_acceptance_real_batch_full_loss_record(
            binding=acceptance_full_loss_binding,
            probe=probe,
            execution=execution,
            registered_execution_id=lineage["execution"],
            diagnostic_batch_sha256=diagnostic_artifact["content_hash"],
            target_generation_sha256=lineage["target_generation"],
            target_logical_sha256=lineage["target_logical"],
            target_manifest_sha256=target_bank.manifest["content_hash"],
        )
        write_immutable_json(
            output / "acceptance_real_batch_full_loss.json",
            acceptance_full_loss,
        )
    resume_root = output / "resume"
    selected_root = output / "checkpoints" / "selected" / "staging" / "candidates"
    selected_root.mkdir(parents=True, exist_ok=True)
    target_bindings = {
        "generation_sha256": lineage["target_generation"],
        "logical_target_sha256": lineage["target_logical"],
    }
    selection_state: dict[str, Any] = {
        "best": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
    }
    calibration = _default_calibration_state(
        diagnostic_batch_sha256=(
            None if diagnostic_artifact is None else diagnostic_artifact["content_hash"]
        ),
        selection=calibration_selection,
    )
    validation_history: list[dict[str, Any]] = []
    completed_pass = 0
    completed_update = 0
    next_canonical_batch = 0
    pass_identity_digests = np.empty((0, 32), dtype=np.uint8)
    resume_sequence = 0
    loaded, scan = load_highest_valid_resume(
        resume_root, expected_lineage=lineage,
    )
    if scan.invalid_commits:
        # A corrupt newest generation is recoverable only when an older valid
        # generation exists.  The audit remains available in the final report.
        if loaded is None:
            raise ValueError("all committed HCWDL-RKD resume generations are invalid")
    if loaded is not None:
        (
            completed_pass, completed_update, next_canonical_batch,
            validation_history, selection_state, calibration, pass_identity_digests,
        ) = _restore_state(
            loaded.state, model=model, optimizer=optimizer, config=config,
            interval=interval, sampler_external_restore=sampler_external_restore,
            expected_rng_streams=rng_streams,
        )
        resume_sequence = loaded.sequence + 1
        if loaded.state["target_bindings"] != target_bindings:
            raise ValueError("resume target logical/generation hashes differ")
        expected_diagnostic_sha256 = (
            None if diagnostic_artifact is None else diagnostic_artifact["content_hash"]
        )
        if calibration.get("diagnostic_batch_sha256") != expected_diagnostic_sha256:
            raise ValueError("resume diagnostic-batch lineage differs")
    active_projection_names = tuple(
        sorted(name for name, _ in model.representation_heads.projection_items())
    )

    def external_sampler_state() -> Mapping[str, Any]:
        return {} if sampler_external_snapshot is None else sampler_external_snapshot()

    def snapshot_state() -> dict[str, Any]:
        return _state_snapshot(
            model=model, optimizer=optimizer, config=config,
            completed_pass=completed_pass, completed_update=completed_update,
            next_canonical_batch=next_canonical_batch, interval=interval,
            validation_history=validation_history,
            selection_state=selection_state, calibration=calibration,
            target_bindings=target_bindings,
            rng_streams=rng_streams,
            producer_runtime_signature=producer_runtime_signature,
            sampler_external_state=external_sampler_state(),
            pass_identity_digests=pass_identity_digests,
        )

    def commit_resume() -> None:
        nonlocal resume_sequence
        publish_resume_generation(
            resume_root,
            sequence=resume_sequence,
            state=snapshot_state(),
            lineage=lineage,
            completed_pass=completed_pass,
            completed_update=completed_update,
            next_canonical_batch=next_canonical_batch,
            active_projections=active_projection_names,
            calibration_artifact_hashes=calibration["artifact_hashes"],
            retain_generations=2,
        )
        resume_sequence += 1
        # Candidate checkpoints are external members of the resume state.
        # Retain every candidate referenced by either of the two committed
        # resume generations, and prune only after the older generation has
        # itself been retired.  This preserves fallback when the newest
        # commit/state becomes corrupt after a best-checkpoint change.
        _prune_checkpoint_candidates_for_resume(
            selected_root, resume_root=resume_root, lineage=lineage,
        )

    def maybe_calibrate() -> None:
        nonlocal calibration
        phases = []
        if completed_pass == 2:
            names = tuple(name for name in execution.active_components if name != "relation")
            phases.append(("jet_set", names))
        if completed_pass == 4 and "relation" in execution.active_components:
            phases.append(("relation", ("relation",)))
        for phase, names in phases:
            if phase in calibration["artifact_hashes"]:
                continue
            if calibration_batches is None:
                raise ValueError(f"HCWDL-RKD {phase} calibration batches are absent")
            boundary_state = snapshot_state()
            _, boundary_resume_logical_sha256 = build_state_inventory(boundary_state)
            boundary_deployable_sha256 = _state_dict_logical_sha256(
                model.deployable_model.state_dict(),
            )
            boundary_training_sha256 = _state_dict_logical_sha256(model.state_dict())
            observed_calibration_identities: list[str] = []
            observed_calibration_batches = 0

            def validated_calibration_batches():
                nonlocal observed_calibration_batches
                seen: set[bytes] = set()
                for batch_index, raw_batch in enumerate(calibration_batches(phase)):
                    normalized_batch = normalize_hlt_batch(raw_batch)
                    keys = [bytes(row) for row in normalized_batch.identity_digests]
                    if seen.intersection(keys):
                        raise ValueError("calibration population repeats an identity")
                    seen.update(keys)
                    if batch_index == 0 and diagnostic_artifact is not None and (
                        identity_order_sha256(normalized_batch.identity_digests)
                        != diagnostic_artifact["payload"]["ordered_identity_sha256"]
                    ):
                        raise ValueError(
                            "diagnostic batch is not the first canonical calibration batch"
                        )
                    observed_calibration_identities.extend(key.hex() for key in keys)
                    observed_calibration_batches += 1
                    yield raw_batch

            result = _run_calibration(
                phase=phase, component_names=names, execution=execution,
                model=model, optimizer=optimizer, batches=validated_calibration_batches(),
                device=target_device, amp_dtype=config.amp_dtype,
                class_weights=class_weights, target_bank=target_bank,
                predecessor_bank=predecessor_bank,
                shuffled_representation_joiner=shuffled_representation_joiner,
                token_resources=token_resources, relation_resources=relation_resources,
                expected_batches=calibration_expected_batches,
                minimum_valid_batches=calibration_minimum_valid_batches,
                external_snapshot=calibration_external_snapshot,
                external_restore=calibration_external_restore,
            )
            if calibration_selection is None:
                raise ValueError("calibration barrier lacks its selection artifact")
            expected_calibration_identities = list(
                calibration_selection["ordered_identity_sha256s"]
            )
            if observed_calibration_identities != expected_calibration_identities:
                raise ValueError(
                    "calibration batches differ from the frozen selection order"
                )
            ordered_calibration_sha256 = canonical_sha256(
                observed_calibration_identities,
            )
            if ordered_calibration_sha256 != calibration_selection[
                "canonical_identity_order_sha256"
            ]:
                raise ValueError("calibration identity-order hash differs")
            if calibration.get("selection_sha256") != calibration_selection[
                "content_hash"
            ] or calibration.get("ordered_selection_sha256") != (
                calibration_selection["ordered_selection_sha256"]
            ):
                raise ValueError("calibration selection lineage changed")
            prior_order = calibration.get("ordered_calibration_identity_sha256")
            if prior_order is not None and prior_order != ordered_calibration_sha256:
                raise ValueError("calibration phases use different identity orders")
            calibration["ordered_calibration_identity_sha256"] = (
                ordered_calibration_sha256
            )
            calibration["calibration_rows"] = len(observed_calibration_identities)
            artifact = build_versioned_artifact(
                GRADIENT_CALIBRATION_CONTRACT,
                parents={
                    "execution": lineage["execution"],
                    "representation_recipe": lineage["representation_recipe"],
                    "target_generation": lineage["target_generation"],
                    "target_logical": lineage["target_logical"],
                    "calibration_selection": calibration_selection["content_hash"],
                },
                payload={
                    "execution_id": execution_id,
                    "phase": phase,
                    "completed_pass": completed_pass,
                    "completed_update": completed_update,
                    "shuffle_map_sha256": shuffle_map_sha256,
                    "boundary_deployable_state_logical_sha256": (
                        boundary_deployable_sha256
                    ),
                    "boundary_training_state_logical_sha256": (
                        boundary_training_sha256
                    ),
                    "boundary_resume_state_logical_sha256": (
                        boundary_resume_logical_sha256
                    ),
                    "ordered_calibration_identity_sha256": (
                        ordered_calibration_sha256
                    ),
                    "ordered_selection_sha256": calibration_selection[
                        "ordered_selection_sha256"
                    ],
                    "calibration_rows": len(observed_calibration_identities),
                    "calibration_batches": observed_calibration_batches,
                    "result": _calibration_result_payload(result),
                },
            )
            write_immutable_json(calibration_output / f"{phase}.json", artifact)
            _apply_calibration_result(
                calibration, phase=phase, result=result,
                artifact_hash=artifact["content_hash"], completed_pass=completed_pass,
                completed_update=completed_update,
            )
            if _state_dict_logical_sha256(
                model.deployable_model.state_dict(),
            ) != boundary_deployable_sha256 or _state_dict_logical_sha256(
                model.state_dict(),
            ) != boundary_training_sha256:
                raise RuntimeError("gradient calibration mutated boundary model state")
        _publish_calibration_manifest(
            output,
            execution=execution,
            lineage=lineage,
            calibration=calibration,
            completed_pass=completed_pass,
            completed_update=completed_update,
            calibration_directory=calibration_output,
        )

    parity_inputs = None

    def validation_boundary(*, completed_natural_pass: bool) -> None:
        nonlocal selection_state, parity_inputs
        metrics, current_parity = _validation(
            model, validation_batches(), device=target_device,
            amp_dtype=config.amp_dtype,
        )
        parity_inputs = current_parity
        checkpoint_id = (
            f"{execution_id}__pass_{completed_pass:02d}__update_{completed_update:09d}"
            if completed_natural_pass
            else f"{execution_id}__smoke_update_{completed_update:09d}"
        )
        row = {
            "checkpoint_id": checkpoint_id,
            "completed_pass": completed_pass,
            "update": completed_update,
            "validation": metrics,
            "selector_inputs": {
                name: {
                    "value": float(metrics[name]),
                    "hex": float(metrics[name]).hex(),
                }
                for name in (
                    "macro_ovr_auc", "cross_entropy",
                    "macro_mean_log_qcd_rejection_at_50pct_signal",
                )
            },
        }
        validation_history.append(row)
        maybe_calibrate()
        if diagnostic_materialized is None:
            if config.mode in {"scientific", "smoke"}:
                raise ValueError("HCWDL-RKD fixed diagnostic batch is absent")
            row["representation_diagnostic"] = {
                "status": "not_run_in_synthetic_test",
                "scientific_authorization": False,
            }
        else:
            row["representation_diagnostic"] = run_representation_diagnostic(
                execution=execution, model=model, optimizer=optimizer,
                batch=diagnostic_materialized, completed_pass=completed_pass,
                completed_update=completed_update,
                device=target_device, amp_dtype=config.amp_dtype,
                class_weights=class_weights, target_bank=target_bank,
                predecessor_bank=predecessor_bank,
                shuffled_representation_joiner=shuffled_representation_joiner,
                token_resources=token_resources,
                relation_resources=relation_resources,
                calibration=calibration,
                parameter_selector=diagnostic_parameter_selector,
                external_snapshot=diagnostic_external_snapshot,
                external_restore=diagnostic_external_restore,
            )
        row["boundary_order"] = [
            "validation",
            "required_gradient_calibration_barrier",
            "representation_diagnostic",
            "boundary_resume_commit",
        ]
        selected = min(validation_history, key=checkpoint_key)
        if selected["checkpoint_id"] == checkpoint_id:
            candidate_state = snapshot_state()
            candidate_state["selection_state"] = {
                "best": copy.deepcopy(row),
                "checkpoint_path": None,
                "checkpoint_sha256": None,
            }
            path, digest = _publish_selected_training_checkpoint(
                output, execution=execution, row=row, state=candidate_state,
            )
            selection_state = {
                "best": copy.deepcopy(row),
                "checkpoint_path": str(path.resolve()),
                "checkpoint_sha256": digest,
            }
            commit_resume()
        else:
            commit_resume()

    terminal_bounded = False
    while completed_update < config.active_total_updates:
        pass_index = completed_pass
        start_batch = next_canonical_batch
        observed_batches = start_batch
        observed_rows = len(pass_identity_digests)
        if observed_rows > config.train_rows:
            raise ValueError("resume in-pass identity count exceeds train rows")
        seen_pass_identities = {bytes(row) for row in pass_identity_digests}
        model.train()
        stream = _train_batch_stream(
            train_batches, pass_index=pass_index, start_batch=start_batch,
        )
        for offset, raw in enumerate(stream):
            batch_index = start_batch + offset
            observed_batches += 1
            batch = normalize_hlt_batch(raw)
            observed_rows += len(batch.identity_digests)
            keys = [bytes(row) for row in batch.identity_digests]
            if seen_pass_identities.intersection(keys):
                raise ValueError("natural-population pass repeats an identity")
            seen_pass_identities.update(keys)
            pass_identity_digests = np.ascontiguousarray(np.concatenate(
                (pass_identity_digests, batch.identity_digests), axis=0,
            ))
            if batch_index != next_canonical_batch:
                raise RuntimeError("canonical train batch cursor skipped a batch")
            features, vectors, mask, visible, family, labels = _batch_tensors(
                batch, target_device,
            )
            optimizer.zero_grad(set_to_none=True)
            lr = _learning_rate(config, completed_update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16,
                enabled=config.amp_dtype == "bfloat16",
            ):
                surfaces = model.forward_hcwdl_surfaces(
                    features, vectors, mask, visible, family,
                )
            targets = _target_tensors(
                target_bank, batch.identity_digests, device=target_device,
                execution=execution,
                shuffled_representation_joiner=shuffled_representation_joiner,
            )
            predecessor_logits = (
                None
                if predecessor_bank is None
                else torch.as_tensor(
                    predecessor_bank.join(batch.identity_digests),
                    device=target_device, dtype=torch.float32,
                ).detach()
            )
            effective_pass = effective_pass_for_update(
                completed_update, config.updates_per_pass,
            )
            loss = compute_node_loss(
                execution=execution, model=model, surfaces=surfaces, labels=labels,
                class_weights=class_weights, privileged_targets=targets,
                predecessor_logits=predecessor_logits,
                calibration_scales=_calibration_scales(
                    calibration, execution, effective_pass,
                ),
                effective_pass=effective_pass,
                token_resources=token_resources,
                relation_resources=relation_resources,
            )
            loss.total.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite HCWDL-RKD gradient at {name}")
            optimizer.step()
            completed_update += 1
            next_canonical_batch = batch_index + 1
            interval_values = {
                "total": loss.total,
                "base_total": loss.base["total"],
                "ce": loss.base["ce"],
                "hlt_kd": loss.base["hlt_kd"],
                "privileged_kd": loss.base["privileged_kd"],
                **loss.reporting_components,
            }
            interval.add(interval_values, len(labels))
            if completed_update % config.logging_interval_updates == 0:
                interval.flush(update=completed_update)
            if completed_update == preemption_wait_after_update:
                # The USR1 acceptance worker commits the exact cursor first,
                # then blocks until the operating system delivers a signal.
                # A returned callback without a recorded signal fails closed;
                # reference and resume actions never enter this barrier.
                assert preemption_requested is not None
                assert preemption_wait is not None
                _commit_then_wait_for_preemption(
                    preemption_requested=preemption_requested,
                    commit_resume=commit_resume,
                    preemption_wait=preemption_wait,
                )
                raise RepresentationTrainingInterrupted(
                    f"HCWDL-RKD state committed after update {completed_update}"
                )
            should_preempt = preemption_requested is not None and preemption_requested()
            should_stop = stop_after_update is not None and completed_update == stop_after_update
            if should_preempt or should_stop:
                commit_resume()
                raise RepresentationTrainingInterrupted(
                    f"HCWDL-RKD state committed after update {completed_update}"
                )
            if completed_update >= config.active_total_updates:
                if next_canonical_batch < config.updates_per_pass:
                    terminal_bounded = True
                break
        if terminal_bounded:
            validation_boundary(completed_natural_pass=False)
            break
        if observed_batches != config.updates_per_pass or observed_rows != config.train_rows:
            raise ValueError("natural-population pass batch/row count differs")
        if identity_set_sha256(pass_identity_digests) != target_manifest_payload[
            "identity_set_sha256"
        ]:
            raise ValueError("natural-population pass identity coverage differs")
        completed_pass += 1
        next_canonical_batch = 0
        pass_identity_digests = np.empty((0, 32), dtype=np.uint8)
        validation_boundary(completed_natural_pass=True)

    expected_validations = (
        60 if config.mode == "scientific" else 1
        if config.mode == "smoke" else config.training_passes
    )
    if len(validation_history) != expected_validations:
        raise RuntimeError(
            f"HCWDL-RKD validation cadence differs: {len(validation_history)} != {expected_validations}"
        )
    if config.mode == "scientific" and completed_pass != 60:
        raise RuntimeError("scientific HCWDL-RKD node did not complete 60 passes")
    interval.flush(update=completed_update, partial=True)
    if selection_state["checkpoint_path"] is None:
        raise RuntimeError("HCWDL-RKD node has no selected training checkpoint")
    selected_path = Path(selection_state["checkpoint_path"])
    selected_payload = _load_selected_training_checkpoint(
        selected_path, selection_state["checkpoint_sha256"],
    )
    # Preserve the actual terminal optimizer/model/sampler state before the
    # deployable is restored to the validation-selected checkpoint.
    final_training_state = snapshot_state()
    _restore_state(
        selected_payload["state"], model=model, optimizer=optimizer, config=config,
        interval=_IntervalMeans(target_device), sampler_external_restore=None,
        expected_rng_streams=rng_streams,
    )
    if parity_inputs is None:
        _, parity_inputs = _validation(
            model, validation_batches(), device=target_device,
            amp_dtype=config.amp_dtype,
        )
    extraction = dict(extractor(
        model,
        checkpoint_path=(
            output / "checkpoints" / "selected" / "staging"
            / "extraction" / "deployable_state.pt"
        ),
        selected_training_checkpoint_sha256=selection_state["checkpoint_sha256"],
        architecture_attestation_sha256=architecture_attestation_sha256,
        parity_inputs=parity_inputs,
    ))
    if extraction.get("strict_hlt_only") is not True:
        raise ValueError("deployable extractor did not attest strict HLT-only state")
    for name in ("checkpoint_sha256", "report_sha256"):
        require_sha256(extraction.get(name), name=f"deployable extraction {name}")

    selection = select_checkpoint(validation_history)
    if selection["selected_checkpoint_id"] != selection_state["best"]["checkpoint_id"]:
        raise RuntimeError("independent checkpoint selector differs from retained state")
    selection_artifact = build_versioned_artifact(
        CHECKPOINT_SELECTION_CONTRACT,
        parents={
            "execution": lineage["execution"],
            "representation_recipe": lineage["representation_recipe"],
            "selected_training_checkpoint": selection_state["checkpoint_sha256"],
        },
        payload={
            **selection,
            "selector": [
                "highest_macro_ovr_auc", "lowest_cross_entropy",
                "highest_macro_mean_log_qcd_rejection_at_50pct_signal",
                "earliest_optimizer_update", "lexicographically_smallest_checkpoint_identity",
            ],
            "validation_records": len(validation_history),
        },
    )
    write_immutable_json(output / "checkpoint_selection.json", selection_artifact)
    extraction, checkpoint_envelopes = _publish_terminal_checkpoint_envelopes(
        output=output,
        execution=execution,
        lineage=lineage,
        architecture_attestation_sha256=architecture_attestation_sha256,
        selected_path=selected_path,
        selected_sha256=selection_state["checkpoint_sha256"],
        final_state=final_training_state,
        extraction=extraction,
        registered_output_row=registered_output_row,
        publication_owner=publication_owner,
    )
    selected_path = Path(checkpoint_envelopes["selected"]["training_state_path"])
    selected_metrics = copy.deepcopy(selection_state["best"]["validation"])
    report = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT,
        "schema_version": 1,
        "node_id": execution_id,
        "execution_id": execution_id,
        # Physical executions (notably the five confirmation seeds) have a
        # registry identity distinct from the canonical graph node.  The
        # target consumer registry authorizes this immutable 64-hex identity,
        # while ``execution_id`` remains the resolvable graph/control ID.
        "registered_execution_id": lineage["execution"],
        "replicate_seed": int(replicate_seed),
        "campaign_sha256": campaign_sha256,
        "paired_rng_streams": rng_streams,
        "graph_sha256": expected_graph_sha256,
        "recipe_sha256": representation_recipe_sha256,
        "parent_recipe_sha256": parent_recipe_sha256,
        "parent_counterpart": execution.parent_counterpart,
        "control_counterpart": execution.control_counterpart,
        "strategy": execution.short_strategy,
        "track": execution.track,
        "rung": execution.rung,
        "mode": config.mode,
        "student_domain": execution.student_domain,
        "deployment_authorized": execution.deployable,
        "complete": True,
        "scientific_complete": config.mode == "scientific",
        "finite_poor_results_retained": True,
        "performance_early_stopping": False,
        "completed_optimizer_updates": completed_update,
        "completed_natural_population_passes": completed_pass,
        "validation_every_complete_pass": config.mode != "smoke",
        "validation_history": validation_history,
        "validation": selected_metrics,
        "selection_sha256": selection_artifact["content_hash"],
        "selected_checkpoint_id": selection["selected_checkpoint_id"],
        "selected_training_checkpoint_path": str(selected_path),
        "selected_training_checkpoint_sha256": selection_state["checkpoint_sha256"],
        "deployable_extraction": extraction,
        "checkpoint_envelopes": checkpoint_envelopes,
        "interval_mean_history": interval.history,
        "calibration": calibration,
        "calibration_selection_sha256": calibration.get("selection_sha256"),
        "diagnostic_batch_sha256": calibration.get("diagnostic_batch_sha256"),
        "calibration_manifest_sha256": calibration["artifact_hashes"].get(
            "manifest"
        ),
        "boundary_protocol": [
            "validation", "required_gradient_calibration_barrier",
            "representation_diagnostic", "boundary_resume_commit",
        ],
        "target_generation_sha256": lineage["target_generation"],
        "target_logical_sha256": lineage["target_logical"],
        "target_manifest_sha256": target_bank.manifest["content_hash"],
        "target_cache_diagnostics": cache_diagnostics,
        "predecessor_logit_logical_sha256": (
            None if predecessor_bank is None else predecessor_bank.logical_sha256
        ),
        "predecessor_logit_cache_rows": (
            0 if predecessor_bank is None else len(predecessor_bank.identities)
        ),
        "predecessor_logit_cache_bytes": (
            0 if predecessor_bank is None else (
                predecessor_bank.identities.nbytes + predecessor_bank.logits.nbytes
            )
        ),
        "predecessor_logit_cache_construction_seconds": (
            predecessor_cache_construction_seconds
        ),
        "predecessor_model_released_before_optimization": (
            execution.predecessor_logit_teacher is None or predecessor_bank is not None
        ),
        "shuffled_representation_targets": execution.shuffled_representation_targets,
        "shuffle_map_sha256": shuffle_map_sha256,
        "resume_audit": {
            "highest_loaded_sequence": None if loaded is None else loaded.sequence,
            "invalid_commits": [asdict(row) for row in scan.invalid_commits],
            "orphan_files": list(scan.orphan_files),
            "retain_generations": 2,
        },
        "projection_diagnostics": _projection_diagnostic_payload(model),
    })
    validate_representation_training_report(
        report,
        expected_execution_id=execution_id,
        expected_recipe_sha256=representation_recipe_sha256,
    )
    write_immutable_json(output / "training_report.json", report)
    published_report = load_json(output / "training_report.json")
    if validate_representation_training_report(
        published_report,
        expected_execution_id=execution_id,
        expected_recipe_sha256=representation_recipe_sha256,
    ) != report["content_hash"]:
        raise ValueError("published HCWDL-RKD training report differs")
    # A crash can leave an unreferenced candidate; after successful terminal
    # publication only the selected full training state remains.
    for path in selected_root.glob("*.pt"):
        if path.resolve() != selected_path.resolve():
            path.unlink()
    return report


def validate_representation_training_report(
    report: Mapping[str, Any],
    *,
    expected_execution_id: str | None = None,
    expected_recipe_sha256: str | None = None,
) -> str:
    """Authenticate a terminal report and recompute its frozen selector."""

    digest = validate_content_hash(
        report,
        expected_contract=TRAINING_REPORT_CONTRACT,
        expected_schema_version=1,
    )
    required = {
        "node_id", "execution_id", "registered_execution_id",
        "replicate_seed", "campaign_sha256", "paired_rng_streams",
        "graph_sha256", "recipe_sha256",
        "parent_recipe_sha256",
        "parent_counterpart", "strategy", "track", "rung", "mode",
        "student_domain", "deployment_authorized", "complete", "scientific_complete",
        "finite_poor_results_retained", "performance_early_stopping",
        "completed_optimizer_updates", "completed_natural_population_passes",
        "validation_history", "validation", "selection_sha256",
        "selected_checkpoint_id", "selected_training_checkpoint_sha256",
        "deployable_extraction", "checkpoint_envelopes",
        "interval_mean_history", "calibration",
        "calibration_selection_sha256",
        "diagnostic_batch_sha256", "calibration_manifest_sha256",
        "boundary_protocol", "target_generation_sha256",
        "target_logical_sha256", "target_manifest_sha256",
        "target_cache_diagnostics",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"HCWDL-RKD training report lacks {sorted(missing)}")
    execution_id = str(report["execution_id"])
    if report["node_id"] != execution_id:
        raise ValueError("HCWDL-RKD report node/execution identity differs")
    if expected_execution_id is not None and execution_id != expected_execution_id:
        raise ValueError("HCWDL-RKD report execution identity differs")
    execution = resolve_node_execution(execution_id)
    require_sha256(
        report["registered_execution_id"],
        name="training-report registered execution identity",
    )
    seed = report["replicate_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("HCWDL-RKD report replicate seed differs")
    require_sha256(report["campaign_sha256"], name="training-report campaign")
    validate_paired_rng_streams(
        report["paired_rng_streams"],
        execution_id=execution_id,
        replicate_seed=seed,
    )
    expected_graph_sha256 = _registered_graph_sha256(execution_id)
    if report["graph_sha256"] != expected_graph_sha256:
        raise ValueError("HCWDL-RKD report graph lineage differs")
    recipe_sha256 = require_sha256(
        report["recipe_sha256"], name="training-report representation recipe",
    )
    if expected_recipe_sha256 is not None and recipe_sha256 != require_sha256(
        expected_recipe_sha256, name="expected representation recipe",
    ):
        raise ValueError("HCWDL-RKD report recipe lineage differs")
    require_sha256(
        report["parent_recipe_sha256"], name="training-report parent recipe",
    )
    if (
        report["parent_counterpart"] != execution.parent_counterpart
        or report["strategy"] != execution.short_strategy
        or report["track"] != execution.track
        or int(report["rung"]) != execution.rung
    ):
        raise ValueError("HCWDL-RKD report graph semantics differ")
    if (
        report["student_domain"] != execution.student_domain
        or report["deployment_authorized"] is not execution.deployable
        or report["complete"] is not True
        or report["finite_poor_results_retained"] is not True
        or report["performance_early_stopping"] is not False
    ):
        raise ValueError("HCWDL-RKD report completion/deployment semantics differ")
    mode = str(report["mode"])
    if mode not in EXECUTION_MODES:
        raise ValueError("HCWDL-RKD report mode differs")
    if report["scientific_complete"] is not (mode == "scientific"):
        raise ValueError("HCWDL-RKD scientific completion flag differs")
    history = report["validation_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("HCWDL-RKD validation history is empty")
    expected_records = 60 if mode == "scientific" else 1 if mode == "smoke" else None
    if expected_records is not None and len(history) != expected_records:
        raise ValueError("HCWDL-RKD validation record count differs")
    boundary_protocol = [
        "validation", "required_gradient_calibration_barrier",
        "representation_diagnostic", "boundary_resume_commit",
    ]
    if report["boundary_protocol"] != boundary_protocol:
        raise ValueError("HCWDL-RKD boundary protocol differs")
    seen_checkpoints: set[str] = set()
    prior_update = -1
    for row in history:
        checkpoint_key(row)
        checkpoint_id = str(row.get("checkpoint_id", ""))
        if not checkpoint_id or checkpoint_id in seen_checkpoints:
            raise ValueError("HCWDL-RKD validation checkpoint identity differs")
        seen_checkpoints.add(checkpoint_id)
        update = int(row["update"])
        if update <= prior_update:
            raise ValueError("HCWDL-RKD validation updates are not increasing")
        prior_update = update
        selector_inputs = row.get("selector_inputs")
        if not isinstance(selector_inputs, Mapping):
            raise ValueError("HCWDL-RKD hexadecimal selector inputs are absent")
        for name in (
            "macro_ovr_auc", "cross_entropy",
            "macro_mean_log_qcd_rejection_at_50pct_signal",
        ):
            value = float(row["validation"][name])
            supplied = selector_inputs.get(name)
            if not isinstance(supplied, Mapping) or (
                float(supplied.get("value")) != value
                or supplied.get("hex") != value.hex()
            ):
                raise ValueError("HCWDL-RKD selector float serialization differs")
        if row.get("boundary_order") != boundary_protocol:
            raise ValueError("HCWDL-RKD validation boundary ordering differs")
        diagnostic = row.get("representation_diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise ValueError("HCWDL-RKD validation diagnostic is absent")
        if mode in {"scientific", "smoke"} and (
            diagnostic.get("student_forward_calls") != 1
            or diagnostic.get("finite") is not True
        ):
            raise ValueError("HCWDL-RKD fixed diagnostic execution differs")
    selected = select_checkpoint(history)
    if report["selected_checkpoint_id"] != selected["selected_checkpoint_id"]:
        raise ValueError("HCWDL-RKD report checkpoint selector differs")
    selected_row = next(
        row for row in history
        if row["checkpoint_id"] == selected["selected_checkpoint_id"]
    )
    if report["validation"] != selected_row["validation"]:
        raise ValueError("HCWDL-RKD selected validation metrics differ")
    metric_keys = {
        "rows", "cross_entropy", "accuracy", "balanced_accuracy",
        "always_qcd_accuracy", "macro_ovr_auc",
        "macro_mean_log_qcd_rejection_at_50pct_signal",
        "multiclass_brier", "multiclass_brier_score",
        "top_label_ece_15_bin", "confusion_matrix", "per_class",
    }
    if not metric_keys.issubset(report["validation"]):
        raise ValueError("HCWDL-RKD selected validation metric schema is incomplete")
    for name in metric_keys - {"confusion_matrix", "per_class"}:
        value = report["validation"][name]
        if name == "rows":
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("HCWDL-RKD validation row count differs")
        elif value is None or not math.isfinite(float(value)):
            raise FloatingPointError(f"HCWDL-RKD validation metric {name!r} is nonfinite")
    if np.shape(report["validation"]["confusion_matrix"]) != (15, 15):
        raise ValueError("HCWDL-RKD validation confusion matrix differs")
    if not isinstance(report["validation"]["per_class"], Mapping) or len(
        report["validation"]["per_class"]
    ) != 15:
        raise ValueError("HCWDL-RKD validation per-class metric registry differs")
    for name in (
        "selection_sha256", "selected_training_checkpoint_sha256",
        "target_generation_sha256", "target_logical_sha256",
        "target_manifest_sha256",
    ):
        require_sha256(report[name], name=f"training-report {name}")
    cache_diagnostics = report["target_cache_diagnostics"]
    if not isinstance(cache_diagnostics, Mapping):
        raise ValueError("HCWDL-RKD target-cache diagnostics are absent")
    for name in ("construction_seconds", "load_seconds"):
        value = cache_diagnostics.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("HCWDL-RKD target-cache timing differs")
    for name, report_name in (
        ("generation_sha256", "target_generation_sha256"),
        ("logical_sha256", "target_logical_sha256"),
        ("manifest_sha256", "target_manifest_sha256"),
    ):
        if cache_diagnostics.get(name) != report[report_name]:
            raise ValueError("HCWDL-RKD target-cache hash diagnostics differ")
    if "row_selection_sha256" in cache_diagnostics:
        require_sha256(
            cache_diagnostics["row_selection_sha256"],
            name="training-report target-cache row selection",
        )
    extraction = report["deployable_extraction"]
    if (
        not isinstance(extraction, Mapping)
        or extraction.get("student_domain") != execution.student_domain
        or extraction.get("deployment_authorized") is not execution.deployable
        or extraction.get("strict_hlt_only") is not execution.deployable
    ):
        raise ValueError("HCWDL-RKD model extraction domain/deployment differs")
    require_sha256(
        extraction.get("checkpoint_sha256"), name="deployable checkpoint",
    )
    require_sha256(extraction.get("report_sha256"), name="deployable report")
    envelopes = report["checkpoint_envelopes"]
    if not isinstance(envelopes, Mapping) or set(envelopes) != {"selected", "final"}:
        raise ValueError("HCWDL-RKD checkpoint-envelope registry differs")
    for kind in ("selected", "final"):
        row = envelopes[kind]
        if not isinstance(row, Mapping):
            raise ValueError("HCWDL-RKD checkpoint-envelope row differs")
        require_sha256(row.get("envelope_id"), name=f"{kind} envelope identity")
        require_sha256(row.get("commit_sha256"), name=f"{kind} envelope commit")
    diagnostic_sha256 = report["diagnostic_batch_sha256"]
    if mode in {"scientific", "smoke"}:
        require_sha256(diagnostic_sha256, name="diagnostic-batch artifact")
    elif diagnostic_sha256 is not None:
        require_sha256(diagnostic_sha256, name="diagnostic-batch artifact")
    manifest_sha256 = report["calibration_manifest_sha256"]
    if mode == "scientific":
        require_sha256(manifest_sha256, name="gradient-calibration manifest")
    elif manifest_sha256 is not None:
        require_sha256(manifest_sha256, name="gradient-calibration manifest")
    calibration = report["calibration"]
    if not isinstance(calibration, Mapping) or calibration.get(
        "diagnostic_batch_sha256"
    ) != diagnostic_sha256:
        raise ValueError("HCWDL-RKD diagnostic lineage differs in report")
    if calibration.get("artifact_hashes", {}).get("manifest") != manifest_sha256:
        raise ValueError("HCWDL-RKD calibration manifest lineage differs in report")
    if calibration.get("selection_sha256") != report[
        "calibration_selection_sha256"
    ]:
        raise ValueError("HCWDL-RKD calibration-selection lineage differs in report")
    if mode in {"scientific", "smoke"}:
        require_sha256(
            report["calibration_selection_sha256"],
            name="gradient-calibration selection",
        )
        require_sha256(
            calibration.get("ordered_selection_sha256"),
            name="ordered calibration selection",
        )
    if mode == "scientific":
        require_sha256(
            calibration.get("ordered_calibration_identity_sha256"),
            name="ordered calibration population",
        )
        if int(calibration.get("calibration_rows", -1)) != 4096:
            raise ValueError("scientific calibration population size differs")
        rows_by_pass = {int(row["completed_pass"]): row for row in history}
        if 2 not in rows_by_pass or 4 not in rows_by_pass:
            raise ValueError("scientific calibration diagnostic boundaries are absent")
        pass_two = rows_by_pass[2]["representation_diagnostic"]["components"]
        for name in ("jet", "set"):
            if name in execution.active_components and pass_two[name]["status"] not in {
                "active", "inactive_valid_support", "no_eligible_rows",
            }:
                raise ValueError("pass-two diagnostic preceded jet/set calibration")
        if "relation" in execution.active_components and pass_two["relation"][
            "status"
        ] != "not_yet_calibrated":
            raise ValueError("pass-two diagnostic exposed a future relation scale")
        pass_four = rows_by_pass[4]["representation_diagnostic"]["components"]
        if "relation" in execution.active_components:
            if pass_four["relation"]["status"] not in {
                "active", "inactive_valid_support", "no_eligible_rows",
            }:
                raise ValueError("pass-four diagnostic preceded relation calibration")
        elif pass_four["relation"]["status"] != "not_part_of_strategy":
            raise ValueError("RSET diagnostic invented a relation component")
    for interval in report["interval_mean_history"]:
        if int(interval.get("examples", 0)) <= 0:
            raise ValueError("HCWDL-RKD interval aggregate is empty")
        if any(not math.isfinite(float(value)) for value in interval["means"].values()):
            raise FloatingPointError("HCWDL-RKD interval aggregate is nonfinite")
    return digest


__all__ = [
    "ACCEPTANCE_REAL_BATCH_FULL_LOSS_CONTRACT",
    "BATCH_PROTOCOL",
    "EXECUTION_MODES",
    "HLTBatch",
    "InMemoryPredecessorLogits",
    "NodeExecution",
    "NodeLossResult",
    "PAIRED_RNG_STREAMS_CONTRACT",
    "PREDECESSOR_LOGIT_BANK_CONTRACT",
    "RepresentationTrainingConfiguration",
    "RepresentationTrainingInterrupted",
    "SELECTED_TRAINING_CHECKPOINT_CONTRACT",
    "build_predecessor_logit_bank",
    "compute_node_loss",
    "exercise_full_representation_loss",
    "initialize_representation_student",
    "node_base_loss_configuration",
    "normalize_hlt_batch",
    "paired_rng_streams",
    "representation_training_configuration",
    "resolve_node_execution",
    "run_representation_diagnostic",
    "train_hcwdl_representation_node",
    "validate_acceptance_real_batch_full_loss_record",
    "validate_paired_rng_streams",
    "validate_representation_training_report",
]
