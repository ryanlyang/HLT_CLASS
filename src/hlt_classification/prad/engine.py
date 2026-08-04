"""Fixed-budget, exactly resumable training engine for PRAD student graphs."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    require_sha256,
    sha256_file,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.data.part_inputs import build_particle_transformer_inputs
from hlt_classification.data.replicas import replica_for
from hlt_classification.models.prad_particle_transformer import PradParticleTransformer
from hlt_classification.training.engine import DisabledScaler
from hlt_classification.training.checkpoints import restore_model_runtime_state

from .cache import PradCacheDataset
from .artifacts import prad_view_config_sha256
from .checkpoints import (
    PradSelectionRecord,
    build_prad_checkpoint_payload,
    build_prad_model_checkpoint_payload,
    load_completed_prad_training_report,
    load_prad_checkpoint,
    load_prad_model_checkpoint,
    prad_selection_is_better,
    remove_transient_prad_checkpoint,
    restore_prad_checkpoint_state,
    save_prad_checkpoint,
    save_prad_model_checkpoint,
)
from .evaluation import binary_auc, prad_classification_metrics
from .experiments import experiment_requires_teacher
from .training import (
    PradTrainingConfig,
    assert_frozen_teacher_has_no_gradients,
    configure_student_stage,
    deterministic_relation_shuffle,
    freeze_teacher,
    kd_coefficient,
    map_offline_pairs_to_hlt,
    pack_training_pair_payload,
    semantic_targets_from_assignments,
    stage_for_epoch,
    student_loss,
    unpack_training_pair_payload,
)

PRAD_TRAINING_REPORT_CONTRACT = "hlt_classification_prad_training_report_v1"
PRAD_SAMPLER_CONTRACT = "hlt_classification_prad_shard_epoch_sampler_v1"


class PradEpochScheduler:
    """Stage-aware LR controller with cosine decay during stage C."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: PradTrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.epoch = -1

    def set_epoch(self, epoch: int) -> None:
        stage = stage_for_epoch(self.config, epoch)
        if stage in {"A", "B"}:
            relation_lr = self.config.relation_lr_a_b
            pretrained_lr = self.config.pretrained_lr_b
        else:
            stage_c_epoch = epoch - self.config.stage_a_epochs - self.config.stage_b_epochs
            denominator = max(1, self.config.stage_c_epochs - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * stage_c_epoch / denominator))
            relation_lr = self.config.relation_lr_c * cosine
            pretrained_lr = self.config.pretrained_lr_c * cosine
        self.optimizer.param_groups[0]["lr"] = relation_lr
        self.optimizer.param_groups[1]["lr"] = pretrained_lr
        self.epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"contract": "hlt_classification_prad_epoch_scheduler_v1", "epoch": self.epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("contract") != "hlt_classification_prad_epoch_scheduler_v1":
            raise ValueError("PRAD scheduler contract differs")
        epoch = int(state["epoch"])
        if epoch >= 0:
            self.set_epoch(epoch)
        else:
            self.epoch = -1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _plan(
    dataset: PradCacheDataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, int, np.ndarray], ...]:
    digest = hashlib.sha256(
        f"{PRAD_SAMPLER_CONTRACT}\0{seed}\0{epoch}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    rows: list[tuple[int, int, np.ndarray]] = []
    for shard_id in rng.permutation(len(dataset.records)).tolist():
        record = dataset.records[int(shard_id)]
        start, stop = int(record["row_start"]), int(record["row_stop"])
        order = rng.permutation(stop - start).astype(np.int64)
        for cursor in range(0, len(order), batch_size):
            rows.append((start, stop, np.ascontiguousarray(order[cursor : cursor + batch_size])))
    return tuple(rows)


def _plan_sha256(plan: Sequence[tuple[int, int, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for start, stop, indices in plan:
        digest.update(f"{start}:{stop}".encode("ascii"))
        digest.update(b"\0")
        digest.update(indices.tobytes())
    return digest.hexdigest()


def _check_cache_population(
    reference: PradCacheDataset,
    other: PradCacheDataset,
    *,
    expected_kind: str,
    expected_role: str,
) -> None:
    if other.manifest.get("cache_kind") != expected_kind:
        raise ValueError("PRAD training cache kind differs")
    if other.manifest.get("logical_role") != expected_role:
        raise ValueError("PRAD training cache role differs")
    if (
        len(reference) != len(other)
        or reference.manifest.get("identity_order_sha256")
        != other.manifest.get("identity_order_sha256")
    ):
        raise ValueError("PRAD training cache populations differ")


def _tensor_inputs(
    arrays: Mapping[str, np.ndarray],
    *,
    view: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tokens = np.asarray(arrays[f"{view}_tokens"], dtype=np.float32)
    mask = np.asarray(arrays[f"{view}_mask"], dtype=np.bool_)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    keys = tuple(str(value) for value in arrays["identity_keys"].tolist())
    inputs = build_particle_transformer_inputs(
        tokens, mask, labels, keys, source_view=view
    )
    tensors = {
        name: torch.from_numpy(value).to(device)
        for name, value in inputs.model_inputs().items()
    }
    return tensors, torch.from_numpy(labels).to(device=device, dtype=torch.long)


def _take(arrays: Mapping[str, np.ndarray], local: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(value[local]) for name, value in arrays.items()}


def _replica_batch(
    caches: Mapping[int, PradCacheDataset],
    *,
    start: int,
    stop: int,
    local: np.ndarray,
    epoch: int,
    policy: str,
) -> dict[str, np.ndarray]:
    by_replica = {
        replica: _take(cache.read_range(start, stop), local)
        for replica, cache in caches.items()
    }
    base = {name: np.array(value, copy=True) for name, value in by_replica[0].items()}
    keys = [str(value) for value in base["identity_keys"].tolist()]
    selected = np.asarray(
        [
            replica_for(
                policy=policy,
                logical_role="model_train",
                epoch=epoch,
                canonical_identity=key,
            )
            for key in keys
        ],
        dtype=np.int64,
    )
    if not set(selected.tolist()).issubset(caches):
        raise ValueError("PRAD training replica cache set is incomplete")
    for name in ("hlt_tokens", "hlt_mask", "measurement_states"):
        base[name] = np.ascontiguousarray(
            np.stack(
                [by_replica[int(replica)][name][row] for row, replica in enumerate(selected)]
            )
        )
    base["selected_replicas"] = selected
    return base


def _replica_target_batch(
    caches: Mapping[int, PradCacheDataset],
    *,
    start: int,
    stop: int,
    local: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    by_replica = {
        replica: _take(cache.read_range(start, stop), local)
        for replica, cache in caches.items()
    }
    if len(selected) != len(local) or not set(selected.tolist()).issubset(caches):
        raise ValueError("PRAD target replica selection differs")
    result = {}
    for name in by_replica[0]:
        if name in {"identity_keys", "labels"}:
            result[name] = by_replica[0][name]
        else:
            result[name] = np.ascontiguousarray(
                np.stack(
                    [by_replica[int(replica)][name][row] for row, replica in enumerate(selected)]
                )
            )
    return result


def _replica_target_indices(
    caches: Mapping[int, PradCacheDataset],
    indices: np.ndarray,
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(indices) != len(selected) or not set(selected.tolist()).issubset(caches):
        raise ValueError("PRAD shuffled target replica selection differs")
    by_replica = {
        replica: cache.read_indices(indices) for replica, cache in caches.items()
    }
    result = {}
    for name in by_replica[0]:
        if name in {"identity_keys", "labels"}:
            result[name] = by_replica[0][name]
        else:
            result[name] = np.ascontiguousarray(
                np.stack(
                    [by_replica[int(replica)][name][row] for row, replica in enumerate(selected)]
                )
            )
    return result


def _teacher_batch(
    teacher: PradParticleTransformer | None,
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    relation_dim: int,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, particles = arrays["offline_mask"].shape
    labels = torch.from_numpy(np.asarray(arrays["labels"], np.int64)).to(device)
    if teacher is None:
        return (
            torch.zeros(rows, particles, particles, relation_dim, device=device),
            torch.zeros(rows, heads, particles, particles, device=device),
            torch.zeros(rows, 10, device=device),
            torch.ones(rows, device=device),
            torch.zeros(rows, heads, particles, particles, device=device),
        )
    assert_frozen_teacher_has_no_gradients(teacher)
    inputs, _ = _tensor_inputs(arrays, view="offline", device=device)
    teacher.eval()
    with torch.no_grad():
        output = teacher.forward_with_relations(**inputs)
        confidence = F.softmax(output.logits.float(), dim=-1)[
            torch.arange(rows, device=device), labels
        ]
    return (
        output.relation,
        output.privileged_bias,
        output.logits,
        confidence,
        output.standard_bias,
    )


def _autocast(config: PradTrainingConfig, device: torch.device):
    if config.amp_dtype == "none":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _validation(
    model: PradParticleTransformer,
    cache: PradCacheDataset,
    *,
    target_cache: PradCacheDataset | None,
    teacher: PradParticleTransformer | None,
    batch_size: int,
    device: torch.device,
    config: PradTrainingConfig,
) -> dict[str, Any]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            arrays = cache.read_range(start, stop)
            inputs, targets = _tensor_inputs(arrays, view="hlt", device=device)
            with _autocast(config, device):
                if config.experiment.oracle_bias:
                    if target_cache is None or teacher is None:
                        raise ValueError("oracle validation lacks targets or teacher")
                    target_arrays = target_cache.read_range(start, stop)
                    mapping = torch.from_numpy(
                        target_arrays["hlt_to_offline"].astype(np.int64)
                    ).to(device)
                    _, teacher_bias, _, _, _ = _teacher_batch(
                        teacher,
                        arrays,
                        device=device,
                        relation_dim=model.relation.relation_dim,
                        heads=model.attention_heads,
                    )
                    mapped_last, _ = map_offline_pairs_to_hlt(
                        teacher_bias.permute(0, 2, 3, 1), mapping
                    )
                    output = model.forward_oracle(
                        **inputs,
                        offline_teacher_bias=mapped_last.permute(0, 3, 1, 2),
                    ).logits
                else:
                    output = model(**inputs)
            if output.shape != (len(targets), 10) or not torch.isfinite(output).all():
                raise FloatingPointError("PRAD validation logits are invalid")
            logits.append(output.float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    return prad_classification_metrics(
        np.concatenate(logits).astype(np.float32),
        np.concatenate(labels).astype(np.int64),
    )


def _relation_validation_diagnostics(
    model: PradParticleTransformer,
    teacher: PradParticleTransformer | None,
    cache: PradCacheDataset,
    target_cache: PradCacheDataset,
    *,
    batch_size: int,
    device: torch.device,
    config: PradTrainingConfig,
    max_pairs: int = 100_000,
) -> dict[str, Any]:
    """Measure selected-checkpoint relation fidelity on a bounded val prefix."""

    student_relation: list[np.ndarray] = []
    teacher_relation: list[np.ndarray] = []
    student_bias: list[np.ndarray] = []
    teacher_bias_values: list[np.ndarray] = []
    semantic_scores: list[list[np.ndarray]] = [[], [], []]
    semantic_labels: list[list[np.ndarray]] = [[], [], []]
    pair_count = 0
    jets = 0
    matched_particles = 0
    hlt_particles = 0
    matched_pairs = 0
    hlt_pairs = 0
    matched_pt = 0.0
    hlt_pt = 0.0
    model.eval()
    if teacher is not None:
        teacher.eval()
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            if pair_count >= max_pairs:
                break
            stop = min(start + batch_size, len(cache))
            arrays = cache.read_range(start, stop)
            targets = target_cache.read_range(start, stop)
            inputs, _ = _tensor_inputs(arrays, view="hlt", device=device)
            mapping = torch.from_numpy(targets["hlt_to_offline"].astype(np.int64)).to(device)
            assignments = torch.from_numpy(targets["ca_assignments"].astype(np.int64)).to(device)
            semantic_target, semantic_valid = semantic_targets_from_assignments(
                assignments, mapping
            )
            if teacher is None:
                batch, offline_particles = arrays["offline_mask"].shape
                teacher_relation_raw = torch.zeros(
                    batch,
                    offline_particles,
                    offline_particles,
                    model.relation.relation_dim,
                    device=device,
                )
                teacher_bias_raw = torch.zeros(
                    batch,
                    model.attention_heads,
                    offline_particles,
                    offline_particles,
                    device=device,
                )
                teacher_pair_embed = teacher_bias_raw
            else:
                (
                    teacher_relation_raw,
                    teacher_bias_raw,
                    _,
                    _,
                    teacher_pair_embed,
                ) = _teacher_batch(
                    teacher,
                    arrays,
                    device=device,
                    relation_dim=model.relation.relation_dim,
                    heads=model.attention_heads,
                )
            mapped_relation, pair_mask = map_offline_pairs_to_hlt(
                teacher_relation_raw, mapping
            )
            if config.experiment.relation_target == "teacher_pair_embed":
                teacher_bias_raw = teacher_pair_embed
            mapped_bias_last, _ = map_offline_pairs_to_hlt(
                teacher_bias_raw.permute(0, 2, 3, 1), mapping
            )
            mapped_bias = mapped_bias_last.permute(0, 3, 1, 2)
            payload, layout = pack_training_pair_payload(
                teacher_relation=mapped_relation,
                teacher_bias=mapped_bias,
                semantic_targets=semantic_target,
                semantic_valid=semantic_valid,
                pair_mask=pair_mask,
            )
            with _autocast(config, device):
                output = model.forward_training(**inputs, pair_payload=payload)
            if output.aligned_pair_payload is None:
                raise RuntimeError("PRAD relation diagnostic payload was not returned")
            aligned = unpack_training_pair_payload(output.aligned_pair_payload, layout)
            mask = aligned["pair_mask"]
            available = int(mask.sum().cpu())
            take = min(available, max_pairs - pair_count)
            if take:
                if teacher is not None:
                    student_relation.append(
                        output.relation[mask][:take].float().cpu().numpy()
                    )
                    teacher_relation.append(
                        aligned["teacher_relation"][mask][:take]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    student_bias.append(
                        output.privileged_bias.permute(0, 2, 3, 1)[mask][:take]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    teacher_bias_values.append(
                        aligned["teacher_bias"]
                        .permute(0, 2, 3, 1)[mask][:take]
                        .float()
                        .cpu()
                        .numpy()
                    )
                for scale in range(3):
                    valid = aligned["semantic_valid"][..., scale][mask][:take]
                    if bool(valid.any()):
                        semantic_scores[scale].append(
                            output.semantic_logits[..., scale][mask][:take][valid]
                            .float()
                            .cpu()
                            .numpy()
                        )
                        semantic_labels[scale].append(
                            aligned["semantic_targets"][..., scale][mask][:take][valid]
                            .to(torch.bool)
                            .cpu()
                            .numpy()
                        )
                pair_count += take
            original_mask = arrays["hlt_mask"]
            original_mapping = targets["hlt_to_offline"] >= 0
            hlt_particles += int(original_mask.sum())
            matched_particles += int((original_mapping & original_mask).sum())
            per_jet_particles = original_mask.sum(axis=1)
            per_jet_matched = (original_mapping & original_mask).sum(axis=1)
            hlt_pairs += int(np.sum(per_jet_particles * np.maximum(per_jet_particles - 1, 0)))
            matched_pairs += int(np.sum(per_jet_matched * np.maximum(per_jet_matched - 1, 0)))
            hlt_pt += float((arrays["hlt_tokens"][:, :, 0] * original_mask).sum())
            matched_pt += float(
                (arrays["hlt_tokens"][:, :, 0] * original_mapping * original_mask).sum()
            )
            jets += stop - start
    if pair_count == 0:
        return {"available": False, "reason": "no_matched_validation_pairs"}

    def smooth_l1(left: np.ndarray, right: np.ndarray) -> float:
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        return float(np.mean(np.where(difference < 1.0, 0.5 * difference**2, difference - 0.5)))

    if teacher is None:
        relation_smooth_l1 = None
        bias_smooth_l1 = None
        correlation = None
    else:
        student_relation_array = np.concatenate(student_relation)
        teacher_relation_array = np.concatenate(teacher_relation)
        student_bias_array = np.concatenate(student_bias)
        teacher_bias_array = np.concatenate(teacher_bias_values)
        relation_smooth_l1 = smooth_l1(
            student_relation_array, teacher_relation_array
        )
        bias_smooth_l1 = smooth_l1(student_bias_array, teacher_bias_array)
        left = student_bias_array.ravel().astype(np.float64)
        right = teacher_bias_array.ravel().astype(np.float64)
        correlation = (
            None
            if np.std(left) == 0.0 or np.std(right) == 0.0
            else float(np.corrcoef(left, right)[0, 1])
        )
    aucs: dict[str, Any] = {}
    for scale in range(3):
        if not semantic_scores[scale]:
            aucs[f"same_exclusive_{scale + 2}_subjet"] = None
            continue
        scores = np.concatenate(semantic_scores[scale])
        labels = np.concatenate(semantic_labels[scale])
        try:
            aucs[f"same_exclusive_{scale + 2}_subjet"] = binary_auc(scores, labels)
        except ValueError:
            aucs[f"same_exclusive_{scale + 2}_subjet"] = None
    return {
        "available": True,
        "sampling": "deterministic_validation_identity_prefix",
        "max_pairs": max_pairs,
        "pairs": pair_count,
        "jets": jets,
        "teacher_relation_available": teacher is not None,
        "normalized_relation_smooth_l1": relation_smooth_l1,
        "normalized_bias_smooth_l1": bias_smooth_l1,
        "teacher_student_bias_pearson": correlation,
        "semantic_auc": aucs,
        "vertex_auc": None,
        "vertex_status": "unavailable_no_offline_vertex_assignment",
        "matched_particle_coverage": matched_particles / max(hlt_particles, 1),
        "matched_pair_coverage": matched_pairs / max(hlt_pairs, 1),
        "matched_pt_coverage": matched_pt / max(hlt_pt, 1.0e-12),
    }


def train_prad_student(
    *,
    model: PradParticleTransformer,
    teacher: PradParticleTransformer | None,
    train_paired_caches: Mapping[int, PradCacheDataset],
    train_targets: Mapping[int, PradCacheDataset],
    validation_paired_cache: PradCacheDataset,
    validation_targets: PradCacheDataset | None,
    config: PradTrainingConfig,
    semantic_positive_weights: torch.Tensor,
    output_dir: str | Path,
    source_snapshot_sha256: str,
    initialization_checkpoint_sha256: str,
    teacher_checkpoint_sha256: str | None = None,
    device: str | torch.device = "cpu",
    resume: bool = True,
    stop_after_update: int | None = None,
) -> dict[str, Any]:
    """Train one PRAD relation graph; performance never shortens its budget."""

    if 0 not in train_paired_caches:
        raise ValueError("PRAD training requires paired replica zero")
    primary = train_paired_caches[0]
    _check_cache_population(primary, primary, expected_kind="paired_views", expected_role="train")
    for cache in train_paired_caches.values():
        _check_cache_population(primary, cache, expected_kind="paired_views", expected_role="train")
    for replica, cache in train_paired_caches.items():
        expected_view = prad_view_config_sha256(
            logical_role="train",
            replica_id=replica,
            realization_policy=config.realization_policy,
        )
        if cache.manifest["parents"].get("view_config_sha256") != expected_view:
            raise ValueError("PRAD training cache replica/policy lineage differs")
    if set(train_targets) != set(train_paired_caches):
        raise ValueError("PRAD paired and target replica sets differ")
    for replica, cache in train_targets.items():
        _check_cache_population(
            train_paired_caches[replica],
            cache,
            expected_kind="structural_targets",
            expected_role="train",
        )
        if cache.manifest["parents"].get("paired_view_manifest_sha256") != train_paired_caches[replica].manifest_sha256:
            raise ValueError("PRAD target cache has a different paired-view parent")
    if validation_paired_cache.manifest.get("logical_role") != "val":
        raise ValueError("PRAD checkpoint selection requires the validation role")
    if validation_paired_cache.manifest.get("cache_kind") != "paired_views":
        raise ValueError("PRAD validation cache kind differs")
    if validation_targets is not None:
        _check_cache_population(
            validation_paired_cache,
            validation_targets,
            expected_kind="structural_targets",
            expected_role="val",
        )
    if config.experiment.oracle_bias and validation_targets is None:
        raise ValueError("oracle validation requires the validation target cache")
    expected_validation_view = prad_view_config_sha256(
        logical_role="val", replica_id=0, realization_policy=config.realization_policy
    )
    if (
        validation_paired_cache.manifest["parents"].get("view_config_sha256")
        != expected_validation_view
    ):
        raise ValueError("PRAD validation HLT lineage differs")
    if config.realization_policy == "R_MULTI" and set(train_paired_caches) != {0, 1, 2, 3}:
        raise ValueError("R_MULTI requires all four PRAD training replicas")
    source_hash = require_sha256(source_snapshot_sha256, name="source_snapshot_sha256")
    initialization_hash = require_sha256(
        initialization_checkpoint_sha256,
        name="initialization_checkpoint_sha256",
    )
    if teacher is not None and teacher_checkpoint_sha256 is None:
        raise ValueError("PRAD frozen teacher requires a checkpoint hash")
    if teacher is None and teacher_checkpoint_sha256 is not None:
        raise ValueError("PRAD teacher checkpoint supplied without a teacher")
    target = torch.device(device)
    if teacher is not None:
        freeze_teacher(teacher.to(target))
    needs_teacher = experiment_requires_teacher(config.experiment)
    if needs_teacher and teacher is None:
        raise ValueError("registered PRAD graph requires a frozen teacher")
    model = model.to(target)
    relation_modules = (
        model.relation,
        model.relation_to_bias,
        model.gated_bias,
        model.semantic_heads,
    )
    relation_ids = {id(p) for module in relation_modules for p in module.parameters()}
    relation_parameters = [p for p in model.parameters() if id(p) in relation_ids]
    pretrained_parameters = [p for p in model.parameters() if id(p) not in relation_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": relation_parameters, "lr": config.relation_lr_a_b},
            {"params": pretrained_parameters, "lr": config.pretrained_lr_b},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = PradEpochScheduler(optimizer, config)
    scaler = DisabledScaler()
    config_payload = config.to_dict()
    train_cache_set_hash = canonical_sha256(
        {
            "paired": {
                str(replica): cache.manifest_sha256
                for replica, cache in sorted(train_paired_caches.items())
            },
            "targets": {
                str(replica): cache.manifest_sha256
                for replica, cache in sorted(train_targets.items())
            },
        }
    )
    parents = {
        "config_sha256": canonical_sha256(config_payload),
        "source_snapshot_sha256": source_hash,
        "train_cache_set_sha256": train_cache_set_hash,
        "validation_cache_sha256": validation_paired_cache.manifest_sha256,
        "initialization_checkpoint_sha256": initialization_hash,
    }
    if teacher_checkpoint_sha256 is not None:
        parents["teacher_checkpoint_sha256"] = require_sha256(
            teacher_checkpoint_sha256,
            name="teacher_checkpoint_sha256",
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    last_path = root / "last.pt"
    best_path = root / "selected_model.pt"
    final_path = root / "final_model.pt"
    completed = load_completed_prad_training_report(
        root / "training_report.json",
        expected_contract=PRAD_TRAINING_REPORT_CONTRACT,
        expected_config=config_payload,
        expected_parents=parents,
        map_location=target,
    )
    if completed is not None:
        remove_transient_prad_checkpoint(last_path)
        return completed
    epoch = 0
    batch_cursor = 0
    update = 0
    history: list[dict[str, Any]] = []
    best: PradSelectionRecord | None = None
    elapsed_before_resume = 0.0
    invocation_started = time.perf_counter()
    _seed_everything(config.seed)
    if resume and last_path.exists():
        payload = load_prad_checkpoint(
            last_path,
            expected_config=config_payload,
            expected_parents=parents,
            map_location=target,
        )
        restore_prad_checkpoint_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        epoch = int(payload["epoch"])
        update = int(payload["update"])
        batch_cursor = int(payload["sampler_state"]["batch_cursor"])
        history = [dict(item) for item in payload["history"]]
        best = (
            None
            if payload["best_selection"] is None
            else PradSelectionRecord.from_dict(payload["best_selection"])
        )
        elapsed_before_resume = float(payload["elapsed_training_seconds"])
        invocation_started = time.perf_counter()
        if epoch < config.total_epochs:
            active_plan = _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
            if payload["sampler_state"].get("plan_sha256") != _plan_sha256(active_plan):
                raise ValueError("resumed PRAD sampler plan differs")

    shuffle_relations = config.experiment.shuffle_relation_targets

    def checkpoint(path: Path) -> dict[str, str]:
        current_plan = (
            _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
            if epoch < config.total_epochs
            else ()
        )
        payload = build_prad_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config_payload,
            parents=parents,
            epoch=epoch,
            update=update,
            sampler_state={
                "contract": PRAD_SAMPLER_CONTRACT,
                "epoch": epoch,
                "batch_cursor": batch_cursor,
                "plan_sha256": _plan_sha256(current_plan),
            },
            history=history,
            best_selection=best,
            elapsed_training_seconds=(
                elapsed_before_resume + time.perf_counter() - invocation_started
            ),
        )
        return save_prad_checkpoint(path, payload)

    def model_checkpoint(
        path: Path,
        *,
        role: str,
        selection: PradSelectionRecord | None,
    ) -> dict[str, str]:
        return save_prad_model_checkpoint(
            path,
            build_prad_model_checkpoint_payload(
                model=model,
                config=config_payload,
                parents=parents,
                checkpoint_role=role,
                epoch=epoch if selection is None else selection.epoch,
                update=update if selection is None else selection.update,
                selection=selection,
            ),
        )

    while epoch < config.total_epochs:
        stage = stage_for_epoch(config, epoch)
        configure_student_stage(model, config, epoch)
        scheduler.set_epoch(epoch)
        epoch_plan = _plan(primary, batch_size=config.batch_size, seed=config.seed, epoch=epoch)
        if batch_cursor > len(epoch_plan):
            raise ValueError("PRAD resumed batch cursor exceeds epoch plan")
        model.train()
        for plan_index in range(batch_cursor, len(epoch_plan)):
            start, stop, local = epoch_plan[plan_index]
            arrays = _replica_batch(
                train_paired_caches,
                start=start,
                stop=stop,
                local=local,
                epoch=epoch,
                policy=config.realization_policy,
            )
            selected_replicas = arrays.pop("selected_replicas")
            target_arrays = _replica_target_batch(
                train_targets,
                start=start,
                stop=stop,
                local=local,
                selected=selected_replicas,
            )
            inputs, labels = _tensor_inputs(arrays, view="hlt", device=target)
            mapping = torch.from_numpy(target_arrays["hlt_to_offline"].astype(np.int64)).to(target)
            assignments = torch.from_numpy(target_arrays["ca_assignments"].astype(np.int64)).to(target)
            semantic_targets, semantic_valid = semantic_targets_from_assignments(assignments, mapping)
            teacher_arrays = arrays
            if shuffle_relations:
                global_indices = start + local
                batch_keys = [str(value) for value in arrays["identity_keys"].tolist()]
                if len(batch_keys) < 2:
                    raise ValueError(
                        "E10 relation shuffling requires at least two jets per batch"
                    )
                batch_shuffle = deterministic_relation_shuffle(
                    batch_keys,
                    seed=config.seed + epoch * 1_000_003 + plan_index,
                )
                source_indices = global_indices[batch_shuffle]
                teacher_arrays = primary.read_indices(source_indices.astype(np.int64))
            (
                teacher_relation,
                teacher_bias,
                teacher_logits,
                confidence,
                teacher_pair_embed,
            ) = _teacher_batch(
                teacher if needs_teacher else None,
                teacher_arrays,
                device=target,
                relation_dim=model.relation.relation_dim,
                heads=model.attention_heads,
            )
            if config.experiment.relation_target == "teacher_pair_embed":
                teacher_bias = teacher_pair_embed
            if not shuffle_relations:
                mapped_relation, pair_mask = map_offline_pairs_to_hlt(teacher_relation, mapping)
                mapped_bias_last, _ = map_offline_pairs_to_hlt(
                    teacher_bias.permute(0, 2, 3, 1), mapping
                )
            else:
                source_targets = _replica_target_indices(
                    train_targets,
                    source_indices.astype(np.int64),
                    selected_replicas,
                )
                source_mapping = torch.from_numpy(
                    source_targets["hlt_to_offline"].astype(np.int64)
                ).to(target)
                mapped_relation, source_pair_mask = map_offline_pairs_to_hlt(
                    teacher_relation, source_mapping
                )
                mapped_bias_last, _ = map_offline_pairs_to_hlt(
                    teacher_bias.permute(0, 2, 3, 1), source_mapping
                )
                pair_mask = source_pair_mask & (
                    mapping[:, :, None].ge(0)
                    & mapping[:, None, :].ge(0)
                    & ~torch.eye(mapping.shape[1], dtype=torch.bool, device=target)[None]
                )
            mapped_bias = mapped_bias_last.permute(0, 3, 1, 2)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, target):
                if config.experiment.oracle_bias:
                    output = model.forward_oracle(
                        **inputs,
                        offline_teacher_bias=mapped_bias,
                    )
                    aligned = {
                        "teacher_relation": mapped_relation,
                        "teacher_bias": mapped_bias,
                        "semantic_targets": semantic_targets,
                        "semantic_valid": semantic_valid,
                        "pair_mask": pair_mask,
                    }
                else:
                    pair_payload, layout = pack_training_pair_payload(
                        teacher_relation=mapped_relation,
                        teacher_bias=mapped_bias,
                        semantic_targets=semantic_targets,
                        semantic_valid=semantic_valid,
                        pair_mask=pair_mask,
                    )
                    output = model.forward_training(**inputs, pair_payload=pair_payload)
                    if output.aligned_pair_payload is None:
                        raise RuntimeError("PRAD training payload was not returned")
                    aligned = unpack_training_pair_payload(
                        output.aligned_pair_payload, layout
                    )
                loss = student_loss(
                    output=output,
                    labels=labels,
                    experiment=config.experiment,
                    stage=stage,
                    semantic_targets=aligned["semantic_targets"],
                    semantic_valid=aligned["semantic_valid"],
                    semantic_positive_weights=semantic_positive_weights.to(target),
                    teacher_relation=aligned["teacher_relation"],
                    teacher_bias=aligned["teacher_bias"],
                    teacher_logits=teacher_logits,
                    teacher_true_class_confidence=confidence,
                    pair_mask=aligned["pair_mask"],
                    lambda_relation=config.lambda_relation,
                    lambda_semantic=config.lambda_semantic,
                    lambda_kd=kd_coefficient(config, epoch),
                    kd_temperature=config.kd_temperature,
                )
            loss.total.backward()
            if teacher is not None:
                assert_frozen_teacher_has_no_gradients(teacher)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite PRAD gradient for {name}")
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            update += 1
            batch_cursor = plan_index + 1
            if update == 1 or update % config.history_interval_updates == 0:
                history.append(
                    {
                        "kind": "train",
                        "epoch": epoch,
                        "update": update,
                        "stage": stage,
                        "loss": float(loss.total.detach().cpu()),
                        "hard": float(loss.hard.detach().cpu()),
                        "relation": float(loss.relation.detach().cpu()),
                        "relation_bottleneck": float(
                            loss.relation_bottleneck.detach().cpu()
                        ),
                        "relation_bias": float(loss.relation_bias.detach().cpu()),
                        "semantic": float(loss.semantic.detach().cpu()),
                        "kd": float(loss.kd.detach().cpu()),
                        "kd_coefficient": kd_coefficient(config, epoch),
                    }
                )
            if update % config.checkpoint_interval_updates == 0:
                checkpoint(last_path)
            if stop_after_update is not None and update == stop_after_update:
                checkpoint(last_path)
                return {
                    "contract": PRAD_TRAINING_REPORT_CONTRACT,
                    "complete": False,
                    "epoch": epoch,
                    "update": update,
                    "last_checkpoint": str(last_path),
                }

        metrics = _validation(
            model,
            validation_paired_cache,
            target_cache=validation_targets,
            teacher=teacher,
            batch_size=config.batch_size,
            device=target,
            config=config,
        )
        candidate = PradSelectionRecord(
            macro_log_rejection=float(metrics["macro_log_rejection"]),
            accuracy=float(metrics["secondary"]["accuracy"]),
            epoch=epoch,
            update=update,
        )
        history.append({"kind": "validation", "epoch": epoch, "update": update, "metrics": metrics})
        if prad_selection_is_better(candidate, best):
            best = candidate
            model_checkpoint(best_path, role="selected", selection=best)
        epoch += 1
        batch_cursor = 0
        checkpoint(last_path)

    if best is None or not best_path.is_file():
        raise RuntimeError("PRAD training produced no validation checkpoint")
    final_checkpoint = model_checkpoint(final_path, role="final", selection=None)
    training_time_seconds = (
        elapsed_before_resume + time.perf_counter() - invocation_started
    )
    relation_diagnostics: dict[str, Any] = {
        "available": False,
        "reason": "graph_has_no_frozen_teacher_relation_diagnostic",
    }
    selected_payload = load_prad_model_checkpoint(
        best_path,
        expected_config=config_payload,
        expected_parents=parents,
        expected_role="selected",
        map_location=target,
    )
    model.load_state_dict(selected_payload["model_state"], strict=True)
    restore_model_runtime_state(model, selected_payload["model_runtime_state"])
    if (
        validation_targets is not None
        and config.experiment.relation_module
        and not config.experiment.oracle_bias
        and (teacher is not None or config.experiment.semantic_loss)
    ):
        relation_diagnostics = _relation_validation_diagnostics(
            model,
            teacher,
            validation_paired_cache,
            validation_targets,
            batch_size=config.batch_size,
            device=target,
            config=config,
        )
    report = with_content_hash(
        {
            "contract": PRAD_TRAINING_REPORT_CONTRACT,
            "schema_version": 1,
            "complete": True,
            "parents": parents,
            "config": config_payload,
            "epochs_completed": epoch,
            "updates_completed": update,
            "training_time_seconds": training_time_seconds,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "best_selection": best.to_dict(),
            "selected_checkpoint": {
                "path": str(best_path),
                "sha256": sha256_file(best_path),
                "format": "model_only",
            },
            "final_checkpoint": {
                "path": final_checkpoint["path"],
                "sha256": final_checkpoint["sha256"],
                "format": "model_only",
            },
            "history": history,
            "gate_values": model.gated_bias.gates.detach().cpu().tolist(),
            "relation_diagnostics": relation_diagnostics,
            "performance_gate_applied": False,
        }
    )
    write_immutable_json(root / "training_report.json", report)
    remove_transient_prad_checkpoint(last_path)
    return report


__all__ = ["PradEpochScheduler", "train_prad_student"]
