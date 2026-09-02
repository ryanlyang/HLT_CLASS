"""Production workers for Strategy-B adjacent learned fusion handoff."""

from __future__ import annotations

import gc
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from hlt_classification.data.cache_contracts import (
    array_sha256, atomic_publish_bytes, deterministic_npz_bytes, load_json,
    require_sha256, sha256_file,
    write_immutable_json,
)
from hlt_classification.models.hcwdl_adjacent_fusion_transformer import (
    AdjacentFusionParticleTransformer, ParameterMatchedSingleViewParticleTransformer,
)
from hlt_classification.models.hcwdl_offline_hlt_fusion_transformer import _content_view
from hlt_classification.models.scouting_particle_transformer import ScoutingParticleTransformer
from .evaluation import classification_metrics, softmax
from .hcwdl_adjacent_learned_handoff_campaign import validate_campaign
from .hcwdl_adjacent_learned_handoff_contracts import (
    CAPACITY_AUDIT_CONTRACT, DIAGNOSTIC_REPORT_CONTRACT,
    EXECUTION_ACCEPTANCE_CONTRACT, EXTRACTED_CHECKPOINT_CONTRACT,
    FINAL_CHECKPOINT_CONTRACT, SELECTED_CHECKPOINT_CONTRACT,
    STAGE_REPORT_CONTRACT, TRAINING_REPORT_CONTRACT, artifact, validate_artifact,
)
from .hcwdl_adjacent_learned_handoff_data import (
    MorphPairCache, MorphPairManager, PairedViewCache, _tagged_pair,
    morph_context_for_pass,
)
from .hcwdl_adjacent_learned_handoff_graph import (
    COORDINATES, GRAPH_SHA256, NODE_REGISTRY, PARENT_COORDINATE, RUNG_ORDER,
    TRAINING, WITHDRAWAL_ALPHA, acquisition_distribution,
    carrier_distribution, distribution_consumers,
)
from .hcwdl_adjacent_learned_handoff_probability import (
    LearnedProbabilityTargets, REPORT_ONLY_ROLES, ROLES, load_role,
    publish_lock, publish_role, validate_lock,
)
from .hcwdl_adjacent_learned_handoff_source import (
    validate_control_lock, validate_source_lock,
)
from .hcwdl_adjacent_learned_withdrawal import alpha_for_effective_pass
from .hcwdl_adjacent_learned_handoff_partition import (
    PARTITION_NAMES, load_partition, publish_partition,
)
from .hcwdl_adjacent_output_handoff_runner import (
    _ValidationSubset, _fullcard_student_caches, _source_training_authority,
)
from .hcwdl_mhpe_tri60_ce_control import load_control_model
from .hcwdl_mhpe_tri60_campaign import ACCOUNT, PARTITION
from .hcwdl_mhpe_tri60_runner import _configure_deterministic_backend
from .hcwdl_mhpe_tri60_runner import _student_caches as tri60_student_caches
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingAuthority, Tri60TrainingRuntime, _BatchPrefetcher,
    _peak_cuda_bytes, _peak_rss_bytes, _torch_bytes, load_tri60_model,
    train_tri60_node,
)
from .hcwdl_tri100_spine4_graph import NODE_REGISTRY as SOURCE_NODES
from .training import derive_seed
from .hcwdl_homotopy import HomotopyCoordinate


LR_SCHEDULE = {
    "kind": TRAINING["schedule"], "warmup_passes": TRAINING["warmup_passes"],
    "hold_through_pass": TRAINING["hold_through_pass"],
    "decay_through_pass": TRAINING["decay_through_pass"],
    "minimum_lr_fraction": TRAINING["learning_rate_floor_fraction"],
}
EARLY_STOPPING = {
    "kind": "macro_auc_patience_v1",
    "minimum_passes": TRAINING["minimum_passes"],
    "patience_passes": TRAINING["patience_passes"],
    "minimum_auc_delta": TRAINING["minimum_auc_delta"],
}


class _CoordinateNode:
    def __init__(self, name: str, seed_alias: str, coordinate=None):
        self.coordinate_name = name; self.coordinate = coordinate or COORDINATES[name]
        self.seed_alias = seed_alias


def training_dir(spec: Mapping[str, Any], node_id: str) -> Path:
    return Path(spec["campaign_root"]) / "training" / node_id


def probability_dir(spec: Mapping[str, Any], distribution_id: str) -> Path:
    return Path(spec["campaign_root"]) / "probabilities" / distribution_id


def training_authority(node_id: str) -> Tri60TrainingAuthority:
    authority = Tri60TrainingAuthority(
        node=NODE_REGISTRY[node_id], graph_sha256=GRAPH_SHA256,
        training_report_contract=TRAINING_REPORT_CONTRACT,
        selected_checkpoint_contract=SELECTED_CHECKPOINT_CONTRACT,
        final_checkpoint_contract=FINAL_CHECKPOINT_CONTRACT,
        allowed_initializations=("fresh", "warm_selected_checkpoint"),
        allowed_training_passes=(100,),
    )
    authority.validate(); return authority


def _runtime() -> Tri60TrainingRuntime:
    return Tri60TrainingRuntime(
        passes=100, batch_size=256, peak_learning_rate=3e-4,
        weight_decay=.01, warmup_fraction=.03,
        minimum_lr_fraction=.05, amp_dtype="bfloat16",
    )


def _source(spec):
    value = load_json(spec["artifact_paths"]["source_lock"])
    validate_source_lock(value); return value


def _controls(spec):
    value = load_json(spec["artifact_paths"]["control_lock"])
    validate_control_lock(value); return value


def _cache_coordinate(
    spec, coordinate: str, *, seed_alias: str,
    coordinate_value: HomotopyCoordinate | None = None,
):
    return _fullcard_student_caches(
        spec, node=_CoordinateNode(coordinate, seed_alias, coordinate_value),
    )[3]


def _standard_caches(spec, node):
    return _fullcard_student_caches(spec, node=node)


def _pair_caches(spec, *, context: str, primary: str, seed_alias: str):
    context_caches = _cache_coordinate(spec, context, seed_alias=seed_alias)
    primary_caches = _cache_coordinate(spec, primary, seed_alias=seed_alias)
    return {
        role: PairedViewCache(
            context_caches[role], primary_caches[role], role=role,
            lineage={"context": context, "primary": primary, "seed_alias": seed_alias},
        ) for role in ("train", "validation")
    }, (context_caches, primary_caches)


def _morph_caches(spec, node):
    primary = _cache_coordinate(spec, "D000", seed_alias=node.seed_alias)

    def build(coordinate_node):
        return _fullcard_student_caches(spec, node=coordinate_node)[3]

    manager = MorphPairManager(
        primary_caches=primary, build_coordinate=build,
        seed_alias=node.seed_alias,
    )
    return {
        role: MorphPairCache(manager, role=role) for role in ("train", "validation")
    }, manager


def _fixed_pair_for_node(spec, node):
    if node.node_id == "DIRECT_VIEW_MORPH_U100_TO_D000":
        return _morph_caches(spec, node)
    context = node.context_coordinate
    if node.node_id == "DIRECT_VIEW_MORPH_WITHDRAW_D000": context = "D000"
    if context is None: raise ValueError("fusion node lacks context coordinate")
    return _pair_caches(
        spec, context=context, primary=node.primary_coordinate,
        seed_alias=node.seed_alias,
    )


def _cache_owners(value):
    """Normalize fixed-pair tuples and dynamic morph managers for cleanup."""

    return list(value) if isinstance(value, tuple) else [value]


def _load_new(spec, node_id: str, *, device: str, factory=None):
    return load_tri60_model(
        training_dir(spec, node_id) / "training_report.json", device=device,
        model_factory=factory or _model_factory(spec, node_id),
        authority=training_authority(node_id),
    )


def _load_source(spec, *, device: str):
    source = _source(spec); upstream = source["upstream_adapter"]
    return load_tri60_model(
        source["u100_report_path"], device=device,
        authority=_source_training_authority(upstream),
    )


def _source_node(spec):
    source = _source(spec)
    node_id = str(source["u100_node_id"])
    try:
        return SOURCE_NODES[node_id]
    except KeyError as error:
        raise ValueError(
            "learned-handoff U100 source node is not registered"
        ) from error


def _ordinary_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _model_factory(spec, node_id: str):
    node = NODE_REGISTRY[node_id]
    context_seed = derive_seed(
        int(spec["replicate_seed"]),
        node.seed_alias + "/fusion_context_architecture",
    )

    def cold_fusion():
        return AdjacentFusionParticleTransformer(
            context_initialization_seed=context_seed,
        )

    if node.role == "parameter_matched_ce":
        return ParameterMatchedSingleViewParticleTransformer
    if node.role == "warm_continue_ce":
        direct_id = f"LEARNED_DIRECT_{node.primary_coordinate}"
        model, _ = _load_new(spec, direct_id, device="cpu", factory=ScoutingParticleTransformer)
        state = _ordinary_state(model); del model
        def warm():
            value = ScoutingParticleTransformer(); value.load_state_dict(state, strict=True); return value
        return warm
    if node.role in {"fusion_withdrawal", "morph_withdrawal"}:
        parent = (
            f"LEARNED_ACQUIRE_{node.primary_coordinate}"
            if node.role == "fusion_withdrawal"
            else "DIRECT_VIEW_MORPH_U100_TO_D000"
        )
        model, _ = _load_new(spec, parent, device="cpu")
        state = _ordinary_state(model); del model
        def warm_fusion():
            value = AdjacentFusionParticleTransformer(); value.load_state_dict(state, strict=True); return value
        return warm_fusion
    if node.input_protocol != "standard_hlt_v1":
        return cold_fusion
    return ScoutingParticleTransformer


def _targets(spec, distribution_id: str, *, consumer: str):
    root = probability_dir(spec, distribution_id)
    lock, _ = validate_lock(root / "lock.json", distribution_id=distribution_id)
    if consumer not in lock["consumers"]:
        raise PermissionError("learned-handoff probability consumer unauthorized")
    return LearnedProbabilityTargets(
        root / "train_manifest.json", distribution_id=distribution_id,
    ), lock


def run_partition(spec):
    validate_campaign(spec); node = _source_node(spec)
    _, split_hash, selection_hash, caches, _ = _standard_caches(spec, node)
    try:
        labels = np.concatenate([
            np.asarray(batch["labels"], dtype=np.int16)
            for batch in caches["validation"].iterate_canonical_batches(batch_size=4096)
        ])
        return publish_partition(
            spec["artifact_paths"]["validation_partition"],
            identity_digests=caches["validation"].identity_digests, labels=labels,
            parents={"campaign_spec": spec["content_hash"], "source_lock": spec["parents"]["source_lock"], "population_lock": spec["parents"]["population_lock"], "seed_lock": spec["parents"]["seed_lock"], "split_manifest": split_hash, "selection_manifest": selection_hash},
            source_commit=spec["source_commit"],
        )
    finally: caches.clear()


def build_capacity_audit(spec):
    rows = int(spec["role_counts"]["train"]) + int(spec["role_counts"]["validation"])
    # Two fixed 200-token float32 views plus masks and integer metadata.
    row_bytes = 2 * 200 * (25 * 4 + 1 + 8 + 1 + 1)
    projection = rows * row_bytes
    if projection > 420 * 1024**3:
        raise MemoryError("learned-handoff paired RAM projection exceeds locked limit")
    return artifact({
        "parents": {"campaign_spec": spec["content_hash"], "source_lock": spec["parents"]["source_lock"], "population_lock": spec["parents"]["population_lock"]},
        "paired_view_row_bytes_upper_bound": row_bytes,
        "projected_paired_cache_bytes_upper_bound": projection,
        "locked_ram_limit_bytes": 420 * 1024**3,
        "requested_training_memory_bytes": 500 * 1024**3,
        "morph_context_coordinates_resident_at_once": 1,
        "preflight_coordinate_caches_resident_at_once": 2,
        "full_role_teacher_probability_banks": 11,
        "report_only_probability_banks": 15,
        "projected_total_durable_bytes_upper_bound": int(spec["projected_durable_bytes"]),
        "durable_particle_view_bytes": 0, "durable_hidden_state_bytes": 0,
        "rolling_resume": False, "final_test_accessed": False,
    }, contract=CAPACITY_AUDIT_CONTRACT)


def validate_capacity_audit(spec, value):
    digest = validate_artifact(value, contract=CAPACITY_AUDIT_CONTRACT)
    if dict(value) != build_capacity_audit(spec):
        raise ValueError("learned-handoff capacity audit differs")
    return digest


def run_execution_acceptance(spec, *, device="cuda"):
    import torch
    acceptance_started = time.monotonic()
    validate_campaign(spec)
    target = torch.device(device)
    visible_cuda_devices = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    device_name = (
        torch.cuda.get_device_name(target)
        if target.type == "cuda" and torch.cuda.is_available() else "cpu"
    )
    hostname = socket.gethostname()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_nodes = os.environ.get("SLURM_NNODES")
    slurm_tasks = os.environ.get("SLURM_NTASKS")
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    slurm_account = os.environ.get("SLURM_JOB_ACCOUNT")
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION")
    genuine_tigris = bool(
        re.fullmatch(r"[0-9]+", slurm_job_id or "")
        and slurm_nodes == "1" and slurm_tasks == "1"
        and slurm_cpus_per_task == "72"
        and slurm_account == ACCOUNT and slurm_partition == PARTITION
        and hostname.startswith("gh-a-") and target.type == "cuda"
        and visible_cuda_devices == 1 and "GH200" in device_name.upper()
    )
    if not genuine_tigris:
        raise RuntimeError(
            "learned-handoff preflight requires one genuine Tigris GH200 worker"
        )
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    audit_hash = validate_capacity_audit(spec, audit)
    node = NODE_REGISTRY["LEARNED_ACQUIRE_D080"]
    u100_caches = _cache_coordinate(
        spec, "U100", seed_alias=node.seed_alias,
    )
    owners = [u100_caches]
    boundary_caches = []
    d100_caches = _fullcard_student_caches(
        spec, node=_CoordinateNode(
            "D100", node.seed_alias, HomotopyCoordinate(1, 1, 0, 50),
        ),
    )[3]
    try:
        u100_digest = hashlib.sha256()
        d100_digest = hashlib.sha256()
        endpoint_equal = True
        endpoint_rows = 0
        endpoint_mismatched_arrays = 0
        for role in ("train", "validation"):
            left = u100_caches[role].iterate_canonical_batches(batch_size=4096)
            right = d100_caches[role].iterate_canonical_batches(batch_size=4096)
            from itertools import zip_longest
            sentinel = object()
            for u100_batch, d100_batch in zip_longest(left, right, fillvalue=sentinel):
                if u100_batch is sentinel or d100_batch is sentinel:
                    raise ValueError("learned-handoff U100/D100 batch coverage differs")
                ukey = "hlt" if "hlt" in u100_batch else "privileged"
                dkey = "hlt" if "hlt" in d100_batch else "privileged"
                endpoint_rows += len(u100_batch["labels"])
                u100_digest.update(role.encode())
                d100_digest.update(role.encode())
                for name in (
                    "features", "vectors", "mask", "raw_lengths",
                    "visible_indices", "family_codes", "family_reason_codes",
                ):
                    uvalue = np.asarray(getattr(u100_batch[ukey], name))
                    dvalue = np.asarray(getattr(d100_batch[dkey], name))
                    equal = np.array_equal(uvalue, dvalue)
                    endpoint_equal &= equal
                    endpoint_mismatched_arrays += int(not equal)
                    u100_digest.update(name.encode()); u100_digest.update(uvalue.tobytes())
                    d100_digest.update(name.encode()); d100_digest.update(dvalue.tobytes())
                for name in ("identity_digests", "labels"):
                    uvalue = np.asarray(u100_batch[name])
                    dvalue = np.asarray(d100_batch[name])
                    equal = np.array_equal(uvalue, dvalue)
                    if not equal:
                        raise ValueError(
                            f"learned-handoff U100/D100 {name} differ"
                        )
                    u100_digest.update(name.encode()); u100_digest.update(uvalue.tobytes())
                    d100_digest.update(name.encode()); d100_digest.update(dvalue.tobytes())
        if endpoint_rows != (
            int(spec["role_counts"]["train"])
            + int(spec["role_counts"]["validation"])
        ):
            raise ValueError("learned-handoff U100/D100 population rows differ")
        d100_caches.clear(); gc.collect()

        def boundary_evidence(batch):
            key = "hlt" if "hlt" in batch else "privileged"
            view = batch[key]
            return {
                "rows": len(batch["labels"]),
                "identity_digest_sha256": array_sha256(
                    "identity_digest", np.asarray(batch["identity_digests"]),
                ),
                "features_sha256": array_sha256(
                    "features", np.asarray(view.features),
                ),
                "vectors_sha256": array_sha256(
                    "vectors", np.asarray(view.vectors),
                ),
                "mask_sha256": array_sha256("mask", np.asarray(view.mask)),
            }

        u100_boundary_batch = next(
            u100_caches["validation"].iterate_canonical_batches(
                batch_size=256,
            )
        )
        morph_boundary_evidence = {
            "U100": boundary_evidence(u100_boundary_batch),
        }
        d098_caches = _cache_coordinate(
            spec, "D098", seed_alias=node.seed_alias,
            coordinate_value=HomotopyCoordinate(1, 1, 1, 50),
        )
        boundary_caches.append(d098_caches)
        d098_batch = next(
            d098_caches["validation"].iterate_canonical_batches(
                batch_size=256,
            )
        )
        morph_boundary_evidence["D098"] = boundary_evidence(d098_batch)
        d098_caches.clear(); gc.collect()
        d000_caches = _cache_coordinate(
            spec, "D000", seed_alias=node.seed_alias,
        )
        boundary_caches.append(d000_caches)
        d000_batch = next(
            d000_caches["validation"].iterate_canonical_batches(
                batch_size=256,
            )
        )
        morph_boundary_evidence["D000"] = boundary_evidence(d000_batch)
        d000_caches.clear(); gc.collect()
        d080_caches = _cache_coordinate(
            spec, "D080", seed_alias=node.seed_alias,
        )
        owners.append(d080_caches)
        caches = {
            role: PairedViewCache(
                u100_caches[role], d080_caches[role], role=role,
                lineage={
                    "context": "U100", "primary": "D080",
                    "seed_alias": node.seed_alias,
                },
            )
            for role in ("train", "validation")
        }
        batch = next(caches["validation"].iterate_canonical_batches(batch_size=256))
        view = batch["hlt"]; args = (
            torch.as_tensor(view.features, device=device).float(),
            torch.as_tensor(view.vectors, device=device).float(),
            torch.as_tensor(view.mask, device=device).bool(),
            torch.as_tensor(view.content_source_codes, device=device).to(torch.int8),
        )
        labels = torch.as_tensor(batch["labels"], device=device).long()
        primary_seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias)
        context_seed = derive_seed(
            int(spec["replicate_seed"]),
            node.seed_alias + "/fusion_context_architecture",
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(primary_seed)
            direct_reference = ScoutingParticleTransformer()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(primary_seed)
            model = AdjacentFusionParticleTransformer(
                context_initialization_seed=context_seed,
            )
        direct_state = direct_reference.mod.state_dict()
        fusion_primary_state = model.hlt_mod.state_dict()
        primary_initialization_equal = (
            direct_state.keys() == fusion_primary_state.keys()
            and all(
                torch.equal(direct_state[name], fusion_primary_state[name])
                for name in direct_state
            )
        )
        if not primary_initialization_equal:
            raise RuntimeError("learned-handoff matched primary initialization differs")
        del direct_reference, direct_state, fusion_primary_state
        model = model.to(device).eval()
        context_mask = args[3] == 0
        extracted = model.extract_primary().to(device).eval()
        primary = _content_view(*args, code=1, capacity=200)
        with torch.inference_mode():
            zero = model.forward_zero(*args)
            initial_both = model.forward_fused(*args, alpha=1.0)
            zero_with_unavailable_context = model.forward_zero(
                args[0].masked_fill(context_mask[:, None], float("nan")),
                args[1].masked_fill(context_mask[:, None], float("nan")),
                args[2], args[3],
            )
            actual = extracted(*primary)
            padded = ~args[2]
            padded_zero = model.forward_zero(
                args[0].masked_fill(padded, 37.0),
                args[1].masked_fill(padded, -19.0), args[2], args[3],
            )
        initial_zero_residual_error = float(
            (zero.logits - initial_both.logits).abs().max().item()
        )
        parity = float((zero.logits - actual).abs().max().item())
        unavailable_error = float(
            (zero.logits - zero_with_unavailable_context.logits).abs().max().item()
        )
        padding_error = float(
            (zero.logits - padded_zero.logits).abs().max().item()
        )
        split_calls = 0
        def forbidden_context_split(*_args, **_kwargs):
            nonlocal split_calls
            split_calls += 1
            raise RuntimeError("alpha-zero attempted context construction")
        model._split = forbidden_context_split
        try:
            with torch.inference_mode():
                dispatch_probe = model.forward_zero(*args)
        finally:
            del model._split
        dispatch_error = float(
            (zero.logits - dispatch_probe.logits).abs().max().item()
        )
        if (
            parity > 2e-5 or unavailable_error != 0
            or initial_zero_residual_error != 0 or padding_error > 2e-5
            or split_calls != 0 or dispatch_error != 0
        ):
            raise RuntimeError("learned-handoff alpha-zero extraction/dispatch parity differs")
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(device).type, dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            bf_zero = model.forward_zero(*args).logits.float()
            bf_actual = extracted(*primary).float()
        bf16_parity = float((bf_zero - bf_actual).abs().max().item())
        if bf16_parity > 2e-2:
            raise RuntimeError("learned-handoff BF16 extraction parity differs")
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4,
        )
        both = model.forward_fused(*args, alpha=1.0)
        acquisition_loss = torch.nn.functional.cross_entropy(both.logits.float(), labels)
        acquisition_loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        del both, acquisition_loss
        # Exercise the actual selected-checkpoint-to-withdrawal state transfer
        # through serialized bytes.  Optimizer state is deliberately absent.
        checkpoint_buffer = BytesIO()
        torch.save(_ordinary_state(model), checkpoint_buffer)
        checkpoint_bytes = checkpoint_buffer.getvalue()
        checkpoint_buffer.seek(0)
        transferred_state = torch.load(
            checkpoint_buffer, map_location="cpu", weights_only=True,
        )
        withdrawal_model = AdjacentFusionParticleTransformer(
            context_initialization_seed=context_seed,
        )
        transfer_result = withdrawal_model.load_state_dict(
            transferred_state, strict=True,
        )
        if transfer_result.missing_keys or transfer_result.unexpected_keys:
            raise RuntimeError("learned-handoff miniature state transfer differs")
        withdrawal_model = withdrawal_model.to(device).eval()
        with torch.inference_mode():
            transferred_both = withdrawal_model.forward_fused(
                *args, alpha=1.0,
            ).logits.float()
            transferred_padded_both = withdrawal_model.forward_fused(
                args[0].masked_fill(padded, 37.0),
                args[1].masked_fill(padded, -19.0), args[2], args[3],
                alpha=1.0,
            ).logits.float()
        fusion_padding_error = float(
            (transferred_both - transferred_padded_both).abs().max().item()
        )
        if fusion_padding_error > 2e-5:
            raise RuntimeError("learned-handoff fused padding invariance differs")
        withdrawal_model.train()
        withdrawal_optimizer = torch.optim.AdamW(
            (
                parameter for parameter in withdrawal_model.parameters()
                if parameter.requires_grad
            ),
            lr=3e-4,
        )
        from .hcwdl_adjacent_learned_withdrawal import withdrawal_loss
        output = withdrawal_model.forward_withdrawal(*args, alpha=.5)
        teacher = torch.full_like(output.zero.logits.float(), 1 / 15)
        losses = withdrawal_loss(output, labels, teacher)
        losses["total"].backward(); withdrawal_optimizer.step()
        if not all(
            x.residual_projection.weight.grad is not None
            for x in withdrawal_model.injections
        ):
            raise RuntimeError("learned-handoff cross residual gradients differ")
        withdrawal_model.eval()
        transferred_extract = withdrawal_model.extract_primary().to(device).eval()
        with torch.inference_mode():
            transferred_zero = withdrawal_model.forward_zero(*args).logits.float()
            transferred_actual = transferred_extract(*primary).float()
        transferred_extraction_error = float(
            (transferred_zero - transferred_actual).abs().max().item()
        )
        if transferred_extraction_error > 2e-5:
            raise RuntimeError("learned-handoff miniature extraction differs")
        low_low_batch = _tagged_pair(
            d000_batch, d000_batch, require_identical_views=True,
        )
        low_view = low_low_batch["hlt"]
        low_low_features = torch.as_tensor(
            low_view.features, device=device,
        ).float()
        low_low_vectors = torch.as_tensor(
            low_view.vectors, device=device,
        ).float()
        low_low_mask = torch.as_tensor(
            low_view.mask, device=device,
        ).bool()
        low_low_sources = torch.as_tensor(
            low_view.content_source_codes, device=device,
        ).to(torch.int8)
        low_low_labels = torch.as_tensor(
            low_low_batch["labels"], device=device,
        ).long()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(primary_seed)
            low_low_model = AdjacentFusionParticleTransformer(
                context_initialization_seed=context_seed,
            ).to(device).train()
        low_low_optimizer = torch.optim.AdamW(
            (
                parameter for parameter in low_low_model.parameters()
                if parameter.requires_grad
            ),
            lr=3e-4,
        )
        low_low_logits = low_low_model(
            low_low_features, low_low_vectors, low_low_mask, low_low_sources,
        )
        low_low_loss = torch.nn.functional.cross_entropy(
            low_low_logits.float(), low_low_labels,
        )
        low_low_loss.backward(); low_low_optimizer.step()
        morph_checkpoint_buffer = BytesIO()
        torch.save(_ordinary_state(low_low_model), morph_checkpoint_buffer)
        morph_checkpoint_bytes = morph_checkpoint_buffer.getvalue()
        morph_checkpoint_buffer.seek(0)
        morph_state = torch.load(
            morph_checkpoint_buffer, map_location="cpu", weights_only=True,
        )
        morph_withdrawal_model = AdjacentFusionParticleTransformer(
            context_initialization_seed=context_seed,
        )
        morph_transfer_result = morph_withdrawal_model.load_state_dict(
            morph_state, strict=True,
        )
        if (
            morph_transfer_result.missing_keys
            or morph_transfer_result.unexpected_keys
        ):
            raise RuntimeError(
                "learned-handoff morph miniature state transfer differs"
            )
        morph_withdrawal_model = morph_withdrawal_model.to(device).train()
        morph_withdrawal_optimizer = torch.optim.AdamW(
            (
                parameter for parameter in morph_withdrawal_model.parameters()
                if parameter.requires_grad
            ),
            lr=3e-4,
        )
        morph_withdrawal_output = morph_withdrawal_model.forward_withdrawal(
            low_low_features, low_low_vectors, low_low_mask,
            low_low_sources, alpha=.5,
        )
        morph_teacher = torch.full_like(
            morph_withdrawal_output.zero.logits.float(), 1 / 15,
        )
        morph_losses = withdrawal_loss(
            morph_withdrawal_output, low_low_labels, morph_teacher,
        )
        morph_losses["total"].backward()
        morph_withdrawal_optimizer.step()
        morph_withdrawal_model.eval()
        morph_extract = morph_withdrawal_model.extract_primary().to(device).eval()
        morph_primary = _content_view(
            low_low_features, low_low_vectors, low_low_mask,
            low_low_sources, code=1, capacity=200,
        )
        with torch.inference_mode():
            morph_zero = morph_withdrawal_model.forward_zero(
                low_low_features, low_low_vectors, low_low_mask,
                low_low_sources,
            ).logits.float()
            morph_actual = morph_extract(*morph_primary).float()
        morph_extraction_error = float(
            (morph_zero - morph_actual).abs().max().item()
        )
        if morph_extraction_error > 2e-5:
            raise RuntimeError(
                "learned-handoff morph miniature extraction differs"
            )
        context_head_trainable = any(
            parameter.requires_grad
            for name, parameter in model.context_mod.named_parameters()
            if not name.startswith(("embed.", "pair_embed.", "blocks."))
        )
        if context_head_trainable:
            raise RuntimeError("learned-handoff context owns trainable head parameters")
        parameter_counts = {
            "fusion_total": sum(p.numel() for p in model.parameters()),
            "fusion_trainable": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "ordinary_total": sum(p.numel() for p in extracted.parameters()),
            "ordinary_trainable": sum(
                p.numel() for p in extracted.parameters() if p.requires_grad
            ),
            "parameter_matched_total": sum(
                p.numel()
                for p in ParameterMatchedSingleViewParticleTransformer().parameters()
            ),
            "parameter_matched_trainable": sum(
                p.numel()
                for p in ParameterMatchedSingleViewParticleTransformer().parameters()
                if p.requires_grad
            ),
        }
        parameter_counts["parameter_matched_minus_fusion_trainable"] = (
            parameter_counts["parameter_matched_trainable"]
            - parameter_counts["fusion_trainable"]
        )
        schedule = [morph_context_for_pass(p)[0] for p in range(1, 101)]
        if schedule[:3] != ["U100", "D098", "D096"] or schedule[50:] != ["D000"] * 50:
            raise RuntimeError("learned-handoff morph schedule differs")
        return artifact({
            "parents": {"campaign_spec": spec["content_hash"], "capacity_audit": audit_hash},
            "production_worker": True, "cuda": target.type == "cuda",
            "hostname": hostname, "slurm_job_id": slurm_job_id,
            "slurm_nodes": slurm_nodes, "slurm_tasks": slurm_tasks,
            "slurm_cpus_per_task": slurm_cpus_per_task,
            "slurm_account": slurm_account,
            "slurm_partition": slurm_partition,
            "visible_cuda_devices": visible_cuda_devices,
            "device_name": device_name,
            "genuine_tigris_single_gh200_worker": genuine_tigris,
            "installed_weaver_forward_backward": True,
            "production_batch_size": 256,
            "elapsed_seconds": time.monotonic() - acceptance_started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "peak_cuda_bytes": _peak_cuda_bytes(),
            "alpha_zero_max_abs_error": parity, "alpha_zero_skips_context": True,
            "initial_alpha_one_zero_residual_max_abs_error": initial_zero_residual_error,
            "bf16_alpha_zero_max_abs_error": bf16_parity,
            "unavailable_context_max_abs_error": unavailable_error,
            "cross_residual_gradients_present": True, "parameter_counts": parameter_counts,
            "matched_primary_initialization_equal": True,
            "primary_initialization_seed": primary_seed,
            "context_architecture_initialization_seed": context_seed,
            "production_acquisition_optimizer_step": True,
            "production_withdrawal_backward_step": True,
            "serialized_checkpoint_transfer_sha256": hashlib.sha256(
                checkpoint_bytes
            ).hexdigest(),
            "serialized_checkpoint_excludes_optimizer": True,
            "transferred_alpha_zero_extraction_max_abs_error": (
                transferred_extraction_error
            ),
            "production_low_low_ce_optimizer_step": True,
            "production_morph_withdrawal_backward_step": True,
            "morph_checkpoint_transfer_sha256": hashlib.sha256(
                morph_checkpoint_bytes
            ).hexdigest(),
            "morph_checkpoint_excludes_optimizer": True,
            "morph_transferred_alpha_zero_extraction_max_abs_error": (
                morph_extraction_error
            ),
            "one_way_rich_to_lower_information_flow": True,
            "lower_primary_owns_only_trainable_classifier": True,
            "fully_masked_padding_max_abs_error": padding_error,
            "trained_fusion_padding_max_abs_error": fusion_padding_error,
            "alpha_zero_context_split_calls": split_calls,
            "morph_schedule": schedule, "morph_coordinates_resident_at_once": 1,
            "preflight_coordinate_caches_resident_at_once": 2,
            "morph_boundary_coordinates_exercised": ["U100", "D098", "D000"],
            "morph_boundary_evidence": morph_boundary_evidence,
            "low_low_control_coordinate_exercised": "D000",
            "u100_d100_full_population_byte_equal": bool(endpoint_equal),
            "u100_d100_rows_compared": endpoint_rows,
            "u100_d100_mismatched_array_count": endpoint_mismatched_arrays,
            "u100_endpoint_data_sha256": u100_digest.hexdigest(),
            "d100_endpoint_data_sha256": d100_digest.hexdigest(),
            "morph_first_coordinate": (
                "D100_alias_of_U100" if endpoint_equal else "explicit_U100_then_D098"
            ),
            "passed": True, "final_test_accessed": False,
        }, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    finally:
        for owner in owners: owner.clear()
        for owner in boundary_caches: owner.clear()
        d100_caches.clear()


def validate_execution_acceptance(spec, value):
    digest = validate_artifact(value, contract=EXECUTION_ACCEPTANCE_CONTRACT)
    audit = load_json(spec["artifact_paths"]["capacity_audit"])
    audit_hash = validate_capacity_audit(spec, audit)
    endpoint_equal = value.get("u100_d100_full_population_byte_equal")
    expected_first = (
        "D100_alias_of_U100" if endpoint_equal is True
        else "explicit_U100_then_D098"
    )
    counts = value.get("parameter_counts", {})
    required_counts = {
        "fusion_total", "fusion_trainable", "ordinary_total",
        "ordinary_trainable", "parameter_matched_total",
        "parameter_matched_trainable",
        "parameter_matched_minus_fusion_trainable",
    }
    boundary = value.get("morph_boundary_evidence", {})
    boundary_fields = {
        "rows", "identity_digest_sha256", "features_sha256",
        "vectors_sha256", "mask_sha256",
    }
    boundary_valid = (
        set(boundary) == {"U100", "D098", "D000"}
        and all(set(row) == boundary_fields for row in boundary.values())
        and all(type(row["rows"]) is int and row["rows"] == 256 for row in boundary.values())
        and len({
            row["identity_digest_sha256"] for row in boundary.values()
        }) == 1
    )
    if value.get("one_way_rich_to_lower_information_flow") is not True:
        raise ValueError("learned-handoff information-flow evidence differs")
    if value.get("parents") != {"campaign_spec": spec["content_hash"], "capacity_audit": audit_hash} or value.get("passed") is not True or value.get("production_worker") is not True or value.get("cuda") is not True or not isinstance(value.get("hostname"), str) or not value["hostname"].startswith("gh-a-") or re.fullmatch(r"[0-9]+", str(value.get("slurm_job_id", ""))) is None or value.get("slurm_nodes") != "1" or value.get("slurm_tasks") != "1" or value.get("slurm_cpus_per_task") != "72" or value.get("slurm_account") != ACCOUNT or value.get("slurm_partition") != PARTITION or value.get("visible_cuda_devices") != 1 or "GH200" not in str(value.get("device_name", "")).upper() or value.get("genuine_tigris_single_gh200_worker") is not True or value.get("installed_weaver_forward_backward") is not True or value.get("production_batch_size") != 256 or not isinstance(value.get("elapsed_seconds"), (int, float)) or value["elapsed_seconds"] <= 0 or not isinstance(value.get("peak_rss_bytes"), int) or value["peak_rss_bytes"] <= 0 or not isinstance(value.get("peak_cuda_bytes"), int) or value["peak_cuda_bytes"] <= 0 or value.get("matched_primary_initialization_equal") is not True or value.get("primary_initialization_seed") != derive_seed(int(spec["replicate_seed"]), NODE_REGISTRY["LEARNED_ACQUIRE_D080"].seed_alias) or value.get("context_architecture_initialization_seed") != derive_seed(int(spec["replicate_seed"]), NODE_REGISTRY["LEARNED_ACQUIRE_D080"].seed_alias + "/fusion_context_architecture") or value.get("alpha_zero_skips_context") is not True or value.get("alpha_zero_max_abs_error", float("inf")) > 2e-5 or value.get("bf16_alpha_zero_max_abs_error", float("inf")) > 2e-2 or value.get("initial_alpha_one_zero_residual_max_abs_error") != 0 or value.get("unavailable_context_max_abs_error") != 0 or value.get("fully_masked_padding_max_abs_error", float("inf")) > 2e-5 or value.get("trained_fusion_padding_max_abs_error", float("inf")) > 2e-5 or value.get("alpha_zero_context_split_calls") != 0 or value.get("cross_residual_gradients_present") is not True or value.get("production_acquisition_optimizer_step") is not True or value.get("production_withdrawal_backward_step") is not True or value.get("production_low_low_ce_optimizer_step") is not True or value.get("production_morph_withdrawal_backward_step") is not True or value.get("serialized_checkpoint_excludes_optimizer") is not True or value.get("morph_checkpoint_excludes_optimizer") is not True or value.get("transferred_alpha_zero_extraction_max_abs_error", float("inf")) > 2e-5 or value.get("morph_transferred_alpha_zero_extraction_max_abs_error", float("inf")) > 2e-5 or value.get("lower_primary_owns_only_trainable_classifier") is not True or set(counts) != required_counts or any(type(counts[name]) is not int or counts[name] <= 0 for name in required_counts - {"parameter_matched_minus_fusion_trainable"}) or counts.get("parameter_matched_minus_fusion_trainable") != counts.get("parameter_matched_trainable", 0) - counts.get("fusion_trainable", 0) or type(endpoint_equal) is not bool or value.get("u100_d100_rows_compared") != int(spec["role_counts"]["train"]) + int(spec["role_counts"]["validation"]) or not isinstance(value.get("u100_d100_mismatched_array_count"), int) or value.get("u100_d100_mismatched_array_count") < 0 or value.get("morph_first_coordinate") != expected_first or value.get("morph_schedule") != [morph_context_for_pass(pass_number)[0] for pass_number in range(1, 101)] or value.get("morph_coordinates_resident_at_once") != 1 or value.get("preflight_coordinate_caches_resident_at_once") != 2 or value.get("morph_boundary_coordinates_exercised") != ["U100", "D098", "D000"] or value.get("low_low_control_coordinate_exercised") != "D000" or not boundary_valid:
        raise ValueError("learned-handoff execution acceptance differs")
    require_sha256(
        value.get("serialized_checkpoint_transfer_sha256"),
        name="serialized checkpoint transfer",
    )
    require_sha256(
        value.get("morph_checkpoint_transfer_sha256"),
        name="morph checkpoint transfer",
    )
    require_sha256(value.get("u100_endpoint_data_sha256"), name="U100 endpoint data")
    require_sha256(value.get("d100_endpoint_data_sha256"), name="D100 endpoint data")
    for coordinate, row in boundary.items():
        for field in boundary_fields - {"rows"}:
            require_sha256(
                row[field], name=f"{coordinate} boundary {field}",
            )
    if endpoint_equal and (
        value["u100_endpoint_data_sha256"] != value["d100_endpoint_data_sha256"]
        or value["u100_d100_mismatched_array_count"] != 0
    ):
        raise ValueError("learned-handoff U100/D100 equality evidence differs")
    return digest


def run_fit(spec, node_id: str, *, device="cuda", recovery_spec_sha256=None, execution_source_commit=None):
    validate_campaign(spec); _configure_deterministic_backend(); node = NODE_REGISTRY[node_id]
    acceptance = load_json(spec["artifact_paths"]["execution_acceptance"])
    acceptance_hash = validate_execution_acceptance(spec, acceptance)
    targets = lock = None
    if node.teacher_distribution_id:
        targets, lock = _targets(spec, node.teacher_distribution_id, consumer=node_id)
    started = time.monotonic(); owners = []
    if node.input_protocol == "standard_hlt_v1":
        _, split_hash, selection_hash, caches, input_key = _standard_caches(spec, node)
        owners = [caches]
    else:
        caches, owner = _fixed_pair_for_node(spec, node)
        owners = _cache_owners(owner)
        input_key = "hlt"
        foundation = load_json(spec["artifact_paths"]["foundation_spec"])
        split_hash = foundation["parents"]["split_manifest"]
        selection_hash = foundation["parents"]["selection_manifest"]
    partition, arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    validation = _ValidationSubset(
        caches["validation"], tuple(bytes(x).hex() for x in arrays["identity_digest"][arrays["partition"] == 0]),
        content_hash=partition["content_hash"],
    )
    parents = {
        "campaign_spec": spec["content_hash"], "source_lock": spec["parents"]["source_lock"],
        "graph": GRAPH_SHA256, "recipe": spec["parents"]["recipe"],
        "population_lock": spec["parents"]["population_lock"],
        "seed_lock": spec["parents"]["seed_lock"],
        "split_manifest": split_hash, "selection_manifest": selection_hash,
        "validation_partition": partition["content_hash"], "execution_acceptance": acceptance_hash,
    }
    if lock is not None: parents["probability_lock"] = lock["content_hash"]
    if recovery_spec_sha256 is not None: parents["recovery_spec"] = recovery_spec_sha256
    initialization_lineage = None
    if node.initialization != "fresh":
        parent_id = (
            f"LEARNED_ACQUIRE_{node.primary_coordinate}"
            if node.role == "fusion_withdrawal"
            else f"LEARNED_DIRECT_{node.primary_coordinate}"
            if node.role == "warm_continue_ce"
            else "DIRECT_VIEW_MORPH_U100_TO_D000"
        )
        parent_report = load_json(training_dir(spec, parent_id) / "training_report.json")
        initialization_lineage = {
            "source_report": parent_report["content_hash"],
            "source_checkpoint": parent_report["selected_checkpoint_sha256"],
        }
    try:
        return train_tri60_node(
            node_id=node_id, train_cache=caches["train"], validation_cache=validation,
            input_key=input_key, probability_targets=targets,
            output_dir=training_dir(spec, node_id), parents=parents,
            campaign_spec_sha256=spec["content_hash"], recipe_sha256=spec["parents"]["recipe"],
            execution_source_commit=execution_source_commit or spec["source_commit"],
            replicate_seed=int(spec["replicate_seed"]), device=device,
            runtime=_runtime(), execution_mode="scientific",
            model_factory=_model_factory(spec, node_id),
            preparation_metrics={"student_view_cache_seconds": time.monotonic() - started, "pre_training_total_seconds": time.monotonic() - started},
            authority=training_authority(node_id), learning_rate_schedule=LR_SCHEDULE,
            early_stopping=EARLY_STOPPING, model_input_protocol=node.input_protocol,
            checkpoint_selection_minimum_pass=(
                51 if node.role == "dynamic_view_morph_ce" else 1
            ),
            withdrawal_schedule=(WITHDRAWAL_ALPHA if node.selection_route == "alpha_zero" else None),
            initialization_lineage=initialization_lineage,
        )
    finally:
        for owner in owners: owner.clear()


def _partition_lookup(spec):
    report, arrays = load_partition(spec["artifact_paths"]["validation_partition"])
    return report, {bytes(x): int(code) for x, code in zip(arrays["identity_digest"], arrays["partition"], strict=True)}


def _split_validation(ids, probabilities, labels, lookup):
    codes = np.asarray([lookup[bytes(x)] for x in ids], dtype=np.uint8)
    return {role: (ids[codes == code], probabilities[codes == code], labels[codes == code]) for code, role in enumerate(PARTITION_NAMES)}


def _infer(model, cache, *, sampler_seed: int, device: str, protocol: str,
           alpha: float = 1.0, context_perturbation: str | None = None):
    import torch
    model.eval(); ids=[]; logits=[]; labels=[]
    with torch.inference_mode(), _BatchPrefetcher(cache.iterate_batches(epoch=0, sampler_seed=sampler_seed, batch_size=256)) as batches:
        for batch in batches:
            view=batch["hlt"] if "hlt" in batch else batch["privileged"]; args=(torch.as_tensor(view.features,device=device).float(),torch.as_tensor(view.vectors,device=device).float(),torch.as_tensor(view.mask,device=device).bool())
            if protocol == "standard_hlt_v1":
                if context_perturbation is not None:
                    raise ValueError("single-view inference received a context perturbation")
                value=model(*args)
            else:
                source=torch.as_tensor(view.content_source_codes,device=device).to(torch.int8)
                if context_perturbation is not None:
                    context = source == 0
                    if context_perturbation == "zero":
                        args = (
                            args[0].masked_fill(context[:, None], 0),
                            args[1].masked_fill(context[:, None], 0), args[2],
                        )
                    elif context_perturbation == "zero_primary":
                        primary = source == 1
                        args = (
                            args[0].masked_fill(primary[:, None], 0),
                            args[1].masked_fill(primary[:, None], 0), args[2],
                        )
                    elif context_perturbation == "identity_permutation":
                        order = torch.roll(torch.arange(len(source), device=device), 1)
                        width = source.shape[1] // 2
                        features, vectors, mask = (x.clone() for x in args)
                        features[:, :, :width] = features[order, :, :width]
                        vectors[:, :, :width] = vectors[order, :, :width]
                        mask[:, :, :width] = mask[order, :, :width]
                        source = source.clone(); source[:, :width] = source[order, :width]
                        args = features, vectors, mask
                    else: raise ValueError("unknown learned-handoff context perturbation")
                value=model.forward_fused(*args,source,alpha=alpha).logits
            ids.append(np.asarray(batch["identity_digests"],dtype=np.uint8)); logits.append(value.float().cpu().numpy()); labels.append(np.asarray(batch["labels"],dtype=np.int64))
    return np.concatenate(ids), np.concatenate(logits), np.concatenate(labels)


def _alpha_validation_curve(
    model, cache, *, sampler_seed: int, device: str, protocol: str,
    partition_lookup, expected_ids, expected_labels, alpha_zero_metrics,
    alpha_one_metrics,
):
    """Measure the selected checkpoint at fixed registered fusion strengths.

    Endpoint metrics are reused from the mandatory route diagnostics.  Only
    the three interior strengths require additional inference, and no logits,
    hidden states, or particle views from the sweep are made durable.
    """

    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        if alpha == 0.0:
            metrics = alpha_zero_metrics
        elif alpha == 1.0:
            metrics = alpha_one_metrics
        else:
            ids, logits, labels = _infer(
                model, cache, sampler_seed=sampler_seed, device=device,
                protocol=protocol, alpha=alpha,
            )
            if (
                not np.array_equal(ids, expected_ids)
                or not np.array_equal(labels, expected_labels)
            ):
                raise ValueError(
                    "learned-handoff alpha-curve identities/labels differ"
                )
            split = _split_validation(
                ids, softmax(logits), labels, partition_lookup,
            )
            metrics = classification_metrics(
                np.log(np.maximum(split["V_report"][1], 1e-12)),
                split["V_report"][2],
            )
        rows.append({"alpha": alpha, "metrics": metrics})
    return rows


def _publish_distribution(spec, *, distribution_id, model, report, caches, protocol, node_id, device, execution_source_commit, target_temperature=2.0, seed_alias=None):
    if seed_alias is None:
        seed_alias = NODE_REGISTRY[node_id].seed_alias
    partition, lookup = _partition_lookup(spec); seed=derive_seed(int(spec["replicate_seed"]), str(seed_alias) + "/sampler")
    consumers=distribution_consumers(distribution_id)
    roles = ROLES if consumers else REPORT_ONLY_ROLES
    train_ids = train_logits = train_p = None
    if consumers:
        train_ids, train_logits, _ = _infer(model,caches["train"],sampler_seed=seed,device=device,protocol=protocol,alpha=(0.0 if distribution_id.startswith(("LEARNED_T_","MORPH_T_")) else 1.0))
        train_p=softmax(train_logits).astype(np.float32)
    val_ids,val_logits,val_labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol=protocol,alpha=(0.0 if distribution_id.startswith(("LEARNED_T_","MORPH_T_")) else 1.0))
    val_p=softmax(val_logits).astype(np.float32); split=_split_validation(val_ids,val_p,val_labels,lookup)
    foundation = load_json(spec["artifact_paths"]["foundation_spec"])
    parents={"campaign_spec":spec["content_hash"],"source_lock":spec["parents"]["source_lock"],"population_lock":spec["parents"]["population_lock"],"seed_lock":spec["parents"]["seed_lock"],"foundation":spec["parents"]["foundation"],"split_manifest":foundation["parents"]["split_manifest"],"selection_manifest":foundation["parents"]["selection_manifest"],"graph":GRAPH_SHA256,"recipe":spec["parents"]["recipe"],"validation_partition":partition["content_hash"],"component_report":report["content_hash"],"component_checkpoint":report["selected_checkpoint_sha256"]}
    lineage={node_id:{"report":report["content_hash"],"checkpoint":report["selected_checkpoint_sha256"],**({} if train_logits is None else {"train_logits":array_sha256("logits",train_logits)}),"validation_logits":array_sha256("logits",val_logits)}}
    manifests={}
    values={**({} if train_ids is None else {"train":(train_ids,train_p)}),**{role:(x[0],x[1]) for role,x in split.items()}}
    for role in roles:
        manifests[role]=publish_role(probability_dir(spec,distribution_id),distribution_id=distribution_id,role=role,identity_digests=values[role][0],probabilities=values[role][1],component_order=(node_id,),component_lineage=lineage,consumers=consumers,parents=parents,producer_commit=execution_source_commit or spec["source_commit"],target_temperature=(target_temperature if role=="train" else 1.0))
    lock=publish_lock(probability_dir(spec,distribution_id)/"lock.json",distribution_id=distribution_id,manifests=manifests,consumers=consumers,parents=parents)
    metrics=classification_metrics(np.log(np.maximum(split["V_report"][1],1e-12)),split["V_report"][2])
    stage=artifact({"parents":{**parents,"probability_lock":lock["content_hash"]},"distribution_id":distribution_id,"component_order":[node_id],"report_role":"V_report","validation_metrics":metrics,"poor_metrics_do_not_control_graph":True,"final_test_accessed":False},contract=STAGE_REPORT_CONTRACT)
    write_immutable_json(Path(spec["campaign_root"])/"reports/stages"/f"{distribution_id}.json",stage); return stage


def run_source_reducer(spec, *, device="cuda", execution_source_commit=None):
    model, report = _load_source(spec, device=device); node = _source_node(spec)
    _,_,_,caches,_=_standard_caches(spec,node)
    try: return _publish_distribution(spec,distribution_id="SOURCE_U100",model=model,report=report,caches=caches,protocol="standard_hlt_v1",node_id="SOURCE_U100",device=device,execution_source_commit=execution_source_commit,seed_alias=node.seed_alias)
    finally: caches.clear(); del model; gc.collect()


def _reducer_caches(spec,node):
    if node.node_id == "DIRECT_VIEW_MORPH_U100_TO_D000":
        return _pair_caches(spec,context="D000",primary="D000",seed_alias=node.seed_alias)
    if node.input_protocol == "standard_hlt_v1":
        caches=_standard_caches(spec,node)[3]; return caches,(caches,)
    return _fixed_pair_for_node(spec,node)


def run_model_reducer(spec,node_id,*,device="cuda",execution_source_commit=None):
    node=NODE_REGISTRY[node_id]; model,report=_load_new(spec,node_id,device=device); caches,owners=_reducer_caches(spec,node)
    distribution=(acquisition_distribution(node.primary_coordinate) if node.role=="fusion_acquisition" else "MORPH_Q_D000" if node_id=="DIRECT_VIEW_MORPH_U100_TO_D000" else node_id)
    try:
        stage=_publish_distribution(spec,distribution_id=distribution,model=model,report=report,caches=caches,protocol=node.input_protocol,node_id=node_id,device=device,execution_source_commit=execution_source_commit,target_temperature=2.0)
        if node.input_protocol != "standard_hlt_v1":
            import torch

            _,lookup=_partition_lookup(spec); seed=derive_seed(int(spec["replicate_seed"]),node.seed_alias+"/sampler")
            ids,logits,labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol=node.input_protocol,alpha=0.0)
            split=_split_validation(ids,softmax(logits),labels,lookup)
            zero_ids,zero_logits,zero_labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol=node.input_protocol,alpha=1.0,context_perturbation="zero")
            rich_ids,rich_logits,rich_labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol=node.input_protocol,alpha=1.0,context_perturbation="zero_primary")
            perm_ids,perm_logits,perm_labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol=node.input_protocol,alpha=1.0,context_perturbation="identity_permutation")
            routes_aligned = (
                np.array_equal(ids, zero_ids)
                and np.array_equal(ids, rich_ids)
                and np.array_equal(ids, perm_ids)
                and np.array_equal(labels, zero_labels)
                and np.array_equal(labels, rich_labels)
                and np.array_equal(labels, perm_labels)
            )
            if not routes_aligned:
                raise ValueError(
                    "learned-handoff diagnostic route identities/labels differ"
                )
            zero_split=_split_validation(zero_ids,softmax(zero_logits),zero_labels,lookup); rich_split=_split_validation(rich_ids,softmax(rich_logits),rich_labels,lookup); perm_split=_split_validation(perm_ids,softmax(perm_logits),perm_labels,lookup)
            alpha_ids, alpha_probabilities, alpha_labels = split["V_report"]
            alpha_zero_metrics = classification_metrics(
                np.log(np.maximum(alpha_probabilities, 1e-12)), alpha_labels,
            )
            alpha_curve = _alpha_validation_curve(
                model, caches["validation"], sampler_seed=seed,
                device=device, protocol=node.input_protocol,
                partition_lookup=lookup, expected_ids=ids,
                expected_labels=labels,
                alpha_zero_metrics=alpha_zero_metrics,
                alpha_one_metrics=stage["validation_metrics"],
            )
            diagnostic_root = Path(spec["campaign_root"]) / "reports/diagnostics"
            alpha_path = diagnostic_root / f"{node_id}_alpha_zero_V_report.npz"
            alpha_arrays = {
                "identity_digest": np.ascontiguousarray(alpha_ids, dtype=np.uint8),
                "probabilities": np.ascontiguousarray(alpha_probabilities, dtype=np.float32),
                "label": np.ascontiguousarray(alpha_labels, dtype=np.int16),
            }
            atomic_publish_bytes(alpha_path, deterministic_npz_bytes(alpha_arrays))
            diagnostic=artifact({"parents":{"campaign_spec":spec["content_hash"],"training_report":report["content_hash"],"stage":stage["content_hash"]},"node_id":node_id,"both_views_metrics":stage["validation_metrics"],"zero_context_metrics":classification_metrics(np.log(np.maximum(zero_split["V_report"][1],1e-12)),zero_split["V_report"][2]),"rich_context_only_zero_primary_metrics":classification_metrics(np.log(np.maximum(rich_split["V_report"][1],1e-12)),rich_split["V_report"][2]),"identity_permuted_context_metrics":classification_metrics(np.log(np.maximum(perm_split["V_report"][1],1e-12)),perm_split["V_report"][2]),"alpha_zero_metrics":alpha_zero_metrics,"alpha_validation_curve":alpha_curve,"selected_residual_gate_sigmoids":[float(torch.sigmoid(injection.gate_logit.detach()).cpu().item()) for injection in model.injections],"alpha_zero_V_report_path":str(alpha_path.resolve()),"alpha_zero_V_report_sha256":sha256_file(alpha_path),"alpha_zero_V_report_array_sha256":{name:array_sha256(name,value) for name,value in alpha_arrays.items()},"all_routes_use_identical_V_report_identities":routes_aligned,"all_routes_use_identical_V_report_labels":routes_aligned,"final_test_accessed":False},contract=DIAGNOSTIC_REPORT_CONTRACT)
            write_immutable_json(diagnostic_root/f"{node_id}.json",diagnostic)
        return stage
    finally:
        for owner in owners: owner.clear()
        del model; gc.collect()


def run_extract(spec,node_id,distribution_id,*,device="cuda",recovery_spec_sha256=None):
    import torch
    node=NODE_REGISTRY[node_id]; model,report=_load_new(spec,node_id,device=device); caches,owners=_reducer_caches(spec,node)
    try:
        batch=next(caches["validation"].iterate_canonical_batches(batch_size=256)); view=batch["hlt"]
        args=(torch.as_tensor(view.features,device=device).float(),torch.as_tensor(view.vectors,device=device).float(),torch.as_tensor(view.mask,device=device).bool(),torch.as_tensor(view.content_source_codes,device=device).to(torch.int8))
        extracted=model.extract_primary().to(device).eval(); primary=_content_view(*args,code=1,capacity=200)
        with torch.inference_mode(): expected=model.forward_zero(*args).logits.float(); actual=extracted(*primary).float()
        error=float((expected-actual).abs().max().item())
        if error>2e-5: raise RuntimeError("learned-handoff extracted primary parity differs")
        root=Path(spec["campaign_root"])/"deployable"/distribution_id
        payload={"contract":EXTRACTED_CHECKPOINT_CONTRACT,"schema_version":1,"node_id":node_id,"distribution_id":distribution_id,"campaign_spec_sha256":spec["content_hash"],"source_training_report_sha256":report["content_hash"],"source_selected_checkpoint_sha256":report["selected_checkpoint_sha256"],"model":_ordinary_state(extracted),"input_fields":["features","vectors","mask"],"context_modules_present":False,"deployable_exact_hlt":node.deployable,"final_test_accessed":False}
        checkpoint=root/"selected_model.pt"; atomic_publish_bytes(checkpoint,_torch_bytes(payload))
        audit=artifact({"parents":{"campaign_spec":spec["content_hash"],"training_report":report["content_hash"],**({} if recovery_spec_sha256 is None else {"recovery_spec":recovery_spec_sha256})},"node_id":node_id,"distribution_id":distribution_id,"checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":sha256_file(checkpoint),"maximum_abs_error":error,"physical_single_view_extraction":True,"context_modules_present":False,"deployable_exact_hlt":node.deployable,"input_fields":["features","vectors","mask"],"final_test_accessed":False},contract=DIAGNOSTIC_REPORT_CONTRACT)
        write_immutable_json(root/"extraction.json",audit)
        _, lookup = _partition_lookup(spec)
        seed = derive_seed(int(spec["replicate_seed"]), node.seed_alias + "/sampler")
        route_metrics = {}
        route_ids = []
        route_labels = []
        for route, alpha, perturbation in (
            ("both_views", 1.0, None), ("zero_context", 1.0, "zero"),
            ("rich_context_only_zero_primary", 1.0, "zero_primary"),
            ("identity_permuted_context", 1.0, "identity_permutation"),
            ("alpha_zero", 0.0, None),
        ):
            ids, logits, labels = _infer(
                model, caches["validation"], sampler_seed=seed, device=device,
                protocol=node.input_protocol, alpha=alpha,
                context_perturbation=perturbation,
            )
            split = _split_validation(ids, softmax(logits), labels, lookup)
            route_metrics[route] = classification_metrics(
                np.log(np.maximum(split["V_report"][1], 1e-12)),
                split["V_report"][2],
            )
            route_ids.append(ids)
            route_labels.append(labels)
        routes_aligned = (
            all(np.array_equal(route_ids[0], value) for value in route_ids[1:])
            and all(
                np.array_equal(route_labels[0], value)
                for value in route_labels[1:]
            )
        )
        if not routes_aligned:
            raise ValueError(
                "learned-handoff extraction route identities/labels differ"
            )
        alpha_curve = _alpha_validation_curve(
            model, caches["validation"], sampler_seed=seed,
            device=device, protocol=node.input_protocol,
            partition_lookup=lookup, expected_ids=route_ids[0],
            expected_labels=route_labels[0],
            alpha_zero_metrics=route_metrics["alpha_zero"],
            alpha_one_metrics=route_metrics["both_views"],
        )
        diagnostic = artifact({
            "parents": {"campaign_spec": spec["content_hash"],
                        "training_report": report["content_hash"],
                        "extraction": audit["content_hash"]},
            "node_id": node_id, "route_metrics": route_metrics,
            "alpha_validation_curve": alpha_curve,
            "selected_residual_gate_sigmoids": [
                float(torch.sigmoid(injection.gate_logit.detach()).cpu().item())
                for injection in model.injections
            ],
            "withdrawal_alpha_trajectory": [
                {
                    "pass": pass_number,
                    "alpha": alpha_for_effective_pass(
                        WITHDRAWAL_ALPHA, effective_pass=pass_number,
                    ),
                }
                for pass_number in range(1, int(report["validations"]) + 1)
            ],
            "registered_withdrawal_alpha_trajectory": [
                {
                    "pass": pass_number,
                    "alpha": alpha_for_effective_pass(
                        WITHDRAWAL_ALPHA, effective_pass=pass_number,
                    ),
                }
                for pass_number in range(1, 101)
            ],
            "both_minus_alpha_zero_auc": float(
                route_metrics["both_views"]["macro_ovr_auc"]
            ) - float(route_metrics["alpha_zero"]["macro_ovr_auc"]),
            "all_routes_use_identical_validation_identities": routes_aligned,
            "all_routes_use_identical_validation_labels": routes_aligned,
            "selection_route": "alpha_zero", "final_test_accessed": False,
        }, contract=DIAGNOSTIC_REPORT_CONTRACT)
        write_immutable_json(
            Path(spec["campaign_root"]) / "reports/diagnostics" / f"{node_id}.json",
            diagnostic,
        )
        return audit
    finally:
        for owner in owners: owner.clear()
        del model; gc.collect()


def _load_extracted(spec,distribution_id,*,device):
    import torch
    root=Path(spec["campaign_root"])/"deployable"/distribution_id; audit=load_json(root/"extraction.json"); validate_artifact(audit,contract=DIAGNOSTIC_REPORT_CONTRACT)
    path=Path(audit["checkpoint_path"])
    if sha256_file(path)!=audit["checkpoint_sha256"]: raise ValueError("learned-handoff extracted bytes differ")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("contract")!=EXTRACTED_CHECKPOINT_CONTRACT or payload.get("context_modules_present") is not False or payload.get("deployable_exact_hlt") is not NODE_REGISTRY[audit["node_id"]].deployable: raise ValueError("learned-handoff extracted payload differs")
    model=ScoutingParticleTransformer(); model.load_state_dict(payload["model"],strict=True); model.to(device).float().eval()
    report={"content_hash":audit["content_hash"],"selected_checkpoint_sha256":audit["checkpoint_sha256"]}
    return model,report


def run_extracted_reducer(spec,node_id,distribution_id,*,device="cuda",execution_source_commit=None):
    model,report=_load_extracted(spec,distribution_id,device=device); node=NODE_REGISTRY[node_id]
    caches=_standard_caches(spec,node)[3]
    try: return _publish_distribution(spec,distribution_id=distribution_id,model=model,report=report,caches=caches,protocol="standard_hlt_v1",node_id=node_id,device=device,execution_source_commit=execution_source_commit)
    finally: caches.clear(); del model; gc.collect()


def run_control_reducer(spec,control_id,*,device="cuda"):
    controls=_controls(spec); _,lookup=_partition_lookup(spec)
    if control_id=="M0CE60": model,report=load_control_model(controls["m0ce60_report_path"],device=device); node=_CoordinateNode("D000","control/m0")
    elif control_id=="U000":
        model,report=load_tri60_model(controls["pure_offline_u000_report_path"],device=device)
        node=_CoordinateNode("U100","control/u000")
    else: raise KeyError("unknown learned-handoff control")
    if control_id == "U000":
        control_spec = load_json(controls["pure_offline_u000_campaign_spec_path"])
        caches = tri60_student_caches(control_spec, node_id="U000")[-2]
    else:
        caches=_standard_caches(spec,node)[3]
    seed=derive_seed(int(spec["replicate_seed"]),node.seed_alias+"/sampler")
    try:
        ids,logits,labels=_infer(model,caches["validation"],sampler_seed=seed,device=device,protocol="standard_hlt_v1"); split=_split_validation(ids,softmax(logits),labels,lookup)
        stage=artifact({"parents":{"campaign_spec":spec["content_hash"],"control_report":report["content_hash"]},"distribution_id":f"CONTROL_{control_id}","report_role":"V_report","validation_metrics":classification_metrics(np.log(np.maximum(split["V_report"][1],1e-12)),split["V_report"][2]),"final_test_accessed":False},contract=STAGE_REPORT_CONTRACT)
        write_immutable_json(Path(spec["campaign_root"])/"reports/stages"/f"CONTROL_{control_id}.json",stage); return stage
    finally: caches.clear(); del model; gc.collect()


__all__=["build_capacity_audit","probability_dir","run_control_reducer","run_execution_acceptance","run_extract","run_extracted_reducer","run_fit","run_model_reducer","run_partition","run_source_reducer","training_authority","training_dir","validate_capacity_audit","validate_execution_acceptance"]
