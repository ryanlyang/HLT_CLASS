"""Non-authorizing local HCWDL-RKD dispatcher and full-loss smoke probes.

The fixtures in this module deliberately contain no repository data paths and
cannot issue a final-population reservation, claim, selection, assignment, or
prediction.  They exist to exercise the exact production loss entry point for
every frozen primary node and control before any real-worker acceptance run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import gc
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash

from .hcwdl_representation_contracts import SMOKE_PROBE_CONTRACT
from .hcwdl_representation_contracts import logical_array_sha256
from .hcwdl_representation_graph import CONTROL_REGISTRY, NODE_REGISTRY


FINAL_ROLE_KINDS: Final = frozenset({
    "shared_final_claim", "final_selection", "assignment_shard",
    "assignment_finalize", "data_attestation", "execution_lock",
    "prediction_shard", "prediction_finalize", "metric_join", "final_aggregate",
})


def _identity_digest(index: int) -> np.ndarray:
    return np.frombuffer(index.to_bytes(32, "big"), dtype=np.uint8).copy()


def _vectors(batch: int, tokens: int) -> np.ndarray:
    # Close sub-clusters plus separated clusters populate all three frozen
    # relation strata for both charged and neutral alternating families.
    pt = np.linspace(20.0, 4.0, tokens, dtype=np.float32)
    eta = np.asarray(
        [(index % 5) * 0.012 + (index // 5) * 0.24 for index in range(tokens)],
        dtype=np.float32,
    )
    phi = np.asarray(
        [(index % 5) * 0.011 + (index // 5) * 0.27 for index in range(tokens)],
        dtype=np.float32,
    )
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + np.float32(0.25))
    row = np.stack((px, py, pz, energy), axis=0)
    return np.ascontiguousarray(np.repeat(row[None], batch, axis=0), dtype=np.float32)


def _batch(indices: Sequence[int], *, tokens: int = 20) -> dict[str, np.ndarray]:
    rows = tuple(int(index) for index in indices)
    generator = np.random.default_rng(sum(rows) + 991)
    return {
        "features": generator.normal(size=(len(rows), 21, tokens)).astype(np.float32),
        "vectors": _vectors(len(rows), tokens),
        "mask": np.ones((len(rows), 1, tokens), dtype=np.bool_),
        "visible_indices": np.tile(np.arange(tokens, dtype=np.int64), (len(rows), 1)),
        "family_codes": np.tile(
            np.asarray([0, 1] * (tokens // 2), dtype=np.int8), (len(rows), 1),
        ),
        "labels": np.asarray([index % 15 for index in rows], dtype=np.int64),
        "identity_digests": np.stack([_identity_digest(index) for index in rows]),
    }


def _smoke_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_vectors(rows: int, tokens: int, *, offset: float = 0.0) -> np.ndarray:
    pt = np.arange(1, tokens + 1, dtype=np.float32)
    phi = offset + np.arange(tokens, dtype=np.float32) * np.float32(0.3)
    eta = np.arange(tokens, dtype=np.float32) * np.float32(0.05)
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + np.float32(0.1))
    one = np.stack((px, py, pz, energy), axis=0)
    return np.ascontiguousarray(np.repeat(one[None], rows, axis=0))


def _target_hidden(rows: int, tokens: int, *, offset: int = 0) -> np.ndarray:
    value = np.zeros((rows, tokens, 128), dtype=np.float32)
    for row in range(rows):
        for token in range(tokens):
            value[row, token, (offset + row + token) % 128] = np.float32(1.0)
    return value


@lru_cache(maxsize=1)
def run_local_target_lifecycle_probe() -> dict[str, Any]:
    """Build, RAM-load, identity-join, and clean ordinary plus TOFF targets.

    This is the real target-generation/recovery implementation over two rows
    per bank.  It performs exactly one teacher forward per bank and remains a
    nonauthorizing local fixture.
    """

    import torch
    from .hcwdl_representation_campaign_artifacts import (
        build_cache_miniature_bank_evidence,
    )
    from .hcwdl_representation_contracts import RESOURCE_PROFILE_CONTRACT
    from .hcwdl_representation_kernels import generate_spectral_resources
    from .hcwdl_representation_resources import build_storage_estimate, resource_table
    from .hcwdl_representation_target_recovery import (
        authorize_miniature_target_cleanup, complete_target_cleanup,
        validate_cleanup_completion,
    )
    from .hcwdl_representation_target_runtime import (
        TargetForwardBatch, build_target_generation_from_teacher,
    )
    from .hcwdl_representation_targets import (
        RepresentationTargetBank, begin_target_generation,
        build_logical_target_bank, build_miniature_target_consumer_row,
        build_target_consumer_registry, build_target_forward_spec,
        identity_order_sha256, identity_set_sha256,
        target_population_rows_sha256,
    )

    token_resources = generate_spectral_resources("token")
    relation_resources = generate_spectral_resources("relation")
    kernel_hashes = {
        block.resource_name: canonical_sha256(block.logical_hashes)
        for resources in (token_resources, relation_resources)
        for block in resources.blocks
    }
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    previous_torch = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="hcwdl-rkd-target-lifecycle-") as directory:
            root = Path(directory)
            for bank_id, bank_kind in (("RSET_D95c", "ordinary"), ("TOFF", "toff")):
                parents = {
                    name: _smoke_sha(f"{bank_id}:{name}")
                    for name in (
                        "source", "split", "train_row_selection", "graph",
                        "assignment", "repair", "architecture", "parent_recipe",
                        "representation_recipe", "kernel_resources", "parent_import",
                        "parent_loss_attestation",
                    )
                }
                checkpoint_bytes = _smoke_sha(f"{bank_id}:bytes")
                checkpoint_logical = _smoke_sha(f"{bank_id}:logical")
                tap_sha256 = _smoke_sha(f"{bank_id}:tap")
                if bank_id == "TOFF":
                    teacher = {
                        "source_kind": "imported_checkpoint",
                        "node_id": bank_id,
                        "domain": "toff",
                        "track": "shared",
                        "selected_report_sha256": _smoke_sha(f"{bank_id}:report"),
                        "checkpoint_byte_sha256": checkpoint_bytes,
                        "checkpoint_logical_sha256": checkpoint_logical,
                        "tap_sha256": tap_sha256,
                        "installed_weaver_signature_sha256": _smoke_sha(
                            f"{bank_id}:weaver"
                        ),
                    }
                else:
                    teacher = {
                        "source_kind": "campaign_execution",
                        "node_id": bank_id,
                        "domain": "d95",
                        "track": "cold",
                        "registered_execution_id": _smoke_sha(
                            f"{bank_id}:registered-execution"
                        ),
                        "tap_sha256": tap_sha256,
                    }
                logical = build_logical_target_bank(
                    bank_id=bank_id, teacher=teacher, parents=parents,
                )
                forward_teacher = {
                    "source_kind": teacher["source_kind"],
                    "architecture_sha256": parents["architecture"],
                    "tap_sha256": tap_sha256,
                    "kernel_resources_sha256": parents["kernel_resources"],
                    "kernel_array_logical_hashes": kernel_hashes,
                }
                if bank_id == "TOFF":
                    forward_teacher.update({
                        "checkpoint_byte_sha256": checkpoint_bytes,
                        "checkpoint_logical_sha256": checkpoint_logical,
                        "model_config_sha256": _smoke_sha("model-config"),
                    })
                else:
                    forward_teacher["registered_execution_id"] = teacher[
                        "registered_execution_id"
                    ]
                consumer = build_miniature_target_consumer_row(
                    logical, campaign_sha256=_smoke_sha("campaign"),
                    recipe_sha256=_smoke_sha("recipe"), bounded_row_limit=2,
                )
                registry = build_target_consumer_registry(
                    logical, purpose="miniature", consumers=[consumer],
                    generation_parent_sha256=_smoke_sha("generation-parent"),
                )
                forward = build_target_forward_spec(
                    parents={"logical_bank": logical["content_hash"]},
                    payload={
                        "teacher": forward_teacher,
                        "producer": {
                            "source_commit": "b" * 40,
                            "source_snapshot_sha256": _smoke_sha("snapshot"),
                            "packages": {
                                name: "local-smoke" for name in (
                                    "python", "torch", "cuda", "cudnn", "numpy",
                                    "awkward", "uproot", "weaver",
                                )
                            },
                        },
                        "device": {
                            "request": "gpu:gh200:1", "architecture": "Hopper",
                            "model": "GH200", "compute_capability": "9.0",
                            "driver": "local", "runtime": "local",
                        },
                        "precision": {
                            "parameters": "float32", "inputs": "float32",
                            "activations": "float32", "autocast": False,
                            "matmul_tf32": False, "cudnn_tf32": False,
                            "reduced_precision_fp32_reduction": False,
                            "output_order": "C",
                        },
                        "determinism": {
                            "deterministic_algorithms": True,
                            "cudnn_deterministic": True,
                            "cudnn_benchmark": False,
                            "cublas_workspace_config": ":4096:8",
                            "rng_states_sha256": _smoke_sha("rng"),
                        },
                        "batching": {
                            "batch_size": 256,
                            "order": "source_file_id_then_source_entry_v1",
                            "cross_source_batches": False,
                            "final_short_batch_per_source": True,
                            "padding": False, "row_duplication": False,
                        },
                        "implementation": {
                            **{
                                name: _smoke_sha(name) for name in (
                                    "input_decoding_sha256", "feature_layout_sha256",
                                    "trimmer_sha256", "family_code_sha256",
                                    "surface_capture_sha256", "sketch_arithmetic_sha256",
                                )
                            },
                            "teacher_input_fields": ["features", "mask", "points", "v"],
                        },
                        "source_partitions": ["p0"],
                    },
                )
                identities = np.stack([_identity_digest(101), _identity_digest(102)])
                labels = np.asarray([0, 1], dtype=np.uint8)
                source_ids = np.full(2, 7, dtype=np.uint32)
                entries = np.asarray([0, 1], dtype=np.uint64)
                charge = np.asarray([[1, 0, -1, 0, 0, 1]] * 2, dtype=np.float32)
                flags = np.zeros((2, 6, 5), dtype=np.float32)
                flags[:, 0, 0] = 1; flags[:, 1, 3] = 1
                batch = TargetForwardBatch(
                    source_partition="p0", source_file_id=source_ids,
                    source_entry=entries, identity_digest=identities, label=labels,
                    teacher_inputs={
                        "features": np.ones((2, 1), dtype=np.float32),
                        "mask": np.ones((2, 1), dtype=np.bool_),
                        "points": np.zeros((2, 2, 1), dtype=np.float32),
                        "v": np.ones((2, 4, 1), dtype=np.float32),
                    },
                    companion_hlt_charge=charge if bank_kind == "toff" else None,
                    companion_hlt_pid_flags=flags if bank_kind == "toff" else None,
                    companion_hlt_visible_mask=(
                        np.ones((2, 6), dtype=np.bool_)
                        if bank_kind == "toff" else None
                    ),
                )
                storage = build_storage_estimate(
                    train_rows=2, validation_rows=0, final_rows=0,
                    parent_import_sha256=parents["parent_import"],
                    prediction_finalists=0,
                )
                requests = resource_table(mode="smoke")
                profile = with_content_hash({
                    "contract": RESOURCE_PROFILE_CONTRACT, "schema_version": 1,
                    "requests": requests,
                    "measurements": {
                        name: {"peak_rss_bytes": 1024.0, "elapsed_seconds": 1.0}
                        for name in requests
                    },
                    "array_concurrency_limits": {},
                })
                context = begin_target_generation(
                    root / bank_id, logical_bank=logical,
                    consumer_registry=registry, forward_spec=forward,
                    partitions={"p0": {"rows": 2, "source_file_id": 7}},
                    expected_class_counts=[1, 1, *([0] * 13)],
                    expected_identity_order_sha256=identity_order_sha256(identities),
                    expected_identity_set_sha256=identity_set_sha256(identities),
                    expected_population_rows_sha256=target_population_rows_sha256(
                        source_file_id=source_ids, source_entry=entries,
                        identity_digest=identities, label=labels,
                    ),
                    build_owner={"local_smoke": bank_id},
                    target_storage_cap_bytes=10_000_000,
                    container_overhead_bytes=0,
                    staging_recovery_reserve_bytes=1_000_000,
                    quarantine_reserve_bytes=1_000_000,
                    filesystem_headroom_bytes=1_000_000,
                    peak_runtime_bytes=1_048_576,
                    slurm_mem_per_node_bytes=64 * 1024**3,
                    filesystem_available_bytes=20_000_000,
                    storage_estimate=storage, resource_profile=profile,
                )

                def teacher_forward(model_inputs):
                    count = len(model_inputs.arrays["features"])
                    common = {
                        "logits": np.repeat(np.arange(15, dtype=np.float32)[None], count, axis=0),
                    }
                    if bank_kind == "ordinary":
                        return {
                            **common, "particle_block_2": _target_hidden(count, 4),
                            "jet_penultimate": np.repeat(np.arange(128, dtype=np.float32)[None], count, axis=0),
                            "particle_mask": np.ones((count, 4), dtype=np.bool_),
                            "vectors": _target_vectors(count, 4),
                            "visible_indices": np.repeat(np.arange(4, dtype=np.int64)[None], count, axis=0),
                            "family_codes": np.zeros((count, 4), dtype=np.int8),
                        }
                    return {
                        **common,
                        "charged_particle_block_2": _target_hidden(count, 4),
                        "neutral_particle_block_2": _target_hidden(count, 1, offset=32),
                        "offline_jet_penultimate": np.repeat(np.arange(128, dtype=np.float32)[None], count, axis=0),
                        "charged_mask": np.ones((count, 4), dtype=np.bool_),
                        "neutral_mask": np.ones((count, 1), dtype=np.bool_),
                        "charged_vectors": _target_vectors(count, 4),
                        "neutral_vectors": _target_vectors(count, 1, offset=0.1),
                        "charged_visible_indices": np.repeat(np.arange(4, dtype=np.int64)[None], count, axis=0),
                        "neutral_visible_indices": np.zeros((count, 1), dtype=np.int64),
                    }

                result = build_target_generation_from_teacher(
                    context,
                    partition_batches={"p0": lambda batch=batch: iter((batch,))},
                    teacher_forward=teacher_forward,
                    token_resources=token_resources,
                    relation_resources=relation_resources,
                    teacher_model=torch.nn.Linear(1, 1, bias=False).float().eval(),
                    runtime_environment={
                        "producer": forward["payload"]["producer"],
                        "device": {**forward["payload"]["device"], "gpu_uuid": "GPU-local"},
                        "precision": forward["payload"]["precision"],
                        "determinism": forward["payload"]["determinism"],
                    },
                )
                bank = RepresentationTargetBank.load(
                    context.bank_root, context.generation_id, strategy="RREL",
                )
                joined = bank.join(identities[::-1].copy())
                ram_bytes = sum(int(np.asarray(value).nbytes) for value in bank.arrays.values())
                evidence = build_cache_miniature_bank_evidence(
                    bank_kind=bank_kind,
                    logical_bank_sha256=logical["content_hash"],
                    generation_id=context.generation_id,
                    generation_manifest_sha256=result.manifest["content_hash"],
                    rows=2, bounded_row_limit=2, identity_join_rows=2,
                    loaded_array_logical_sha256=result.manifest["payload"]["logical_target_sha256"],
                    ram_bytes=ram_bytes,
                )
                del joined, bank
                gc.collect()
                cleanup_root = root / "cleanup"
                authorization = authorize_miniature_target_cleanup(
                    context.bank_root, cleanup_root,
                    generation_id=context.generation_id,
                    cache_bank_evidence=evidence,
                )
                completion = complete_target_cleanup(
                    context.bank_root, cleanup_root,
                    generation_id=context.generation_id,
                )
                validate_cleanup_completion(
                    completion, authorization=authorization,
                    bank_root=context.bank_root,
                )
                rows.append({
                    "bank_kind": bank_kind,
                    "teacher_forward_calls": result.teacher_forward_calls,
                    "rows": result.manifest["payload"]["rows"],
                    "ram_joined": True,
                    "cleanup_completed": completion["payload"]["all_authorized_paths_absent"],
                    "manifest_sha256": result.manifest["content_hash"],
                })
    finally:
        torch.use_deterministic_algorithms(previous_torch["algorithms"])
        torch.backends.cudnn.deterministic = previous_torch["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous_torch["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous_torch["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = previous_torch["cudnn_tf32"]
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
    if [row["bank_kind"] for row in rows] != ["ordinary", "toff"] or any(
        row["teacher_forward_calls"] != 1 or row["rows"] != 2
        or row["cleanup_completed"] is not True for row in rows
    ):
        raise RuntimeError("local target lifecycle did not exercise exact bank semantics")
    return with_content_hash({
        "contract": "HCWDL_REPRESENTATION_LOCAL_TARGET_LIFECYCLE/v1",
        "schema_version": 1, "banks": rows,
        "ordinary_and_toff_one_forward": True,
        "ram_identity_joined": True, "cleanup_completed": True,
        "scientific_authorization": False, "final_role_accessed": False,
    })


class _SyntheticTargetBank:
    """Small identity-safe detached bank with every ordinary/TOFF component."""

    def __init__(self, indices: Sequence[int], *, native_offline: bool) -> None:
        self.manifest = {
            "payload": {
                "logical_target_sha256": canonical_sha256({
                    "local_smoke": "TOFF" if native_offline else "ordinary",
                }),
            },
        }
        generator = np.random.default_rng(720 if native_offline else 610)
        self._rows: dict[bytes, dict[str, np.ndarray]] = {}
        for index in indices:
            row = {
                "logits": generator.normal(size=15).astype(np.float32),
                "jet_penultimate": generator.normal(size=128).astype(np.float32),
                "token_family_eligibility": np.ones(2 if native_offline else 1, np.uint8),
                "relation_eligibility": np.ones(
                    (2 if native_offline else 1, 3), dtype=np.uint8,
                ),
            }
            if native_offline:
                row.update({
                    "token_kernel_mean_charged": generator.normal(size=1024).astype(np.float32),
                    "token_kernel_mean_neutral": generator.normal(size=1024).astype(np.float32),
                    "relation_kernel_mean_charged": generator.normal(size=(3, 256)).astype(np.float32),
                    "relation_kernel_mean_neutral": generator.normal(size=(3, 256)).astype(np.float32),
                })
            else:
                row.update({
                    "token_kernel_mean": generator.normal(size=1024).astype(np.float32),
                    "relation_kernel_mean": generator.normal(size=(3, 256)).astype(np.float32),
                })
            self._rows[bytes(_identity_digest(int(index)))] = row

    def join(self, identities: np.ndarray) -> dict[str, np.ndarray]:
        keys = [bytes(row) for row in np.asarray(identities)]
        if len(keys) != len(set(keys)):
            raise ValueError("local scientific smoke repeats an identity")
        try:
            names = tuple(self._rows[keys[0]])
            return {
                name: np.ascontiguousarray(np.stack([self._rows[key][name] for key in keys]))
                for name in names
            }
        except KeyError as error:
            raise KeyError("local scientific smoke target join is incomplete") from error

    def shuffled_representation_join(self, identities: np.ndarray) -> dict[str, np.ndarray]:
        joined = self.join(np.asarray(identities)[::-1].copy())
        joined.pop("logits")
        return joined


def run_scientific_full_loss_probe(*, device: str = "cpu") -> dict[str, Any]:
    """Exercise every registered node/control at full strength and backward.

    This calls the same ``exercise_full_representation_loss`` entry point used
    by the genuine-worker miniature, while keeping the fixture synthetic and
    explicitly non-authorizing.  Torch is imported lazily so every CLI can run
    ``--help`` on hosts without the training environment.
    """

    import torch
    from torch import nn
    from hlt_classification.training.checkpoints import (
        capture_rng_state, restore_rng_state,
    )

    # The local probe is observational: construction of its synthetic
    # students must not advance the caller's training RNG.
    initial_rng = capture_rng_state()
    initial_cpu_rng = initial_rng["torch_cpu"].clone()

    from hlt_classification.models.hcwdl_representation import HCWDLRepresentationHeads

    from .hcwdl_representation_kernels import generate_spectral_resources
    from .hcwdl_representation_training import (
        exercise_full_representation_loss,
        initialize_representation_student,
        resolve_node_execution,
    )

    class TinyDeployable(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(21, 128)
            self.classifier = nn.Linear(128, 15)

        def no_weight_decay(self) -> set[str]:
            return set()

        def _surfaces(self, features, vectors, mask, visible_indices, family_codes):
            token = torch.tanh(self.embed(features.transpose(1, 2)))
            visible = mask.squeeze(1).bool()
            pooled = (
                (token * visible[..., None]).sum(1)
                / visible.sum(1, keepdim=True).clamp_min(1)
            )
            return SimpleNamespace(
                logits=self.classifier(pooled),
                particle_block_2=token,
                jet_penultimate=pooled,
                particle_mask=visible,
                vectors=vectors,
                visible_indices=visible_indices,
                family_codes=family_codes,
            )

        def forward(self, features, vectors, mask):
            batch_size, _, tokens = features.shape
            visible = torch.arange(tokens, device=features.device).repeat(batch_size, 1)
            family = torch.zeros(
                (batch_size, tokens), dtype=torch.int8, device=features.device,
            )
            return self._surfaces(features, vectors, mask, visible, family).logits

        def forward_hcwdl_surfaces(
            self, features, vectors, mask, visible_indices, family_codes,
        ):
            return self._surfaces(
                features, vectors, mask, visible_indices, family_codes,
            )

    class TinyStudent(nn.Module):
        def __init__(
            self, *, strategy, teacher_latent_domain, jet_only, deployable_model,
        ) -> None:
            super().__init__()
            self.deployable_model = deployable_model
            self.representation_heads = HCWDLRepresentationHeads(
                strategy=strategy,
                teacher_latent_domain=teacher_latent_domain,
                jet_only=jet_only,
            )

        def no_weight_decay(self) -> set[str]:
            return {
                f"representation_heads.{name}.weight"
                for name, _ in self.representation_heads.projection_items()
            }

        def forward(self, features, vectors, mask):
            return self.deployable_model(features, vectors, mask)

        def forward_hcwdl_surfaces(self, *args):
            return self.deployable_model.forward_hcwdl_surfaces(*args)

    batch = _batch((0, 1), tokens=20)
    ordinary = _SyntheticTargetBank(range(20), native_offline=False)
    native = _SyntheticTargetBank(range(20), native_offline=True)
    token_resources = generate_spectral_resources("token")
    relation_resources = generate_spectral_resources("relation")
    warm_hash = canonical_sha256({"local_smoke": "warm deployable"})
    cases: list[dict[str, Any]] = []
    try:
        for execution_id in (*sorted(NODE_REGISTRY), *sorted(CONTROL_REGISTRY)):
            execution = resolve_node_execution(execution_id)
            initialization = {}
            if execution.initialization == "warm":
                initialization = {
                    "warm_checkpoint": "synthetic://deployable",
                    "warm_checkpoint_sha256": warm_hash,
                    "warm_loader": lambda _path, _digest: TinyDeployable(),
                }
            model = initialize_representation_student(
                execution_id,
                replicate_seed=1337,
                deployable_factory=TinyDeployable,
                wrapper_factory=TinyStudent,
                **initialization,
            )
            target = native if execution.teacher_latent_domain == "native_offline" else ordinary
            shuffled = (
                target.shuffled_representation_join
                if execution.shuffled_representation_targets else None
            )
            result = exercise_full_representation_loss(
                model,
                execution_id=execution_id,
                batch=batch,
                target_bank=target,
                predecessor_bank=None,
                class_weights=np.ones(15, dtype=np.float32),
                token_resources=token_resources,
                relation_resources=relation_resources,
                device=device,
                shuffled_representation_joiner=shuffled,
            )
            scalar_fields = ("total_loss", "representation_loss")
            if not all(np.isfinite(float(result[name])) for name in scalar_fields):
                raise FloatingPointError("local full-loss probe produced a nonfinite loss")
            if float(result["representation_loss"]) <= 0:
                raise RuntimeError("local full-loss probe disconnected the auxiliary")
            if set(result["active_components"]) != set(execution.active_components):
                raise RuntimeError("local full-loss probe component registry differs")
            if any(
                not np.isfinite(float(value)) or float(value) <= 0
                for value in result["head_gradient_norms"].values()
            ):
                raise RuntimeError("local full-loss probe has an invalid head gradient")
            cases.append({
                **result,
                "teacher_latent_domain": execution.teacher_latent_domain,
                "track": execution.track,
                "rung": execution.rung,
                "control": execution_id in CONTROL_REGISTRY,
                "all_registered_components_forced": True,
            })
    finally:
        restore_rng_state(initial_rng)
    expected = set(NODE_REGISTRY) | set(CONTROL_REGISTRY)
    if {row["execution_id"] for row in cases} != expected:
        raise RuntimeError("local full-loss probe did not exercise every execution")
    payload = {
        "contract": SMOKE_PROBE_CONTRACT,
        "schema_version": 1,
        "fixture_sha256": canonical_sha256({
            "batch": {
                name: logical_array_sha256(f"local_smoke.batch.{name}", value)
                for name, value in sorted(batch.items())
            },
            "ordinary_target": ordinary.manifest,
            "native_offline_target": native.manifest,
        }),
        "execution_ids": sorted(expected),
        "primary_count": len(NODE_REGISTRY),
        "control_count": len(CONTROL_REGISTRY),
        "cases": cases,
        "all_losses_finite": True,
        "all_active_head_gradients_finite_nonzero": True,
        "caller_rng_restored": torch.equal(torch.get_rng_state(), initial_cpu_rng),
        "optimizer_or_scheduler_step_performed": False,
        "final_role_accessed": False,
        "scientific_authorization": False,
        "authorizes_tigris_or_pilot": False,
    }
    return with_content_hash(payload)


def measure_zero_coefficient_parity(
    *, device: str = "cpu", seed: int = 20260809,
) -> dict[str, Any]:
    """Measure the plan's zero-representation-coefficient invariant.

    This is intentionally an execution, not a caller-authored checklist.  It
    constructs the installed-Weaver HLT ParT and an HCWDL wrapper around an
    exact deep copy, replays the same normal-training forward RNG state, runs
    the frozen CE/two-teacher-KD base loss, backpropagates, and performs one
    AdamW update on both graphs.  Representation heads are present in the
    wrapper but are disconnected when ``rho_repr`` is zero.
    """

    import copy
    import random
    import torch

    from hlt_classification.models.hcwdl_representation import (
        HCWDLRepresentationStudent,
    )
    from hlt_classification.models.scouting_particle_transformer import (
        build_scouting_particle_transformer,
    )
    from hlt_classification.training.checkpoints import (
        capture_model_runtime_state, capture_rng_state, restore_rng_state,
    )
    from .hcwdl_parent_loss import hcwdl_base_loss
    from .training import LossConfiguration

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("zero-coefficient parity requested unavailable CUDA")

    def rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return bool(
            left["python_random"] == right["python_random"]
            and np.array_equal(left["numpy_random"][1], right["numpy_random"][1])
            and left["numpy_random"][:1] == right["numpy_random"][:1]
            and left["numpy_random"][2:] == right["numpy_random"][2:]
            and torch.equal(left["torch_cpu"], right["torch_cpu"])
            and len(left["torch_cuda"]) == len(right["torch_cuda"])
            and all(
                torch.equal(one, two)
                for one, two in zip(
                    left["torch_cuda"], right["torch_cuda"], strict=True,
                )
            )
        )

    def maximum(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left.detach().float() - right.detach().float()).abs().max().cpu())

    caller_rng = capture_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        base = build_scouting_particle_transformer().to(target).float()
        wrapper = HCWDLRepresentationStudent(
            strategy="RSET", teacher_latent_domain="ordinary",
            deployable_model=copy.deepcopy(base),
        ).to(target).float()
        base.train(); wrapper.train()
        base_runtime_before = capture_model_runtime_state(base)
        wrapper_runtime_before = capture_model_runtime_state(wrapper)
        normal_trimming = bool(
            base_runtime_before["trimmers"]
            and wrapper_runtime_before["trimmers"]
            and all(row["enabled"] for row in base_runtime_before["trimmers"])
            and all(row["enabled"] for row in wrapper_runtime_before["trimmers"])
        )

        generator = torch.Generator(device=target).manual_seed(seed + 1)
        features = torch.randn(3, 21, 24, generator=generator, device=target)
        vectors = torch.randn(3, 4, 24, generator=generator, device=target)
        vectors[:, 3] = vectors[:, :3].square().sum(1).add(1).sqrt()
        mask = torch.ones(3, 1, 24, dtype=torch.bool, device=target)
        mask[0, :, -3:] = False
        labels = torch.tensor([1, 7, 14], dtype=torch.long, device=target)
        hlt_teacher = torch.linspace(
            -1.2, 1.4, 45, device=target, dtype=torch.float32,
        ).reshape(3, 15)
        privileged_teacher = torch.cos(
            torch.arange(45, device=target, dtype=torch.float32),
        ).reshape(3, 15)
        class_weights = torch.ones(15, device=target, dtype=torch.float32)
        configuration = LossConfiguration.for_mixture(
            arm="HCWDL_ZERO_COEFFICIENT_PARITY",
            ce=0.25, hlt_kd=0.40, privileged_kd=0.35,
            hlt_temperature=1.0, privileged_temperature=2.0,
        )
        replay_rng = capture_rng_state()
        with torch.autocast(device_type=target.type, enabled=False):
            base_logits = base(features, vectors, mask).float()
        base_post_rng = capture_rng_state()
        base_runtime_after = capture_model_runtime_state(base)
        restore_rng_state(replay_rng)
        with torch.autocast(device_type=target.type, enabled=False):
            wrapper_logits = wrapper(features, vectors, mask).float()
        wrapper_post_rng = capture_rng_state()
        wrapper_runtime_after = capture_model_runtime_state(wrapper)

        base_parts = hcwdl_base_loss(
            base_logits, labels, class_weights=class_weights,
            configuration=configuration, hlt_teacher_logits=hlt_teacher,
            privileged_teacher_logits=privileged_teacher,
        )
        wrapper_parts = hcwdl_base_loss(
            wrapper_logits, labels, class_weights=class_weights,
            configuration=configuration, hlt_teacher_logits=hlt_teacher,
            privileged_teacher_logits=privileged_teacher,
        )
        base.zero_grad(set_to_none=True); wrapper.zero_grad(set_to_none=True)
        base_parts["total"].backward(); wrapper_parts["total"].backward()
        base_named = dict(base.named_parameters())
        wrapped_named = {
            name.removeprefix("deployable_model."): parameter
            for name, parameter in wrapper.named_parameters()
            if name.startswith("deployable_model.")
        }
        if set(base_named) != set(wrapped_named):
            raise RuntimeError("zero-coefficient shared parameter registry differs")
        gradient_differences = []
        for name in base_named:
            left_gradient = base_named[name].grad
            right_gradient = wrapped_named[name].grad
            if (left_gradient is None) != (right_gradient is None):
                raise RuntimeError(
                    "zero-coefficient shared gradient topology differs"
                )
            gradient_differences.append(
                0.0 if left_gradient is None
                else maximum(left_gradient, right_gradient)
            )
        shared_gradient_max = max(gradient_differences, default=0.0)
        heads_disconnected = all(
            parameter.grad is None
            for name, parameter in wrapper.named_parameters()
            if name.startswith("representation_heads.")
        )
        base_optimizer = torch.optim.AdamW(base.parameters(), lr=3.0e-4)
        wrapper_optimizer = torch.optim.AdamW(wrapper.parameters(), lr=3.0e-4)
        base_optimizer.step(); wrapper_optimizer.step()
        optimizer_parameter_max = max(
            maximum(base_named[name], wrapped_named[name]) for name in base_named
        )
        optimizer_state_max = 0.0
        for name in base_named:
            left_state = base_optimizer.state[base_named[name]]
            right_state = wrapper_optimizer.state[wrapped_named[name]]
            if set(left_state) != set(right_state):
                raise RuntimeError("zero-coefficient optimizer state registry differs")
            for state_name in left_state:
                left = left_state[state_name]
                right = right_state[state_name]
                if torch.is_tensor(left):
                    optimizer_state_max = max(optimizer_state_max, maximum(left, right))
                elif left != right:
                    raise RuntimeError("zero-coefficient optimizer scalar state differs")

        normalized_base_runtime = [
            {"enabled": row["enabled"], "counter": row["counter"]}
            for row in base_runtime_after["trimmers"]
        ]
        normalized_wrapper_runtime = [
            {"enabled": row["enabled"], "counter": row["counter"]}
            for row in wrapper_runtime_after["trimmers"]
        ]
        return {
            "logits_max_abs": maximum(base_logits, wrapper_logits),
            "base_loss_max_abs": maximum(
                base_parts["total"], wrapper_parts["total"],
            ),
            "shared_gradient_max_abs": shared_gradient_max,
            "optimizer_state_max_abs": max(
                optimizer_parameter_max, optimizer_state_max,
            ),
            "ce_equal": maximum(base_parts["ce"], wrapper_parts["ce"]) <= 1.0e-7,
            "hlt_kd_equal": maximum(
                base_parts["hlt_kd"], wrapper_parts["hlt_kd"],
            ) <= 1.0e-7,
            "privileged_kd_equal": maximum(
                base_parts["privileged_kd"], wrapper_parts["privileged_kd"],
            ) <= 1.0e-7,
            "shared_parameter_names_equal": set(base_named) == set(wrapped_named),
            "representation_heads_have_no_logit_path": heads_disconnected,
            "rng_state_equal": rng_equal(base_post_rng, wrapper_post_rng),
            "trimmer_progression_equal": (
                normalized_base_runtime == normalized_wrapper_runtime
            ),
            "optimizer_update_equal": optimizer_parameter_max <= 1.0e-6,
            "installed_weaver": True,
            "normal_training_trimming": normal_trimming,
        }
    finally:
        restore_rng_state(caller_rng)


def validate_local_task_kind(kind: str) -> None:
    """Prevent local fixture dispatch from crossing the final boundary."""

    if kind in FINAL_ROLE_KINDS:
        raise PermissionError("local HCWDL-RKD smoke cannot dispatch a final-role task")


__all__ = [
    "FINAL_ROLE_KINDS", "measure_zero_coefficient_parity",
    "run_scientific_full_loss_probe", "validate_local_task_kind",
]
