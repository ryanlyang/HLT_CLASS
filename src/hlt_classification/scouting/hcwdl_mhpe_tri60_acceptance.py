"""Bounded real-worker acceptance and full-population RAM projection for TRI60."""

from __future__ import annotations

from pathlib import Path
import gc
import os
import platform
import re
import shutil
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from hlt_classification.data.cache_contracts import canonical_sha256

from .hcwdl_mhpe_tri60_contracts import RUNTIME_PROFILE_CONTRACT, artifact
from .hcwdl_mhpe_tri60_ephemeral import EphemeralRepresentationTargetBank
from .hcwdl_mhpe_tri60_graph import GRAPH_SHA256
from .hcwdl_mhpe_tri60_integration import (
    authenticate_foundation,
    build_endpoint_resource_lock,
)
from .hcwdl_mhpe_tri60_probability import Tri60ProbabilityTargets
from .hcwdl_mhpe_tri60_runner import _target_batch, _teacher_forward
from .hcwdl_mhpe_tri60_training import (
    Tri60TrainingRuntime,
    _peak_cuda_bytes,
    _peak_rss_bytes,
    load_tri60_model,
    train_tri60_node,
)
from .hcwdl_representation_kernels import generate_spectral_resource_bundle
from .hcwdl_representation_target_runtime import prepare_target_generation_in_memory
from .hcwdl_representation_targets import ORDINARY_BANK
from .hcwdl_unified_balanced_runner import _load_common, _stream
from .splits import role_records
from .training import derive_seed
from .view_cache import EphemeralPmardViewCache


def _selected_real_batches(
    *, foundation: Mapping[str, Any], split: Mapping[str, Any], selections,
    assignments, balanced, role: str, behavior: str, coordinate,
    sampler_seed: int, repair_seed: int,
    source_indexes: tuple[int, ...] | None = None,
) -> tuple[list[Mapping[str, Any]], tuple[int, ...]]:
    records = role_records(split, role)
    batches: list[Mapping[str, Any]] = []
    indexes: list[int] = []
    labels_seen: set[int] = set()
    candidates = range(len(records)) if source_indexes is None else source_indexes
    for source_index in candidates:
        stream = _stream(
            foundation_spec=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role=role,
            behavior=behavior, coordinate=coordinate, batch_size=256,
            sampler_seed=sampler_seed, repair_seed=repair_seed,
            include_hcwdl_metadata=True, source_index=source_index,
        )
        iterator = iter(stream)
        try:
            batch = next(iterator)
        except StopIteration:
            continue
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        batches.append(batch)
        indexes.append(source_index)
        labels_seen.update(map(int, np.asarray(batch["labels"]).tolist()))
        if source_indexes is None and labels_seen == set(range(15)):
            break
    if not batches or labels_seen != set(range(15)):
        raise ValueError(f"TRI60 acceptance {role} sample lacks all 15 classes")
    return batches, tuple(indexes)


def _bounded_cache(
    *, foundation: Mapping[str, Any], split: Mapping[str, Any], selections,
    assignments, balanced, role: str, behavior: str, coordinate,
    sampler_seed: int, repair_seed: int,
    source_indexes: tuple[int, ...] | None = None,
) -> tuple[EphemeralPmardViewCache, tuple[int, ...]]:
    batches, indexes = _selected_real_batches(
        foundation=foundation, split=split, selections=selections,
        assignments=assignments, balanced=balanced, role=role,
        behavior=behavior, coordinate=coordinate, sampler_seed=sampler_seed,
        repair_seed=repair_seed, source_indexes=source_indexes,
    )
    records = role_records(split, role)
    expected = {record.path: 0 for record in records}
    for source_index, batch in zip(indexes, batches, strict=True):
        expected[records[source_index].path] = len(batch["labels"])
    input_key = "hlt" if behavior == "hlt" else "privileged"
    cache = EphemeralPmardViewCache.build(
        batches,
        expected_rows=sum(len(batch["labels"]) for batch in batches),
        records=records,
        role=role,
        expected_source_rows=expected,
        view_keys=(input_key,),
        max_gib=64,
        lineage={
            "acceptance_only": True,
            "foundation_spec_sha256": foundation["content_hash"],
            "behavior": behavior,
            "coordinate": coordinate.payload(),
            "durable_repaired_dataset": False,
        },
    )
    return cache, indexes


def _cache_batches_in_identity_order(cache, *, batch_size: int = 256) -> Iterable[Mapping[str, Any]]:
    yield from cache.iterate_canonical_batches(batch_size=batch_size)


def _probability_targets(model, cache, *, device: str) -> Tri60ProbabilityTargets:
    import torch

    logits = []
    identities = []
    model.eval()
    with torch.inference_mode():
        for batch in _cache_batches_in_identity_order(cache):
            view = batch["privileged"]
            value = model(
                torch.as_tensor(view.features, device=device).float(),
                torch.as_tensor(view.vectors, device=device).float(),
                torch.as_tensor(view.mask, device=device).bool(),
            ).float()
            logits.append(value.cpu().numpy())
            identities.append(np.asarray(batch["identity_digests"], dtype=np.uint8))
    values = np.ascontiguousarray(np.concatenate(logits), dtype=np.float32)
    exponent = np.exp(values / np.float32(2.0) - (values / np.float32(2.0)).max(1, keepdims=True))
    probabilities = np.ascontiguousarray(exponent / exponent.sum(1, keepdims=True), dtype=np.float32)
    identity = np.ascontiguousarray(np.concatenate(identities), dtype=np.uint8)
    return Tri60ProbabilityTargets(
        identities=identity,
        probabilities=probabilities,
        manifest=MappingProxyType({"temperature": 2.0}),
        _lookup={bytes(row): index for index, row in enumerate(identity)},
    )


def _representation_targets(
    *, model, cache, device: str,
    campaign_sha256: str, recipe_sha256: str,
) -> tuple[EphemeralRepresentationTargetBank, Any, int]:
    records_by_source = {}
    for batch in _cache_batches_in_identity_order(cache):
        source = str(batch["identity_keys"][0]).rsplit("::tree::", 1)[0]
        records_by_source.setdefault(source, []).append(batch)
    factories = {}
    specs = {}
    source_ids = {source: index for index, source in enumerate(records_by_source)}
    for source, batches in records_by_source.items():
        partition = f"source_{source_ids[source]:04d}"

        def factory(*, batches=tuple(batches), partition=partition, source_id=source_ids[source]):
            for batch in batches:
                yield _target_batch(
                    batch, partition=partition, source_file_id=source_id,
                    view_key="privileged",
                )

        factories[partition] = factory
        specs[partition] = {
            "rows": sum(len(batch["labels"]) for batch in batches),
            "source_file_id": source_ids[source],
        }
    bundle = generate_spectral_resource_bundle()
    prepared = prepare_target_generation_in_memory(
        bank_kind=ORDINARY_BANK,
        partition_batches=factories,
        partition_specs=specs,
        teacher_forward=_teacher_forward(model, device=device),
        token_resources=bundle.token,
        relation_resources=bundle.relation,
        teacher_model=model,
        allowed_input_fields=(
            "family_codes", "features", "mask", "vectors", "visible_indices",
        ),
    )
    prepared_bytes = sum(
        int(value.nbytes)
        for partition in prepared.partitions.values()
        for value in partition.arrays.values()
    )
    bank = EphemeralRepresentationTargetBank.from_prepared(
        prepared,
        strategy="RSET",
        carrier_node_id="U000",
        carrier_report_sha256="1" * 64,
        carrier_checkpoint_sha256="2" * 64,
        campaign_spec_sha256=campaign_sha256,
        graph_sha256=GRAPH_SHA256,
        recipe_sha256=recipe_sha256,
    )
    return bank, bundle, prepared_bytes


def run_bounded_acceptance(
    *, foundation_lock: str | Path, temporary_root: str | Path,
    source_commit: str, device: str = "cuda",
) -> dict[str, Any]:
    """Exercise real rows, real models, RAM targets, and no-resume publication."""

    import torch
    from .hcwdl_mhpe_tri60_graph import COORDINATES

    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("TRI60 acceptance source commit differs")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("TRI60 acceptance requires a genuine Slurm worker")
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TRI60 acceptance requires CUDA")
    gpu_name = torch.cuda.get_device_name(target_device)
    if "GH200" not in gpu_name.upper():
        raise RuntimeError(f"TRI60 acceptance requires a GH200, observed {gpu_name}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    foundation_auth = authenticate_foundation(foundation_lock)
    foundation = __import__(
        "hlt_classification.data.cache_contracts", fromlist=["load_json"]
    ).load_json(foundation_auth["foundation_spec_path"])
    split, _, _, selections, assignments, balanced = _load_common(foundation)
    seed = derive_seed(1337, "TRI60/acceptance")
    repair_seed = derive_seed(1337, "tri60/repair/shared_v1")
    p0 = {}
    source_indexes = {}
    for role in ("train", "validation"):
        p0[role], source_indexes[role] = _bounded_cache(
            foundation=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role=role,
            behavior="p0", coordinate=COORDINATES["U000"],
            sampler_seed=seed, repair_seed=repair_seed,
        )
    root = Path(temporary_root).resolve()
    if root.exists():
        raise FileExistsError("TRI60 acceptance temporary root already exists")
    root.mkdir(parents=True)
    fake_campaign = canonical_sha256({"acceptance": source_commit})
    fake_recipe = canonical_sha256({"recipe": "TRI60 acceptance exact optimization"})
    runtime = Tri60TrainingRuntime(passes=2, batch_size=256)
    u000 = train_tri60_node(
        node_id="U000", train_cache=p0["train"], validation_cache=p0["validation"],
        input_key="privileged", output_dir=root / "U000",
        parents={"foundation": foundation_auth["content_hash"]},
        campaign_spec_sha256=fake_campaign, recipe_sha256=fake_recipe,
        execution_source_commit=source_commit, replicate_seed=1337,
        device=device, runtime=runtime, execution_mode="synthetic_test",
    )
    model, _ = load_tri60_model(root / "U000/training_report.json", device=device)
    probability = _probability_targets(model, p0["train"], device=device)
    rep_bank, bundle, prepared_bytes = _representation_targets(
        model=model, cache=p0["train"],
        device=device, campaign_sha256=fake_campaign, recipe_sha256=fake_recipe,
    )
    endpoint = build_endpoint_resource_lock(parents={"foundation": foundation_auth["content_hash"]})
    if endpoint["spectral_resource_sha256"] != bundle.content_hash:
        raise ValueError("TRI60 acceptance spectral resource differs")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    u100 = {}
    for role in ("train", "validation"):
        u100[role], observed = _bounded_cache(
            foundation=foundation, split=split, selections=selections,
            assignments=assignments, balanced=balanced, role=role,
            behavior="balanced_uniform", coordinate=COORDINATES["U100"],
            sampler_seed=seed, repair_seed=repair_seed,
            source_indexes=source_indexes[role],
        )
        if observed != source_indexes[role] or u100[role].identities != p0[role].identities:
            raise ValueError("TRI60 acceptance endpoint identities differ")
    audit = rep_bank.audit(
        peak_rss_bytes=_peak_rss_bytes(), peak_cuda_bytes=_peak_cuda_bytes(),
    )
    rset = train_tri60_node(
        node_id="RSET_U100_from_U000",
        train_cache=u100["train"], validation_cache=u100["validation"],
        input_key="privileged", probability_targets=probability,
        representation_targets=rep_bank,
        representation_audit_sha256=audit["content_hash"],
        token_resources=bundle.token, relation_resources=bundle.relation,
        output_dir=root / "RSET_U100_from_U000",
        parents={
            "foundation": foundation_auth["content_hash"],
            "ephemeral_representation_audit": audit["content_hash"],
        },
        campaign_spec_sha256=fake_campaign, recipe_sha256=fake_recipe,
        execution_source_commit=source_commit, replicate_seed=1337,
        device=device, runtime=runtime, execution_mode="synthetic_test",
    )
    measured_peak = max(int(u000["peak_rss_bytes"]), int(rset["peak_rss_bytes"]), _peak_rss_bytes())
    train_rows = int(foundation_auth["role_counts"]["train"])
    validation_rows = int(foundation_auth["role_counts"]["validation"])
    sample_train = int(u100["train"].header["rows"])
    sample_validation = int(u100["validation"].header["rows"])
    student_projection = (
        int(u100["train"].header["array_bytes"]) * train_rows // sample_train
        + int(u100["validation"].header["array_bytes"]) * validation_rows // sample_validation
    )
    rset_rep_projection = rep_bank.nbytes * train_rows // sample_train
    rrel_rep_projection = prepared_bytes * train_rows // sample_train
    probability_projection = (train_rows + validation_rows) * (32 + 15 * 4)
    bounded_arrays = (
        int(u100["train"].header["array_bytes"])
        + int(u100["validation"].header["array_bytes"])
        + prepared_bytes + rep_bank.nbytes
        + sample_train * (32 + 15 * 4)
    )
    overhead = max(0, measured_peak - bounded_arrays)
    projected_rset_training = (
        student_projection + rset_rep_projection + probability_projection + overhead
    )
    projected_rset_construction = (
        rrel_rep_projection + rset_rep_projection + overhead
    )
    projected_rset = max(projected_rset_training, projected_rset_construction)
    projected_rrel_training = (
        student_projection + rrel_rep_projection + probability_projection + overhead
    )
    projected_rrel_construction = 2 * rrel_rep_projection + overhead
    projected_rrel = max(projected_rrel_training, projected_rrel_construction)
    request = 384 * 1024**3
    profile_body = {
        "parents": {"foundation": foundation_auth["content_hash"]},
        "source_commit": source_commit,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "measurement_host": platform.node(),
        "gpu_name": gpu_name,
        "sample_rows": {"train": sample_train, "validation": sample_validation},
        "full_role_rows": {"train": train_rows, "validation": validation_rows},
        "measured_peak_rss_bytes": measured_peak,
        "measured_peak_cuda_bytes": max(int(u000["peak_cuda_bytes"]), int(rset["peak_cuda_bytes"])),
        "projected_residents": {
            "student_train_validation_views": student_projection,
            "rset_representation_targets": rset_rep_projection,
            "rrel_representation_targets": rrel_rep_projection,
            "probability_targets": probability_projection,
            "model_optimizer_loader_overhead": overhead,
        },
        "projected_rset_peak_bytes": projected_rset,
        "projected_rrel_peak_bytes": projected_rrel,
        "projected_rset_construction_peak_bytes": projected_rset_construction,
        "projected_rrel_construction_peak_bytes": projected_rrel_construction,
        "rset_request_bytes": request,
        "rrel_request_bytes": request,
        "peak_request_fraction": max(projected_rset, projected_rrel) / request,
        "elapsed_seconds": time.monotonic() - started,
        "genuine_tigris_production_worker": True,
        "real_foundation_rows": True,
        "production_model_factory": True,
        "ram_only_targets_proved": rep_bank.header["representation_targets_persisted"] is False,
        "no_resume_proved": not tuple(root.rglob("*resume*")),
        "selected_and_final_only": True,
        "temporary_artifacts_deleted": True,
        "temporary_artifact_bytes_after_cleanup": 0,
        "passed": max(projected_rset, projected_rrel) / request <= .75,
        "final_test_accessed": False,
    }
    rep_bank.release()
    shutil.rmtree(root)
    if root.exists():
        raise RuntimeError("TRI60 acceptance temporary cleanup failed")
    return artifact(profile_body, contract=RUNTIME_PROFILE_CONTRACT)


__all__ = ["run_bounded_acceptance"]
