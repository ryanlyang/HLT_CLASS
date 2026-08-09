from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    load_json,
    load_npz_arrays,
    with_content_hash,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    RESOURCE_PROFILE_CONTRACT,
    TRAINING_REPORT_CONTRACT,
)
from hlt_classification.scouting.hcwdl_representation_graph import (
    ASCENT_GRAPH_SHA256,
    CONTROL_REGISTRY,
    NODE_REGISTRY,
)
from hlt_classification.scouting.hcwdl_representation_kernels import (
    generate_spectral_resources,
)
from hlt_classification.scouting.hcwdl_representation_losses import (
    build_native_offline_token_targets,
    build_ordinary_token_targets,
    build_teacher_relation_targets,
)
from hlt_classification.scouting.hcwdl_representation_training import (
    paired_rng_streams,
)
from hlt_classification.scouting.hcwdl_representation_target_recovery import (
    TargetRecoveryState,
    authorize_target_cleanup,
    build_target_recovery_plan,
    complete_target_cleanup,
    inspect_target_recovery_state,
    validate_cleanup_completion,
    validate_reconstructed_generation,
)
from hlt_classification.scouting.hcwdl_representation_target_runtime import (
    TeacherModelInputs,
    TargetForwardBatch,
    build_predecessor_logit_cache,
    build_target_generation_from_prepared,
    build_target_generation_from_teacher,
    prepare_target_generation_in_memory,
)
from hlt_classification.scouting.hcwdl_representation_targets import (
    ORDINARY_BANK,
    TOFF_BANK,
    RepresentationTargetBank,
    begin_target_generation,
    build_logical_target_bank,
    build_miniature_target_consumer_row,
    build_target_consumer_registry,
    build_target_consumer_row,
    build_target_forward_spec,
    expected_screen_consumer_nodes,
    identity_order_sha256,
    identity_set_sha256,
    target_array_schema,
    target_core_values_per_row,
    target_logical_bytes_per_row,
    target_metadata_bytes_per_row,
    target_population_rows_sha256,
    finalize_target_generation,
    stage_target_shard,
    validate_target_arrays,
    validate_target_generation,
    validate_target_consumer_registry,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    build_storage_estimate,
    resource_table,
)
from hlt_classification.scouting.hcwdl_representation_reporting import (
    build_confirmation_registry,
)


H = "a" * 64
CAMPAIGN = hashlib.sha256(b"campaign").hexdigest()
RECIPE = hashlib.sha256(b"recipe").hexdigest()
TARGET_PREFLIGHT = {
    "container_overhead_bytes": 0,
    "staging_recovery_reserve_bytes": 1_000_000,
    "quarantine_reserve_bytes": 1_000_000,
    "filesystem_headroom_bytes": 1_000_000,
    "peak_runtime_bytes": 1_048_576,
    "slurm_mem_per_node_bytes": 64 * 1024**3,
    "filesystem_available_bytes": 20_000_000,
}


@pytest.fixture(autouse=True)
def _deterministic_target_runtime(monkeypatch):
    previous = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.use_deterministic_algorithms(previous["algorithms"])
    torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]


def _inspectable_teacher_model():
    return torch.nn.Linear(1, 1, bias=False).float().eval()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _teacher(bank_id: str) -> dict[str, str]:
    if bank_id == "TOFF":
        node, domain, track = "TOFF", "toff", "shared"
    elif bank_id == "D100":
        node, domain, track = "D100", "d100", "shared"
    else:
        node, domain = bank_id, bank_id[:-1].lower()
        track = "cold" if bank_id.endswith("c") else "warm"
    return {
        "node_id": node,
        "domain": domain,
        "track": track,
        "selected_report_sha256": _sha(f"{bank_id}:report"),
        "checkpoint_byte_sha256": _sha(f"{bank_id}:checkpoint-bytes"),
        "checkpoint_logical_sha256": _sha(f"{bank_id}:checkpoint-logical"),
        "tap_sha256": _sha(f"{bank_id}:tap"),
        "installed_weaver_signature_sha256": _sha(f"{bank_id}:weaver"),
    }


def _logical(bank_id: str = "D0c") -> dict:
    parents = {
        name: _sha(f"{bank_id}:{name}")
        for name in (
            "source", "split", "train_row_selection", "graph", "assignment", "repair",
            "architecture", "parent_recipe", "representation_recipe", "kernel_resources",
            "parent_import", "parent_loss_attestation",
        )
    }
    return build_logical_target_bank(
        bank_id=bank_id, teacher=_teacher(bank_id), parents=parents,
    )


def _consumer(
    logical: dict, node: str, seed: int = 1337, *, purpose: str = "screen",
) -> dict:
    return build_target_consumer_row(
        logical,
        purpose=purpose,
        campaign_sha256=CAMPAIGN,
        recipe_sha256=RECIPE,
        node_id=node,
        seed=seed,
    )


def _registry(
    logical: dict, *, purpose: str = "screen", subset=None, parent=H,
    seed: int = 1337, consumers=None,
) -> dict:
    bank_id = logical["payload"]["logical_bank_id"]
    if consumers is not None:
        rows = list(consumers)
    else:
        nodes = expected_screen_consumer_nodes(bank_id) if subset is None else subset
        if purpose == "confirmation":
            rows = [
                _consumer(logical, node, one_seed, purpose="confirmation")
                for node in nodes for one_seed in (11, 22, 33, 44, 55)
            ]
        else:
            rows = [_consumer(logical, node, seed=seed) for node in nodes]
    return build_target_consumer_registry(
        logical, purpose=purpose, consumers=rows, generation_parent_sha256=parent,
    )


def test_miniature_consumer_is_bounded_nonscientific_and_not_training_compatible() -> None:
    logical = _logical("D100")
    consumer = build_miniature_target_consumer_row(
        logical, campaign_sha256=CAMPAIGN, recipe_sha256=RECIPE,
        bounded_row_limit=37,
    )
    registry = build_target_consumer_registry(
        logical, purpose="miniature", consumers=[consumer],
        generation_parent_sha256=H,
    )
    validate_target_consumer_registry(registry, logical_bank=logical)
    payload = consumer["execution_identity_payload"]
    assert payload["scientific_authorization"] is False
    assert payload["training_consumer_authorized"] is False
    assert payload["bounded_row_limit"] == 37
    for purpose in ("screen", "confirmation"):
        with pytest.raises(ValueError):
            build_target_consumer_registry(
                logical, purpose=purpose, consumers=[consumer],
                generation_parent_sha256=H,
            )


def test_training_consumers_cannot_masquerade_as_miniature_and_bound_is_closed() -> None:
    logical = _logical("D100")
    training_consumer = _consumer(logical, "RSET_M5c")
    with pytest.raises(ValueError):
        build_target_consumer_registry(
            logical, purpose="miniature", consumers=[training_consumer],
            generation_parent_sha256=H,
        )
    with pytest.raises(ValueError, match="bounded row limit"):
        build_miniature_target_consumer_row(
            logical, campaign_sha256=CAMPAIGN, recipe_sha256=RECIPE,
            bounded_row_limit=4097,
        )


def test_target_builder_rechecks_miniature_bound_before_filesystem_mutation(tmp_path) -> None:
    logical = _logical("D100")
    consumer = build_miniature_target_consumer_row(
        logical, campaign_sha256=CAMPAIGN, recipe_sha256=RECIPE,
        bounded_row_limit=3,
    )
    with pytest.raises(ValueError, match="bounded row limit"):
        _context(
            tmp_path, bank_id="D100", purpose="miniature", consumers=[consumer],
        )
    assert not (tmp_path / "banks" / "D100").exists()


def _storage_estimate(logical: dict, *, rows: int) -> dict:
    return build_storage_estimate(
        train_rows=rows,
        validation_rows=0,
        final_rows=0,
        parent_import_sha256=logical["parents"]["parent_import"],
        prediction_finalists=0,
    )


def _resource_profile() -> dict:
    requests = resource_table(mode="smoke")
    return with_content_hash({
        "contract": RESOURCE_PROFILE_CONTRACT,
        "schema_version": 1,
        "requests": requests,
        "measurements": {
            name: {"peak_rss_bytes": 1024.0, "elapsed_seconds": 1.0}
            for name in requests
        },
        "array_concurrency_limits": {},
    })


def _forward_payload(logical: dict, partitions=("p0", "p1")) -> dict:
    teacher = logical["payload"]["teacher"]
    return {
        "teacher": {
            "checkpoint_byte_sha256": teacher["checkpoint_byte_sha256"],
            "checkpoint_logical_sha256": teacher["checkpoint_logical_sha256"],
            "model_config_sha256": _sha("model-config"),
            "architecture_sha256": logical["parents"]["architecture"],
            "tap_sha256": teacher["tap_sha256"],
            "kernel_resources_sha256": logical["parents"]["kernel_resources"],
            "kernel_array_logical_hashes": _kernel_hashes(),
        },
        "producer": {
            "source_commit": "b" * 40,
            "source_snapshot_sha256": _sha("snapshot"),
            "packages": {
                name: "test-version" for name in (
                    "python", "torch", "cuda", "cudnn", "numpy", "awkward", "uproot",
                    "weaver",
                )
            },
        },
        "device": {
            "request": "gpu:gh200:1", "architecture": "Hopper", "model": "GH200",
            "compute_capability": "9.0", "driver": "test", "runtime": "test",
        },
        "precision": {
            "parameters": "float32", "inputs": "float32", "activations": "float32",
            "autocast": False, "matmul_tf32": False, "cudnn_tf32": False,
            "reduced_precision_fp32_reduction": False, "output_order": "C",
        },
        "determinism": {
            "deterministic_algorithms": True, "cudnn_deterministic": True,
            "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8",
            "rng_states_sha256": _sha("rng"),
        },
        "batching": {
            "batch_size": 256, "order": "source_file_id_then_source_entry_v1",
            "cross_source_batches": False, "final_short_batch_per_source": True,
            "padding": False, "row_duplication": False,
        },
        "implementation": {
            **{
                name: _sha(name) for name in (
                    "input_decoding_sha256", "feature_layout_sha256", "trimmer_sha256",
                    "family_code_sha256", "surface_capture_sha256",
                    "sketch_arithmetic_sha256",
                )
            },
            "teacher_input_fields": ["features", "mask", "points", "v"],
        },
        "source_partitions": list(partitions),
    }


@lru_cache(maxsize=1)
def _kernel_hash_items() -> tuple[tuple[str, str], ...]:
    values = {}
    for kind in ("token", "relation"):
        for block in generate_spectral_resources(kind).blocks:
            values[block.resource_name] = canonical_sha256(block.logical_hashes)
    return tuple(sorted(values.items()))


def _kernel_hashes() -> dict[str, str]:
    return dict(_kernel_hash_items())


def _forward(logical: dict, partitions=("p0", "p1")) -> dict:
    return build_target_forward_spec(
        parents={"logical_bank": logical["content_hash"]},
        payload=_forward_payload(logical, partitions),
    )


def _identity(source_id: int, entry: int) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(_sha(f"{source_id}:{entry}")), dtype=np.uint8).copy()


def _batch(partition: str, source_id: int, entries: list[int], *, toff: bool = False):
    rows = len(entries)
    charge = np.repeat(
        np.asarray([[1.0, 0.0, -1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        rows,
        axis=0,
    )
    flags = np.zeros((rows, 6, 5), dtype=np.float32)
    flags[:, 0, 0] = 1.0  # direct charged
    flags[:, 1, 3] = 1.0  # direct neutral
    flags[:, 4, 0] = 1.0  # contradiction: charged PID, neutral charge
    flags[:, 5, 0:2] = 1.0  # malformed: multiple PID bits
    return TargetForwardBatch(
        source_partition=partition,
        source_file_id=np.full(rows, source_id, dtype=np.uint32),
        source_entry=np.asarray(entries, dtype=np.uint64),
        identity_digest=np.ascontiguousarray(np.stack([_identity(source_id, x) for x in entries])),
        label=np.asarray([(source_id + x) % 15 for x in entries], dtype=np.uint8),
        teacher_inputs={
            "features": np.full((rows, 1), source_id, dtype=np.float32),
            "mask": np.ones((rows, 1), dtype=np.bool_),
            "points": np.zeros((rows, 2, 1), dtype=np.float32),
            "v": np.ones((rows, 4, 1), dtype=np.float32),
        },
        companion_hlt_charge=charge if toff else None,
        companion_hlt_pid_flags=flags if toff else None,
        companion_hlt_visible_mask=(
            np.ones((rows, 6), dtype=np.bool_) if toff else None
        ),
    )


def _vectors(rows: int, tokens: int, *, offset: float = 0.0) -> np.ndarray:
    pt = np.asarray([1.0 + index for index in range(tokens)], np.float32)
    phi = np.asarray([offset + 0.3 * index for index in range(tokens)], np.float32)
    eta = np.asarray([0.05 * index for index in range(tokens)], np.float32)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + 0.1)
    one = np.stack((px, py, pz, energy), axis=0)
    return np.ascontiguousarray(np.repeat(one[None], rows, axis=0).astype(np.float32))


def _hidden(rows: int, tokens: int, *, offset: int = 0) -> np.ndarray:
    value = np.zeros((rows, tokens, 128), dtype=np.float32)
    for row in range(rows):
        for token in range(tokens):
            value[row, token, (offset + row + token) % 128] = np.float32(1.0)
            value[row, token, (offset + row + token + 17) % 128] = np.float32(0.25)
    return value


def _surface_rows(batch: TargetForwardBatch | TeacherModelInputs) -> int:
    if isinstance(batch, TargetForwardBatch):
        return batch.rows
    return int(len(batch.arrays["features"]))


def _ordinary_surface(batch: TargetForwardBatch | TeacherModelInputs):
    rows, tokens = _surface_rows(batch), 4
    return {
        "logits": np.repeat(np.arange(15, dtype=np.float32)[None], rows, axis=0),
        "particle_block_2": _hidden(rows, tokens),
        "jet_penultimate": np.repeat(np.arange(128, dtype=np.float32)[None], rows, axis=0),
        "particle_mask": np.ones((rows, tokens), dtype=np.bool_),
        "vectors": _vectors(rows, tokens),
        "visible_indices": np.repeat(np.arange(tokens, dtype=np.int64)[None], rows, axis=0),
        "family_codes": np.zeros((rows, tokens), dtype=np.int8),
    }


def _toff_surface(batch: TargetForwardBatch | TeacherModelInputs):
    rows = _surface_rows(batch)
    return {
        "logits": np.repeat(np.arange(15, dtype=np.float32)[None], rows, axis=0),
        "charged_particle_block_2": _hidden(rows, 4),
        "neutral_particle_block_2": _hidden(rows, 1, offset=32),
        "offline_jet_penultimate": np.repeat(
            np.arange(128, dtype=np.float32)[None], rows, axis=0,
        ),
        "charged_mask": np.ones((rows, 4), dtype=np.bool_),
        "neutral_mask": np.ones((rows, 1), dtype=np.bool_),
        "charged_vectors": _vectors(rows, 4),
        "neutral_vectors": _vectors(rows, 1, offset=0.1),
        "charged_visible_indices": np.repeat(
            np.arange(4, dtype=np.int64)[None], rows, axis=0,
        ),
        "neutral_visible_indices": np.zeros((rows, 1), dtype=np.int64),
    }


def _environment(forward: dict) -> dict:
    payload = forward["payload"]
    return {
        "producer": payload["producer"],
        "device": {**payload["device"], "gpu_uuid": "GPU-test"},
        "precision": payload["precision"],
        "determinism": payload["determinism"],
    }


def _context(
    tmp_path: Path, *, bank_id="D0c", purpose="screen", subset=None,
    generation_parent=H, suffix="", consumer_seed: int = 1337,
    consumers=None,
):
    logical = _logical(bank_id)
    registry = _registry(
        logical, purpose=purpose, subset=subset, parent=generation_parent,
        seed=consumer_seed, consumers=consumers,
    )
    forward = _forward(logical)
    batches = {
        "p0": _batch("p0", 0, [0, 1], toff=bank_id == "TOFF"),
        "p1": _batch("p1", 1, [0, 1], toff=bank_id == "TOFF"),
    }
    identities = np.concatenate([batches["p0"].identity_digest, batches["p1"].identity_digest])
    labels = np.concatenate([batches["p0"].label, batches["p1"].label])
    source_file_ids = np.concatenate([
        batches["p0"].source_file_id, batches["p1"].source_file_id,
    ])
    source_entries = np.concatenate([
        batches["p0"].source_entry, batches["p1"].source_entry,
    ])
    class_counts = np.bincount(labels.astype(np.int64), minlength=15).tolist()
    context = begin_target_generation(
        tmp_path / f"banks{suffix}" / bank_id,
        logical_bank=logical, consumer_registry=registry, forward_spec=forward,
        partitions={
            "p0": {"rows": 2, "source_file_id": 0},
            "p1": {"rows": 2, "source_file_id": 1},
        },
        expected_class_counts=class_counts,
        expected_identity_order_sha256=identity_order_sha256(identities),
        expected_identity_set_sha256=identity_set_sha256(identities),
        expected_population_rows_sha256=target_population_rows_sha256(
            source_file_id=source_file_ids,
            source_entry=source_entries,
            identity_digest=identities,
            label=labels,
        ),
        build_owner={"task": f"builder{suffix}"}, target_storage_cap_bytes=10_000_000,
        storage_estimate=_storage_estimate(logical, rows=4),
        resource_profile=_resource_profile(),
        **TARGET_PREFLIGHT,
    )
    return context, batches


@pytest.fixture(scope="module")
def resources():
    return generate_spectral_resources("token"), generate_spectral_resources("relation")


def _build(tmp_path: Path, resources, *, bank_id="D0c", purpose="screen", subset=None,
           parent=H, suffix="", prior=None):
    context, batches = _context(
        tmp_path, bank_id=bank_id, purpose=purpose, subset=subset,
        generation_parent=parent, suffix=suffix,
    )
    calls = []
    forward = _toff_surface if bank_id == "TOFF" else _ordinary_surface

    def teacher(model_inputs):
        calls.append(int(model_inputs.arrays["features"][0, 0]))
        return forward(model_inputs)

    result = build_target_generation_from_teacher(
        context,
        partition_batches={name: (lambda batch=batch: iter((batch,))) for name, batch in batches.items()},
        teacher_forward=teacher, token_resources=resources[0], relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        runtime_environment=_environment(context.forward_spec),
        prior_execution_attestation=prior,
    )
    return context, batches, result, calls


def test_exact_target_schema_formulas_and_semantics():
    assert target_core_values_per_row(ORDINARY_BANK) == 1935
    assert target_metadata_bytes_per_row(ORDINARY_BANK) == 75
    assert target_logical_bytes_per_row(ORDINARY_BANK) == 7815
    assert target_core_values_per_row(TOFF_BANK) == 3727
    assert target_metadata_bytes_per_row(TOFF_BANK) == 113
    assert target_logical_bytes_per_row(TOFF_BANK) == 15021
    assert target_array_schema(ORDINARY_BANK, 2)["token_kernel_mean"][1] == (2, 1024)
    assert target_array_schema(TOFF_BANK, 2)["family_reason_counts"][1] == (2, 6)


def test_one_forward_ordinary_build_materialization_join_and_reuse(tmp_path: Path, resources):
    context, batches, result, calls = _build(tmp_path, resources)
    first_shard = result.manifest["payload"]["shards"][0]
    with zipfile.ZipFile(
        context.committed_directory / first_shard["payload_path"], "r",
    ) as archive:
        assert archive.infolist()
        assert all(
            member.compress_type == zipfile.ZIP_DEFLATED
            for member in archive.infolist()
        )
    assert result.teacher_forward_calls == 2
    assert len(calls) == 2
    assert result.execution_attestation["payload"]["construction_seconds"] >= 0
    assert result.manifest["payload"]["rows"] == 4
    assert result.manifest["payload"]["teacher_total_tokens"] == 16
    assert result.manifest["payload"]["companion_hlt_total_tokens"] == 16
    assert validate_target_generation(context.bank_root, context.generation_id)["content_hash"] == result.manifest["content_hash"]

    rset = RepresentationTargetBank.load(context.bank_root, context.generation_id, strategy="RSET")
    rrel = RepresentationTargetBank.load(context.bank_root, context.generation_id, strategy="RREL")
    jet = RepresentationTargetBank.load(context.bank_root, context.generation_id, strategy="JET_ONLY")
    assert "token_kernel_mean" in rset.arrays and "relation_kernel_mean" not in rset.arrays
    assert "relation_kernel_mean" in rrel.arrays
    assert set(jet.arrays) == {
        "source_file_id", "source_entry", "identity_digest", "label", "logits",
        "jet_penultimate",
    }
    joined = rrel.join(batches["p1"].identity_digest[::-1].copy())
    assert joined["logits"].shape == (2, 15)
    with pytest.raises(KeyError, match="incomplete"):
        rrel.join(np.zeros((1, 32), dtype=np.uint8))

    second = build_target_generation_from_teacher(
        context,
        partition_batches={name: (lambda batch=batch: iter((batch,))) for name, batch in batches.items()},
        teacher_forward=lambda batch: pytest.fail("committed generation reran teacher"),
        token_resources=resources[0], relation_resources=resources[1],
        runtime_environment=_environment(context.forward_spec),
    )
    assert second.teacher_forward_calls == 0
    assert second.reused_partitions == ("p0", "p1")


def test_prepared_target_path_never_reopens_source_or_reforwards_teacher(
    tmp_path: Path, resources,
):
    context, batches = _context(tmp_path, suffix="-prepared-one-pass")
    source_calls = {name: 0 for name in batches}
    teacher_calls = []

    def factory(name, batch):
        def iterate():
            source_calls[name] += 1
            if source_calls[name] != 1:
                raise AssertionError("prepared target source was reopened")
            yield batch

        return iterate

    def teacher(model_inputs):
        teacher_calls.append(1)
        return _ordinary_surface(model_inputs)

    partition_batches = {
        name: factory(name, batch) for name, batch in batches.items()
    }
    prepared = prepare_target_generation_in_memory(
        bank_kind=ORDINARY_BANK,
        partition_batches=partition_batches,
        partition_specs=context.build_intent["payload"]["partitions"],
        teacher_forward=teacher,
        token_resources=resources[0],
        relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        allowed_input_fields=("features", "mask", "points", "v"),
    )
    assert source_calls == {"p0": 1, "p1": 1}
    assert len(teacher_calls) == 2
    assert prepared.teacher_forward_calls == 2

    result = build_target_generation_from_prepared(
        context,
        prepared=prepared,
        token_resources=resources[0],
        relation_resources=resources[1],
        runtime_environment=_environment(context.forward_spec),
    )
    assert source_calls == {"p0": 1, "p1": 1}
    assert len(teacher_calls) == 2
    assert result.teacher_forward_calls == 2
    assert result.manifest["payload"]["rows"] == 4


def test_runtime_uses_canonical_fp32_target_builders_and_label_free_callback(
    tmp_path: Path, resources,
):
    context, batches = _context(tmp_path, suffix="-canonical-builders")
    observed = []

    def teacher(model_inputs):
        observed.append(model_inputs)
        assert isinstance(model_inputs, TeacherModelInputs)
        assert set(model_inputs.arrays) == {"features", "mask", "points", "v"}
        assert not hasattr(model_inputs, "label")
        assert not hasattr(model_inputs, "source_entry")
        return _ordinary_surface(model_inputs)

    result = build_target_generation_from_teacher(
        context,
        partition_batches={
            name: (lambda batch=batch: iter((batch,)))
            for name, batch in batches.items()
        },
        teacher_forward=teacher,
        token_resources=resources[0],
        relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        runtime_environment=_environment(context.forward_spec),
    )
    assert len(observed) == 2
    first = batches["p0"]
    surface = _ordinary_surface(TeacherModelInputs(first.teacher_inputs))
    token = build_ordinary_token_targets(
        torch.from_numpy(surface["particle_block_2"]),
        torch.from_numpy(surface["vectors"]),
        torch.from_numpy(surface["particle_mask"]),
        resources[0],
    )
    relation = build_teacher_relation_targets(
        torch.from_numpy(surface["particle_block_2"]),
        torch.from_numpy(surface["vectors"]),
        torch.from_numpy(surface["particle_mask"]),
        torch.from_numpy(surface["visible_indices"]),
        resources[1],
    )
    shard = result.manifest["payload"]["shards"][0]
    arrays = load_npz_arrays(context.committed_directory / shard["payload_path"])
    assert arrays["token_kernel_mean"].dtype == np.float32
    assert arrays["relation_kernel_mean"].dtype == np.float32
    assert np.array_equal(arrays["token_kernel_mean"], token.means.numpy())
    assert np.array_equal(
        arrays["relation_kernel_mean"], relation.means[:, 0].numpy(),
    )


def test_toff_runtime_matches_canonical_fp32_family_builders(
    tmp_path: Path, resources,
):
    context, batches, result, _ = _build(
        tmp_path, resources, bank_id="TOFF", suffix="-canonical-toff-builders",
    )
    first = batches["p0"]
    surface = _toff_surface(TeacherModelInputs(first.teacher_inputs))
    token = build_native_offline_token_targets(
        torch.from_numpy(surface["charged_particle_block_2"]),
        torch.from_numpy(surface["charged_vectors"]),
        torch.from_numpy(surface["charged_mask"]),
        torch.from_numpy(surface["neutral_particle_block_2"]),
        torch.from_numpy(surface["neutral_vectors"]),
        torch.from_numpy(surface["neutral_mask"]),
        resources[0],
    )
    charged_relation = build_teacher_relation_targets(
        torch.from_numpy(surface["charged_particle_block_2"]),
        torch.from_numpy(surface["charged_vectors"]),
        torch.from_numpy(surface["charged_mask"]),
        torch.from_numpy(surface["charged_visible_indices"]),
        resources[1],
    )
    neutral_relation = build_teacher_relation_targets(
        torch.from_numpy(surface["neutral_particle_block_2"]),
        torch.from_numpy(surface["neutral_vectors"]),
        torch.from_numpy(surface["neutral_mask"]),
        torch.from_numpy(surface["neutral_visible_indices"]),
        resources[1],
    )
    shard = result.manifest["payload"]["shards"][0]
    arrays = load_npz_arrays(context.committed_directory / shard["payload_path"])
    assert np.array_equal(
        arrays["token_kernel_mean_charged"], token.means[:, 0].numpy(),
    )
    assert np.array_equal(
        arrays["token_kernel_mean_neutral"], token.means[:, 1].numpy(),
    )
    assert np.array_equal(
        arrays["token_family_eligibility"], token.present.numpy().astype(np.uint8),
    )
    assert np.array_equal(
        arrays["relation_kernel_mean_charged"],
        charged_relation.means[:, 0].numpy(),
    )
    assert np.array_equal(
        arrays["relation_kernel_mean_neutral"],
        neutral_relation.means[:, 0].numpy(),
    )
    assert np.array_equal(
        arrays["relation_eligibility"][:, 0],
        charged_relation.eligible[:, 0].numpy().astype(np.uint8),
    )
    assert np.array_equal(
        arrays["relation_eligibility"][:, 1],
        neutral_relation.eligible[:, 0].numpy().astype(np.uint8),
    )


def test_exact_population_mapping_rejects_label_remapping_with_equal_counts(
    tmp_path: Path, resources,
):
    context, batches = _context(tmp_path, suffix="-population-remap")
    original = batches["p0"]
    batches["p0"] = replace(original, label=original.label[::-1].copy())
    assert sorted(batches["p0"].label.tolist()) == sorted(original.label.tolist())
    with pytest.raises(ValueError, match="identity population differs"):
        build_target_generation_from_teacher(
            context,
            partition_batches={
                name: (lambda batch=batch: iter((batch,)))
                for name, batch in batches.items()
            },
            teacher_forward=_ordinary_surface,
            token_resources=resources[0],
            relation_resources=resources[1],
            teacher_model=_inspectable_teacher_model(),
            runtime_environment=_environment(context.forward_spec),
        )
    assert not context.committed_directory.exists()


def test_committed_target_reuse_requires_equal_arrays_and_runtime_audit(
    tmp_path: Path, resources,
):
    context, _, result, _ = _build(tmp_path, resources, suffix="-committed-equality")
    shard = result.manifest["payload"]["shards"][0]
    sidecar = load_json(context.committed_directory / shard["sidecar_path"])
    arrays = load_npz_arrays(context.committed_directory / shard["payload_path"])
    arrays["logits"] = arrays["logits"].copy()
    arrays["logits"][0, 0] += np.float32(1.0)
    with pytest.raises(FileExistsError, match="shard request differs"):
        stage_target_shard(
            context,
            partition=shard["source_partition"],
            arrays=arrays,
            runtime_audit=sidecar["payload"]["target_runtime_audit"],
        )
    execution_facts = {
        name: result.execution_attestation["payload"][name]
        for name in (
            "producer", "device", "precision", "determinism", "batch_count",
            "batch_partition_sha256", "sentinel_hashes", "construction_seconds",
        )
    }
    execution_facts["batch_count"] += 1
    with pytest.raises(FileExistsError, match="execution request differs"):
        finalize_target_generation(context, execution_facts=execution_facts)


def test_staged_corruption_is_quarantined_and_committed_corruption_fails_closed(
    tmp_path: Path, resources,
):
    context, batches = _context(tmp_path, suffix="-staged-corruption")
    arguments = dict(
        context=context,
        partition_batches={
            name: (lambda batch=batch: iter((batch,)))
            for name, batch in batches.items()
        },
        teacher_forward=_ordinary_surface,
        token_resources=resources[0],
        relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        runtime_environment=_environment(context.forward_spec),
    )

    def fail_after_first_sidecar(boundary: str):
        if boundary == "after_shard_sidecar:p0":
            raise RuntimeError("injected after first sidecar")

    with pytest.raises(RuntimeError, match="first sidecar"):
        build_target_generation_from_teacher(
            **arguments, failure_hook=fail_after_first_sidecar,
        )
    staged_payload = context.staging_directory / "shards" / "p0.npz"
    staged_payload.write_bytes(b"not-a-zip")
    result = build_target_generation_from_teacher(**arguments)
    assert result.manifest["payload"]["rows"] == 4
    quarantined = list((context.bank_root / "quarantine").rglob("p0.npz"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"not-a-zip"

    committed_payload = context.committed_directory / result.manifest["payload"]["shards"][0][
        "payload_path"
    ]
    committed_payload.write_bytes(b"committed-corruption")
    with pytest.raises(ValueError, match="absent or corrupt|bytes differ|shard record differs"):
        validate_target_generation(context.bank_root, context.generation_id)


def test_one_forward_toff_separates_families_and_conserves_reasons(tmp_path: Path, resources):
    context, _, result, calls = _build(tmp_path, resources, bank_id="TOFF")
    assert len(calls) == 2
    payload = result.manifest["payload"]
    assert payload["teacher_family_token_counts"] == [16, 4]
    assert payload["companion_hlt_reason_counts"] == [4, 4, 4, 4, 4, 4]
    assert payload["companion_hlt_family_counts"] == [8, 8]
    assert payload["companion_hlt_unclassified_count"] == 8
    assert payload["companion_hlt_total_tokens"] == 24
    bank = RepresentationTargetBank.load(context.bank_root, context.generation_id)
    assert "token_kernel_mean_charged" in bank.arrays
    assert "token_kernel_mean_neutral" in bank.arrays
    assert np.all(bank.arrays["relation_eligibility"][:, 1] == 0)


@pytest.mark.parametrize("field", ("charge", "pid"))
def test_toff_raw_family_inputs_fail_before_privileged_callback(
    tmp_path: Path, resources, field: str,
):
    context, batches = _context(
        tmp_path, bank_id="TOFF", suffix=f"-invalid-{field}",
    )
    original = batches["p0"]
    if field == "charge":
        charge = original.companion_hlt_charge.copy()
        charge[0, 0] = np.float32(2.0)
        batches["p0"] = replace(original, companion_hlt_charge=charge)
    else:
        pid = original.companion_hlt_pid_flags.copy()
        pid[0, 0, 0] = np.float32(np.nan)
        batches["p0"] = replace(original, companion_hlt_pid_flags=pid)
    calls = []
    with pytest.raises((ValueError, FloatingPointError), match="raw charge|raw PID|companion_hlt_pid_flags"):
        build_target_generation_from_teacher(
            context,
            partition_batches={
                name: (lambda batch=batch: iter((batch,)))
                for name, batch in batches.items()
            },
            teacher_forward=lambda model_inputs: (
                calls.append(True), _toff_surface(model_inputs)
            )[1],
            token_resources=resources[0],
            relation_resources=resources[1],
            teacher_model=_inspectable_teacher_model(),
            runtime_environment=_environment(context.forward_spec),
        )
    assert calls == []


@pytest.mark.parametrize(
    "boundary",
    (
        "after_target_build_intent",
        "after_shard_payload:p0",
        "after_shard_sidecar:p0",
        "after_shard_payload:p1",
        "after_shard_sidecar:p1",
        "after_execution_attestation",
        "after_manifest",
        "before_directory_rename",
        "after_directory_rename",
    ),
)
def test_target_generation_crash_boundaries_are_same_owner_resumable(
    tmp_path: Path, resources, boundary: str,
):
    context, batches = _context(tmp_path, suffix=f"-{boundary.replace(':', '-')}")
    calls = []
    injected = False

    def teacher(model_inputs):
        calls.append(int(model_inputs.arrays["features"][0, 0]))
        return _ordinary_surface(model_inputs)

    def fail(name: str):
        nonlocal injected
        if name == boundary and not injected:
            injected = True
            raise RuntimeError(f"injected at {name}")

    arguments = dict(
        context=context,
        partition_batches={name: (lambda batch=batch: iter((batch,))) for name, batch in batches.items()},
        teacher_forward=teacher, token_resources=resources[0], relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        runtime_environment=_environment(context.forward_spec),
    )
    with pytest.raises(RuntimeError, match="injected"):
        build_target_generation_from_teacher(**arguments, failure_hook=fail)
    result = build_target_generation_from_teacher(**arguments)
    assert result.manifest["payload"]["rows"] == 4
    assert validate_target_generation(context.bank_root, context.generation_id)["content_hash"] == result.manifest["content_hash"]
    assert not any(
        path.is_dir() and any(path.iterdir())
        for path in (context.bank_root / "staging" / context.generation_id).glob("*")
    )


def test_wrong_teacher_and_invalid_preflight_fail_before_mutation(tmp_path: Path):
    logical = _logical()
    registry = _registry(logical)
    payload = _forward_payload(logical)
    payload["teacher"]["tap_sha256"] = _sha("wrong-tap")
    wrong = build_target_forward_spec(
        parents={"logical_bank": logical["content_hash"]}, payload=payload,
    )
    identities = np.stack([_identity(0, 0), _identity(0, 1)])
    source_file_ids = np.asarray([0, 0], dtype="<u4")
    source_entries = np.asarray([0, 1], dtype="<u8")
    labels = np.asarray([0, 1], dtype="u1")
    population_hash = target_population_rows_sha256(
        source_file_id=source_file_ids,
        source_entry=source_entries,
        identity_digest=identities,
        label=labels,
    )
    with pytest.raises(ValueError, match="teacher/logical-bank lineage"):
        begin_target_generation(
            tmp_path / "wrong", logical_bank=logical, consumer_registry=registry,
            forward_spec=wrong, partitions={"p0": {"rows": 2, "source_file_id": 0}},
            expected_class_counts=[1, 1] + [0] * 13,
            expected_identity_order_sha256=identity_order_sha256(identities),
            expected_identity_set_sha256=identity_set_sha256(identities),
            expected_population_rows_sha256=population_hash,
            build_owner={"task": "wrong"}, target_storage_cap_bytes=1,
            storage_estimate=_storage_estimate(logical, rows=2),
            resource_profile=_resource_profile(),
            **TARGET_PREFLIGHT,
        )
    assert not (tmp_path / "wrong").exists()

    forward = _forward(logical, partitions=("p0",))
    with pytest.raises(MemoryError, match="storage cap"):
        begin_target_generation(
            tmp_path / "cap", logical_bank=logical, consumer_registry=registry,
            forward_spec=forward, partitions={"p0": {"rows": 2, "source_file_id": 0}},
            expected_class_counts=[1, 1] + [0] * 13,
            expected_identity_order_sha256=identity_order_sha256(identities),
            expected_identity_set_sha256=identity_set_sha256(identities),
            expected_population_rows_sha256=population_hash,
            build_owner={"task": "cap"}, target_storage_cap_bytes=2 * 7815 - 1,
            storage_estimate=_storage_estimate(logical, rows=2),
            resource_profile=_resource_profile(),
            **TARGET_PREFLIGHT,
        )
    assert not (tmp_path / "cap").exists()

    forged_storage = deepcopy(_storage_estimate(logical, rows=2))
    forged_storage["ordinary_bank_bytes"] += 1
    forged_storage = with_content_hash({
        key: value for key, value in forged_storage.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="storage-estimate evidence"):
        begin_target_generation(
            tmp_path / "forged-storage", logical_bank=logical,
            consumer_registry=registry, forward_spec=forward,
            partitions={"p0": {"rows": 2, "source_file_id": 0}},
            expected_class_counts=[1, 1] + [0] * 13,
            expected_identity_order_sha256=identity_order_sha256(identities),
            expected_identity_set_sha256=identity_set_sha256(identities),
            expected_population_rows_sha256=population_hash,
            build_owner={"task": "forged-storage"},
            target_storage_cap_bytes=10_000_000,
            storage_estimate=forged_storage,
            resource_profile=_resource_profile(),
            **TARGET_PREFLIGHT,
        )
    assert not (tmp_path / "forged-storage").exists()


def test_array_ineligibility_and_relation_family_gate_fail_closed():
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in target_array_schema(ORDINARY_BANK, 1).items()}
    arrays["source_entry"][0] = 1
    arrays["identity_digest"][0] = _identity(0, 1)
    arrays["token_count"][0, 0] = 0
    arrays["token_kernel_mean"][0, 0] = 1.0
    with pytest.raises(ValueError, match="ineligible token sketch"):
        validate_target_arrays(arrays, bank_kind=ORDINARY_BANK)
    arrays["token_kernel_mean"][0, 0] = 0.0
    arrays["relation_pair_count"][0, 0, 0] = 4
    arrays["relation_effective_sample"][0, 0, 0] = 3.0
    arrays["relation_eligibility"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="empty token family"):
        validate_target_arrays(arrays, bank_kind=ORDINARY_BANK)


def test_predecessor_logits_are_built_once_joined_and_model_released():
    batches = {
        "p0": _batch("p0", 0, [0, 1]),
        "p1": _batch("p1", 1, [0, 1]),
    }
    identities = np.concatenate([batches["p0"].identity_digest, batches["p1"].identity_digest])
    calls = []
    releases = []

    def forward(model_inputs):
        source_id = int(model_inputs.arrays["features"][0, 0])
        calls.append(f"p{source_id}")
        return np.full(
            (len(model_inputs.arrays["features"]), 15),
            float(source_id),
            dtype=np.float32,
        )

    bank = build_predecessor_logit_cache(
        partition_batches={name: (lambda batch=batch: iter((batch,))) for name, batch in batches.items()},
        predecessor_forward=forward, release_predecessor=lambda: releases.append(True),
        teacher_input_fields=("features", "mask", "points", "v"),
        predecessor_model=_inspectable_teacher_model(),
        expected_rows=4,
        expected_identity_order_sha256=identity_order_sha256(identities),
        expected_identity_set_sha256=identity_set_sha256(identities),
        predecessor_checkpoint_logical_sha256=_sha("predecessor"),
    )
    assert calls == ["p0", "p1"]
    assert releases == [True]
    assert np.all(bank.join(batches["p1"].identity_digest) == 1.0)


def _reports(manifest: dict) -> dict[str, dict]:
    reports = {}
    for row in manifest["payload"]["authorized_consumers"]:
        spec = NODE_REGISTRY.get(row["node_id"], CONTROL_REGISTRY.get(row["node_id"]))
        assert spec is not None
        metrics = {
            "rows": 1,
            "macro_ovr_auc": 0.5,
            "cross_entropy": 1.0,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "always_qcd_accuracy": 0.5,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 0.0,
            "multiclass_brier": 1.0,
            "multiclass_brier_score": 1.0,
            "top_label_ece_15_bin": 0.1,
            "confusion_matrix": [[0] * 15 for _ in range(15)],
            "per_class": {f"class_{index}": {} for index in range(15)},
        }
        checkpoint_id = f"checkpoint-{row['node_id']}"
        boundary = [
            "validation", "required_gradient_calibration_barrier",
            "representation_diagnostic", "boundary_resume_commit",
        ]
        diagnostic_sha256 = _sha(f"{row['execution_id']}:diagnostic")
        calibration_selection_sha256 = _sha(
            f"{row['execution_id']}:calibration-selection"
        )
        history_row = {
            "checkpoint_id": checkpoint_id,
            "completed_pass": 1,
            "update": 1,
            "validation": metrics,
            "selector_inputs": {
                name: {"value": metrics[name], "hex": float(metrics[name]).hex()}
                for name in (
                    "macro_ovr_auc", "cross_entropy",
                    "macro_mean_log_qcd_rejection_at_50pct_signal",
                )
            },
            "boundary_order": boundary,
            "representation_diagnostic": {
                "student_forward_calls": 1,
                "finite": True,
                "components": {},
            },
        }
        reports[row["execution_id"]] = with_content_hash({
            "contract": TRAINING_REPORT_CONTRACT,
            "schema_version": 1,
            "node_id": row["node_id"],
            "execution_id": row["node_id"],
            "registered_execution_id": row["execution_id"],
            "replicate_seed": row["seed"],
            "campaign_sha256": "c" * 64,
            "paired_rng_streams": paired_rng_streams(
                row["node_id"], row["seed"],
            ),
            "graph_sha256": ASCENT_GRAPH_SHA256,
            "recipe_sha256": RECIPE,
            "parent_recipe_sha256": H,
            "parent_counterpart": spec.parent_counterpart,
            "control_counterpart": getattr(spec, "paired_primary_node", None),
            "rung": int(spec.rung),
            "complete": True,
            "scientific_complete": False,
            "mode": "smoke",
            "student_domain": "hlt",
            "strategy": row["strategy"],
            "track": row["track"],
            "completed_optimizer_updates": 1,
            "completed_natural_population_passes": 1,
            "validation_every_complete_pass": True,
            "validation_history": [history_row],
            "validation": metrics,
            "selected_checkpoint_id": checkpoint_id,
            "selected_training_checkpoint_path": f"training/{row['node_id']}.pt",
            "selected_training_checkpoint_sha256": _sha(f"{row['node_id']}:checkpoint"),
            "selection_sha256": _sha(f"{row['node_id']}:selection"),
            "deployable_extraction": {
                "strict_hlt_only": True,
                "checkpoint_sha256": _sha(f"{row['node_id']}:deployable"),
                "report_sha256": _sha(f"{row['node_id']}:extraction"),
            },
            "checkpoint_envelopes": {
                "selected": {
                    "envelope_id": _sha(f"{row['execution_id']}:selected-envelope"),
                    "commit_sha256": _sha(f"{row['execution_id']}:selected-commit"),
                },
                "final": {
                    "envelope_id": _sha(f"{row['execution_id']}:final-envelope"),
                    "commit_sha256": _sha(f"{row['execution_id']}:final-commit"),
                },
            },
            "interval_mean_history": [{
                "examples": 1,
                "means": {"total": 1.0},
            }],
            "calibration": {
                "artifact_hashes": {"manifest": None},
                "diagnostic_batch_sha256": diagnostic_sha256,
                "selection_sha256": calibration_selection_sha256,
                "ordered_selection_sha256": _sha(
                    f"{row['execution_id']}:ordered-calibration-selection"
                ),
            },
            "calibration_selection_sha256": calibration_selection_sha256,
            "diagnostic_batch_sha256": diagnostic_sha256,
            "calibration_manifest_sha256": None,
            "boundary_protocol": boundary,
            "resume_audit": {},
            "target_generation_sha256": manifest["parents"]["target_generation"],
            "target_logical_sha256": manifest["payload"]["logical_target_sha256"],
            "target_manifest_sha256": manifest["content_hash"],
            "target_cache_diagnostics": {
                "construction_seconds": 0.0,
                "load_seconds": 0.0,
                "generation_sha256": manifest["parents"]["target_generation"],
                "logical_sha256": manifest["payload"]["logical_target_sha256"],
                "manifest_sha256": manifest["content_hash"],
            },
            "predecessor_model_released_before_optimization": True,
            "finite_poor_results_retained": True,
            "performance_early_stopping": False,
        })
    return reports


def test_two_phase_cleanup_crash_resume_and_reconstruction(tmp_path: Path, resources):
    context, _, result, _ = _build(tmp_path, resources, bank_id="TOFF")
    cleanup_root = tmp_path / "cleanup"
    reports = _reports(result.manifest)
    with pytest.raises(ValueError, match="incomplete"):
        authorize_target_cleanup(
            context.bank_root, cleanup_root, generation_id=context.generation_id,
            consumer_reports={next(iter(reports)): next(iter(reports.values()))},
            exact_reconstruction_authorized=True,
        )
    authorization = authorize_target_cleanup(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
        consumer_reports=reports, exact_reconstruction_authorized=True,
    )
    assert inspect_target_recovery_state(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
    ) == TargetRecoveryState.CLEANUP_AUTHORIZED_IN_PROGRESS

    seen = False

    def fail(boundary: str):
        nonlocal seen
        if boundary == "after_cleanup_deletion:0" and not seen:
            seen = True
            raise RuntimeError("injected cleanup crash")

    with pytest.raises(RuntimeError, match="injected"):
        complete_target_cleanup(
            context.bank_root, cleanup_root, generation_id=context.generation_id,
            failure_hook=fail,
        )
    completion = complete_target_cleanup(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
    )
    validate_cleanup_completion(completion, authorization=authorization, bank_root=context.bank_root)
    forged_completion = deepcopy(completion)
    forged_completion["payload"]["removal_observations"][0]["expected_bytes"] += 1
    forged_completion = with_content_hash({
        key: value for key, value in forged_completion.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="removal observation"):
        validate_cleanup_completion(
            forged_completion, authorization=authorization,
            bank_root=context.bank_root,
        )
    assert inspect_target_recovery_state(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
    ) == TargetRecoveryState.CLEANUP_COMPLETED
    assert all(
        not (context.committed_directory / row["payload_path"]).exists()
        for row in result.manifest["payload"]["shards"]
    )

    confirmation = build_confirmation_registry(
        screen_sha256=_sha("screen"),
        campaign_sha256=CAMPAIGN,
        recipe_sha256=RECIPE,
        target_logical_bank_sha256=context.logical_bank["content_hash"],
        objectives=("RSET_M6c", "RSET_M6w", "RREL_M6c", "RREL_M6w"),
    )
    unfinished = [
        _consumer(
            context.logical_bank,
            row["objective_id"],
            seed=row["seed"],
            purpose="confirmation",
        )
        for row in confirmation["rows"]
    ]
    assert [row["execution_id"] for row in unfinished] == [
        row["execution_id"] for row in confirmation["rows"]
    ]
    self_authorization = build_target_consumer_registry(
        context.logical_bank,
        purpose="recovery",
        consumers=unfinished,
        generation_parent_sha256=completion["content_hash"],
    )
    with pytest.raises(ValueError, match="cannot self-authorize"):
        build_target_recovery_plan(
            bank_root=context.bank_root,
            logical_bank=context.logical_bank,
            prior_manifest=result.manifest,
            prior_cleanup_authorization=authorization,
            prior_cleanup_completion=completion,
            prior_execution_attestation=result.execution_attestation,
            recovery_consumers=unfinished,
            consumer_authorization=self_authorization,
            recovery_owner={"task": "invalid-self-authorization"},
        )
    forged_confirmation = deepcopy(confirmation)
    forged_confirmation["execution_count"] = 19
    forged_confirmation = with_content_hash({
        key: value for key, value in forged_confirmation.items()
        if key != "content_hash"
    })
    with pytest.raises(ValueError, match="confirmation consumer rows differ"):
        build_target_recovery_plan(
            bank_root=context.bank_root,
            logical_bank=context.logical_bank,
            prior_manifest=result.manifest,
            prior_cleanup_authorization=authorization,
            prior_cleanup_completion=completion,
            prior_execution_attestation=result.execution_attestation,
            recovery_consumers=unfinished,
            consumer_authorization=forged_confirmation,
            recovery_owner={"task": "invalid-confirmation-registry"},
        )
    forged_attestation = deepcopy(result.execution_attestation)
    forged_attestation["payload"]["logical_target_sha256"] = _sha("wrong-target")
    forged_attestation = with_content_hash({
        key: value for key, value in forged_attestation.items() if key != "content_hash"
    })
    with pytest.raises(ValueError, match="prior target|execution/cleanup lineage"):
        build_target_recovery_plan(
            bank_root=context.bank_root,
            logical_bank=context.logical_bank,
            prior_manifest=result.manifest,
            prior_cleanup_authorization=authorization,
            prior_cleanup_completion=completion,
            prior_execution_attestation=forged_attestation,
            recovery_consumers=unfinished,
            consumer_authorization=confirmation,
            recovery_owner={"task": "invalid-lineage"},
        )
    plan = build_target_recovery_plan(
        bank_root=context.bank_root,
        logical_bank=context.logical_bank,
        prior_manifest=result.manifest,
        prior_cleanup_authorization=authorization,
        prior_cleanup_completion=completion,
        prior_execution_attestation=result.execution_attestation,
        recovery_consumers=unfinished,
        consumer_authorization=confirmation,
        recovery_owner={"task": "rebuild"},
    )
    new_context, batches = _context(
        tmp_path, bank_id="TOFF", purpose="recovery",
        generation_parent=plan["content_hash"], suffix="-recovery",
        consumers=unfinished,
    )
    calls = []

    def teacher(model_inputs):
        calls.append(int(model_inputs.arrays["features"][0, 0]))
        return _toff_surface(model_inputs)

    rebuilt = build_target_generation_from_teacher(
        new_context,
        partition_batches={name: (lambda batch=batch: iter((batch,))) for name, batch in batches.items()},
        teacher_forward=teacher, token_resources=resources[0], relation_resources=resources[1],
        teacher_model=_inspectable_teacher_model(),
        runtime_environment=_environment(new_context.forward_spec),
        prior_execution_attestation=result.execution_attestation,
    )
    assert rebuilt.sentinel_replay_calls == 2  # two batches: first and middle/last
    assert rebuilt.teacher_forward_calls == 2
    validate_reconstructed_generation(
        result.manifest, rebuilt.manifest, recovery_plan=plan,
        logical_bank=new_context.logical_bank,
        new_consumer_registry=new_context.consumer_registry,
    )
    assert rebuilt.manifest["payload"]["logical_target_sha256"] == result.manifest["payload"]["logical_target_sha256"]


@pytest.mark.parametrize(
    "boundary",
    (
        "after_cleanup_deletion:0",
        "after_cleanup_deletion:1",
        "before_cleanup_completion",
        "after_cleanup_completion",
    ),
)
def test_cleanup_resumes_every_deletion_and_completion_boundary(
    tmp_path: Path, resources, boundary: str,
):
    context, _, result, _ = _build(
        tmp_path, resources, suffix=f"-cleanup-{boundary.replace(':', '-')}",
    )
    cleanup_root = tmp_path / f"cleanup-{boundary.replace(':', '-')}"
    authorization = authorize_target_cleanup(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
        consumer_reports=_reports(result.manifest),
        exact_reconstruction_authorized=True,
    )
    injected = False

    def fail(name: str):
        nonlocal injected
        if name == boundary and not injected:
            injected = True
            raise RuntimeError(f"injected at {name}")

    with pytest.raises(RuntimeError, match="injected"):
        complete_target_cleanup(
            context.bank_root, cleanup_root, generation_id=context.generation_id,
            failure_hook=fail,
        )
    completion = complete_target_cleanup(
        context.bank_root, cleanup_root, generation_id=context.generation_id,
    )
    validate_cleanup_completion(
        completion, authorization=authorization, bank_root=context.bank_root,
    )
    assert completion["payload"]["all_authorized_paths_absent"] is True


def test_cleanup_report_authentication_binds_registered_execution_node_and_seed(
    tmp_path: Path, resources,
):
    context, _, result, _ = _build(
        tmp_path, resources, suffix="-report-authentication",
    )
    reports = _reports(result.manifest)
    execution_id = next(iter(reports))
    wrong_node = (
        "RREL_M1c" if reports[execution_id]["node_id"] != "RREL_M1c"
        else "RSET_M1c"
    )
    mutations = (
        ("registered_execution_id", _sha("wrong-registered-execution")),
        ("node_id", wrong_node),
        ("replicate_seed", 999),
    )
    for field, value in mutations:
        forged_reports = deepcopy(reports)
        forged_reports[execution_id][field] = value
        forged_reports[execution_id] = with_content_hash({
            key: item for key, item in forged_reports[execution_id].items()
            if key != "content_hash"
        })
        with pytest.raises(ValueError):
            authorize_target_cleanup(
                context.bank_root,
                tmp_path / f"cleanup-forged-{field}",
                generation_id=context.generation_id,
                consumer_reports=forged_reports,
                exact_reconstruction_authorized=True,
            )
        assert not (tmp_path / f"cleanup-forged-{field}").exists()


def test_consumer_registries_bind_d100_controls_and_toff_confirmation():
    d100 = _logical("D100")
    d100_registry = _registry(d100)
    assert len(d100_registry["payload"]["consumers"]) == 8
    toff = _logical("TOFF")
    confirmation = _registry(toff, purpose="confirmation")
    assert len(confirmation["payload"]["consumers"]) == 20
    forged = deepcopy(d100_registry["payload"]["consumers"])
    forged[0]["strategy"] = "RREL" if forged[0]["strategy"] == "RSET" else "RSET"
    with pytest.raises(ValueError, match="node metadata"):
        build_target_consumer_registry(
            d100, purpose="screen", consumers=forged, generation_parent_sha256=H,
        )
    forged_identity = deepcopy(d100_registry["payload"]["consumers"])
    forged_identity[0]["execution_identity_payload"]["campaign"] = _sha("wrong-campaign")
    with pytest.raises(ValueError, match="execution identity differs"):
        build_target_consumer_registry(
            d100,
            purpose="screen",
            consumers=forged_identity,
            generation_parent_sha256=H,
        )
