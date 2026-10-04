"""New identities, not aliases for native Delphes K2 artifacts."""
from copy import deepcopy
from pathlib import Path

from hlt_classification.data.cache_contracts import (
    with_content_hash, validate_content_hash, sha256_file, load_json,
    write_immutable_json,
)
from hlt_classification.jetclass2_delphes.concat_k2_campaign import nodes, TRAINING
from hlt_classification.jetclass2_delphes.concat_k2_model import PAIR_STORAGE
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.model import model_contract

COUNTS = dict(train=100000, validation=50000)
AUTHORIZE = "AUTHORIZE OSCAR NOISE V3 K2 100K 50K EXACT SPEC"
RESOURCES = dict(
    metadata=dict(cpus=1, memory_mb=8192, minutes=240, gpu=False),
    assignment=dict(cpus=1, memory_mb=8192, minutes=720, gpu=False),
    preflight=dict(cpus=6, memory_mb=90000, minutes=720, gpu=True),
    train=dict(cpus=6, memory_mb=90000, minutes=1440, gpu=True),
    reduce=dict(cpus=6, memory_mb=90000, minutes=360, gpu=True),
)


def artifact(kind, **fields):
    return with_content_hash(dict(contract=f"NOISE_V3_K2_{kind}/v1",
        schema_version=1, **fields, final_test_accessed=False))


def validate(value, kind):
    result = validate_content_hash(value, expected_contract=f"NOISE_V3_K2_{kind}/v1")
    if value.get("final_test_accessed") is not False:
        raise PermissionError("Final-test capability is sealed")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def registration():
    return dict(counts=dict(COUNTS), nodes=nodes(), training=deepcopy(TRAINING),
        fresh_fit_count=10, reducer_count=5, science_task_count=17,
        loss=dict(ce=.25, kd=.75, temperature=2.),
        split_profile="NOISE_V3_TRAIN100K_VALID50K_v1", ordinary_final_test_capability=False,
        inference_batch_size=128, pair_storage=deepcopy(PAIR_STORAGE), model=model_contract(),
        resources=deepcopy(RESOURCES), execution_site=execution_site("oscar_l40s"),
        cpu_partition="batch", validation_fractions=[2, 1, 1],
        population_policy="class_identity_sha256_natural_hamilton_min_one/v1",
        gpu_peak_fraction=.90, rss_peak_fraction=.80, cache_fraction=.65,
        projection_margin=1.30, maximum_fit_seconds=23*3600,
        minimum_free_disk_bytes=8*1024**3, final_test_count=1000000,
        scientific_profile="full_quarter_ladder_100passes_batch128",
        dataset_kind="controlled_literature_synthetic_proxy", recipe="NOISE_V3")


def safe(root, relative):
    root = Path(root).resolve()
    path = root / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Unsafe artifact path")
    require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Artifact escapes root")
    return path


def reference(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path), bytes=path.stat().st_size)


def checked(root, record):
    path = safe(root, record["path"])
    require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"],
            "Artifact bytes differ: " + str(path))
    return path


def publish(path, value):
    write_immutable_json(Path(path), value)
    return value


def load(path, kind):
    value = load_json(path)
    validate(value, kind)
    return value
