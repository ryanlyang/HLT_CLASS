"""Small immutable artifact primitives, independent of fitted response schemas."""
from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, sha256_file,
    validate_content_hash, with_content_hash, write_immutable_json,
)


def artifact(kind, **values):
    return with_content_hash(dict(contract=f"JC2_LITERATURE_PROXY_{kind}/v1",
                                  schema_version=1, **values))


def validate(value, kind):
    return validate_content_hash(value, expected_contract=f"JC2_LITERATURE_PROXY_{kind}/v1")


def reference(path):
    from pathlib import Path
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha256_file(p))


def read_reference(ref):
    if sha256_file(ref["path"]) != ref["sha256"]:
        raise ValueError(f"Source bytes changed: {ref['path']}")
    return load_json(ref["path"])
