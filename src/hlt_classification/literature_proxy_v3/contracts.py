"""Versioned, immutable v3 artifacts; v1 remains independently readable."""
from hlt_classification.literature_proxy.contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, read_reference, reference,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)


def artifact(kind, **values):
    return with_content_hash(dict(contract=f"JC2_LITERATURE_PROXY_NOISE_{kind}/v3",
                                  schema_version=3, **values))


def validate(value, kind):
    validate_content_hash(value, expected_contract=f"JC2_LITERATURE_PROXY_NOISE_{kind}/v3",
                          expected_schema_version=3)
    if value.get("schema_version") != 3:
        raise ValueError("Noise schema differs")
