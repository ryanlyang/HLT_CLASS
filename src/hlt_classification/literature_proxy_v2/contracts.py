"""Versioned, immutable v2 artifacts; v1 remains independently readable."""
from hlt_classification.literature_proxy.contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, read_reference, reference,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)


def artifact(kind, **values):
    return with_content_hash(dict(contract=f"JC2_LITERATURE_PROXY_COUNT38_{kind}/v2",
                                  schema_version=2, **values))


def validate(value, kind):
    validate_content_hash(value, expected_contract=f"JC2_LITERATURE_PROXY_COUNT38_{kind}/v2",
                          expected_schema_version=2)
    if value.get("schema_version") != 2:
        raise ValueError("Count38 schema differs")
