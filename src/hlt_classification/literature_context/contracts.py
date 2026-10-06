"""New contracts; never alias the frozen NOISE_V3 artifacts."""
from hlt_classification.literature_proxy.contracts import (
    atomic_publish_bytes, canonical_sha256, load_json, read_reference, reference,
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)


def artifact(kind, **values):
    return with_content_hash(dict(contract=f"JC2_LITERATURE_CONTEXT_{kind}/v1",
                                  schema_version=1, **values))


def validate(value, kind):
    return validate_content_hash(value, expected_contract=f"JC2_LITERATURE_CONTEXT_{kind}/v1",
                                 expected_schema_version=1)
