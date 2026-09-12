"""Record actual implementation bytes, without confusing a dirty tree with HEAD."""
from pathlib import Path
import subprocess

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash, validate_content_hash


def source_record(*files: str) -> dict:
    project = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return with_content_hash(dict(
        contract="JETCLASS2_DELPHES_IMPLEMENTATION_SOURCE/v1", schema_version=1,
        git_head=result.stdout.strip() if result.returncode == 0 else None,
        commit_alone_is_not_clean_source_proof=True,
        file_sha256={name: sha256_file(project / name) for name in files},
    ))


def validate_source_record(value: dict) -> None:
    validate_content_hash(value, expected_contract="JETCLASS2_DELPHES_IMPLEMENTATION_SOURCE/v1", expected_schema_version=1)
    if not value["file_sha256"] or value["commit_alone_is_not_clean_source_proof"] is not True:
        raise ValueError("Incomplete producer code provenance")
