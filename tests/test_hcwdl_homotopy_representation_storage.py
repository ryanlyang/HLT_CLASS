from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (
    RESUME_RETIREMENT_AUTHORIZATION_CONTRACT,
    RESUME_RETIREMENT_COMPLETION_CONTRACT,
    RESUME_RETIREMENT_PHRASE,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_storage import (
    build_resume_retirement_authorization,
    execute_resume_retirement,
    validate_retirement_artifact,
)
import hlt_classification.scouting.hcwdl_homotopy_representation_storage as storage


H = "a" * 64


def _authorization(tmp_path: Path):
    files = []
    for sequence in (59, 60):
        for kind, name in (
            ("state", f"state_{sequence}.pt"),
            ("sidecar", f"state_{sequence}.json"),
            ("commit", f"commit_{sequence}.json"),
        ):
            path = tmp_path / "resume" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{kind}-{sequence}".encode())
            files.append({
                "node_id": "F_RSET_U020",
                "sequence": sequence,
                "kind": kind,
                "path": str(path.resolve()),
                "relative_path": f"resume/{name}",
                "byte_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return with_content_hash({
        "contract": RESUME_RETIREMENT_AUTHORIZATION_CONTRACT,
        "schema_version": 1,
        "parents": {"campaign_spec": H},
        "authorization_phrase": RESUME_RETIREMENT_PHRASE,
        "campaign_root": str(tmp_path.resolve()),
        "nodes": [{"node_id": "F_RSET_U020"}],
        "files": files,
        "retired_bytes": sum(row["bytes"] for row in files),
        "selected_and_final_checkpoints_retained": True,
        "reports_retained": True,
        "final_test_accessed": False,
    })


def test_completed_resume_retirement_is_exact_and_idempotent(tmp_path):
    authorization = _authorization(tmp_path)
    completion = execute_resume_retirement(
        authorization, authorization_phrase=RESUME_RETIREMENT_PHRASE,
    )
    assert completion["contract"] == RESUME_RETIREMENT_COMPLETION_CONTRACT
    assert completion["retired_member_count"] == 6
    assert completion["all_authorized_paths_absent"] is True
    assert all(not Path(row["path"]).exists() for row in authorization["files"])
    repeated = execute_resume_retirement(
        authorization, authorization_phrase=RESUME_RETIREMENT_PHRASE,
    )
    assert repeated == completion
    validate_retirement_artifact(
        completion, contract=RESUME_RETIREMENT_COMPLETION_CONTRACT,
    )


def test_resume_retirement_rejects_changed_member(tmp_path):
    authorization = _authorization(tmp_path)
    Path(authorization["files"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after authorization"):
        execute_resume_retirement(
            authorization, authorization_phrase=RESUME_RETIREMENT_PHRASE,
        )
    assert any(Path(row["path"]).exists() for row in authorization["files"])


def test_resume_retirement_requires_exact_phrase(tmp_path):
    authorization = _authorization(tmp_path)
    with pytest.raises(PermissionError, match="phrase differs"):
        execute_resume_retirement(authorization, authorization_phrase="no")
    assert all(Path(row["path"]).exists() for row in authorization["files"])


def test_resume_retirement_rejects_path_outside_campaign(tmp_path):
    authorization = _authorization(tmp_path)
    outside = tmp_path.parent / "outside.pt"
    outside.write_bytes(b"outside")
    authorization["files"][0].update({
        "path": str(outside.resolve()),
        "relative_path": "../outside.pt",
        "byte_sha256": sha256_file(outside),
    })
    authorization.pop("content_hash")
    authorization = with_content_hash(authorization)
    with pytest.raises(PermissionError, match="escaped campaign root"):
        execute_resume_retirement(
            authorization, authorization_phrase=RESUME_RETIREMENT_PHRASE,
        )
    assert outside.is_file()


def test_authorization_requires_complete_direct_consumer(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    node = root / "training/HCWDL_REP_SET/v1/F_RSET_U020"
    consumer = root / "training/HCWDL_REP_SET/v1/F_RSET_U040"
    node.mkdir(parents=True)
    consumer.mkdir(parents=True)
    report_hash = "b" * 64
    wrapper_hash = "c" * 64
    consumer_hash = "d" * 64
    (node / "training_report.json").write_text(
        '{"complete":true,"scientific_complete":true,'
        '"completed_natural_population_passes":60}'
    )
    (node / "combined_training_report.json").write_text(
        '{"node_id":"F_RSET_U020","completed_passes":60,'
        f'"parents":{{"engine_report":"{report_hash}"}}}}'
    )
    (consumer / "combined_training_report.json").write_text(
        '{"node_id":"F_RSET_U040","completed_passes":60,"parents":{}}'
    )
    monkeypatch.setattr(storage, "validate_campaign", lambda *_args, **_kwargs: "a" * 64)
    monkeypatch.setattr(
        storage, "validate_representation_training_report",
        lambda *_args, **_kwargs: report_hash,
    )
    monkeypatch.setattr(
        storage, "validate_artifact",
        lambda value, **_kwargs: (
            wrapper_hash if value["node_id"] == "F_RSET_U020" else consumer_hash
        ),
    )
    monkeypatch.setattr(
        storage, "_validated_envelope",
        lambda _report, _output, kind: ("e" if kind == "selected" else "f") * 64,
    )
    monkeypatch.setattr(
        storage, "_resume_members",
        lambda *_args, **_kwargs: [{
            "node_id": "F_RSET_U020", "sequence": 60, "kind": "state",
            "path": str(node / "resume/state_60.pt"),
            "relative_path": "training/HCWDL_REP_SET/v1/F_RSET_U020/resume/state_60.pt",
            "byte_sha256": "1" * 64, "bytes": 10,
        }],
    )
    authorization = build_resume_retirement_authorization(
        {
            "campaign_root": str(root),
            "representation_recipe_sha256": "9" * 64,
        },
        node_ids=["F_RSET_U020"], source_commit="8" * 40,
    )
    assert authorization["nodes"][0]["completed_consumers"] == {
        "F_RSET_U040": consumer_hash,
    }
    assert authorization["retired_bytes"] == 10
