from __future__ import annotations

import hashlib
import numpy as np
import pytest

from hlt_classification.scouting.contracts import artifact_envelope, require_role_access, validate_artifact
from hlt_classification.scouting.identity import ScoutingJetIdentity, normalize_source_path, reject_case_aliases
from hlt_classification.scouting.inputs import build_hlt_inputs
from hlt_classification.scouting.labels import baseline_mask, multiclass_labels
from hlt_classification.scouting.schema import CLASS_NAMES, HLT_FEATURE_SPECS
from hlt_classification.scouting.splits import SourceFileRecord, build_split_manifest, validate_split_manifest
from hlt_classification.scouting.streaming import partition_files
from hlt_classification.scouting.targets import EphemeralTeacherTargets, validate_ram_root


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_exact_class_order_selection_and_labels():
    assert len(CLASS_NAMES) == 15
    arrays = {
        "jet_tightId": np.ones(4), "jet_no": np.array([0, 1, 2, 0]),
        "scoutfj_pt": np.array([300, 400, 300, 300]),
        "scoutfj_sdmass": np.array([100, 100, 100, -1]),
        "n_scoutpfcands": np.ones(4),
        "fj_label": np.array([309, 0, 0, 0]),
        "scoutfj_label": np.array([0, 17, 20, 999]),
        "scoutfj_gen_pid": np.array([0, 0, -37, 0]),
    }
    assert baseline_mask(arrays).tolist() == [True, True, False, False]
    assert multiclass_labels(arrays).tolist() == [0, 1, 14, -1]


def test_identity_is_label_free_portable_and_case_safe():
    identity = ScoutingJetIdentity(r"QCD\part.root", 7)
    assert identity.key == "QCD/part.root::tree::7"
    assert normalize_source_path("a/./b.root") == "a/b.root"
    with pytest.raises(ValueError):
        normalize_source_path("../escape.root")
    with pytest.raises(ValueError):
        reject_case_aliases(("A/x.root", "a/x.root"))


def test_split_is_stratified_disjoint_authenticated_and_reference_sized():
    records = []
    for stratum, count in (("signal", 25), ("qcd", 28)):
        records.extend(
            SourceFileRecord(f"{stratum}/{i:02d}.root", stratum, 100 + i, _digest(f"{stratum}-{i}"))
            for i in range(count)
        )
    source = _digest("source")
    first = build_split_manifest(records, source_manifest_sha256=source)
    second = build_split_manifest(reversed(records), source_manifest_sha256=source)
    assert first == second
    assert [first["roles"][role]["file_count"] for role in ("train", "validation", "final_test")] == [42, 6, 5]
    assert validate_split_manifest(first, source_manifest_sha256=source, expected_inventory=records) == first["content_hash"]


def test_hlt_builder_exact_order_padding_and_nonfinite_replacement():
    arrays = {}
    for index, spec in enumerate(HLT_FEATURE_SPECS):
        arrays[spec.branch] = [np.array([spec.median, np.nan, spec.median + 1], np.float32)]
    for branch in ("scoutpfcand_px", "scoutpfcand_py", "scoutpfcand_pz"):
        arrays[branch] = [np.array([1, 2, 3], np.float32)]
    arrays["scoutpfcand_energy"] = [np.array([2, 3, 4], np.float32)]
    built = build_hlt_inputs(arrays)
    assert built.features.shape == (1, 21, 200)
    assert built.vectors.shape == (1, 4, 200)
    assert built.mask[0, 0, :3].all() and not built.mask[0, 0, 3:].any()
    assert built.features[0, 0, 0] == 0
    assert built.features[0, 0, 1] == (0 - HLT_FEATURE_SPECS[0].median) * HLT_FEATURE_SPECS[0].factor
    assert not built.features[:, :, 3:].any()


def test_role_seal_lineage_and_partitioning():
    with pytest.raises(PermissionError):
        require_role_access("final_test", branch_read=True)
    require_role_access("final_test", branch_read=True, completed_locks=("finalist", "execution"))
    with pytest.raises(PermissionError):
        require_role_access("validation", fit=True)
    parent = {"source": _digest("source")}
    artifact = artifact_envelope(kind="data_lock", payload={"ok": True}, parents=parent)
    assert validate_artifact(artifact, expected_kind="data_lock", expected_parents=parent) == artifact["content_hash"]
    pieces = [partition_files(tuple(range(20)), rank=r, world_size=2, worker_id=w, num_workers=2) for r in range(2) for w in range(2)]
    assert sorted(item for piece in pieces for item in piece) == list(range(20))


def test_ram_teacher_targets_join_by_identity_and_reject_project_path(tmp_path):
    table = EphemeralTeacherTargets.create(
        ("a", "b"), np.arange(30, dtype=np.float16).reshape(2, 15),
        teacher_report_sha256=_digest("teacher"), split_manifest_sha256=_digest("split"),
    )
    assert table.join(("b", "a")).shape == (2, 15)
    with pytest.raises(KeyError): table.join(("missing",))
    project = tmp_path / "project"; data = tmp_path / "data"; ram = tmp_path / "ram"
    for path in (project, data, ram): path.mkdir()
    assert validate_ram_root(ram, project_root=project, data_root=data, allowed_roots=(ram,)) == ram.resolve()
    with pytest.raises(ValueError):
        validate_ram_root(project / "tmp", project_root=project, data_root=data, allowed_roots=(project,))
