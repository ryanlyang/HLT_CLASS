from __future__ import annotations

import hashlib
import numpy as np
import pytest

from hlt_classification.scouting.audit import audit_source_inventory, authorize_p4_closure
from hlt_classification.scouting.assignment import EphemeralAssignmentTable, build_even_source_ordinals
from hlt_classification.scouting.contracts import artifact_envelope, require_role_access, validate_artifact
from hlt_classification.scouting.identity import ScoutingJetIdentity, normalize_source_path, reject_case_aliases
from hlt_classification.scouting.inputs import ParticleInputs, build_hlt_inputs
from hlt_classification.scouting.dataset import _concat_batches, _slice_batch
from hlt_classification.scouting.labels import baseline_mask, multiclass_labels
from hlt_classification.scouting.schema import CLASS_NAMES, HLT_FEATURE_SPECS
from hlt_classification.scouting.splits import SourceFileRecord, build_split_manifest, validate_split_manifest
from hlt_classification.scouting.streaming import iterate_projected_chunks, partition_files
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
            SourceFileRecord(
                f"{stratum}/{i:02d}.root", stratum, 1000 + i,
                _digest(f"{stratum}-{i}"), 75, (5,) * 15,
            )
            for i in range(count)
        )
    source = _digest("source")
    first = build_split_manifest(records, source_manifest_sha256=source)
    second = build_split_manifest(reversed(records), source_manifest_sha256=source)
    assert first == second
    assert [first["roles"][role]["file_count"] for role in ("train", "validation", "final_test")] == [32, 11, 10]
    assert first["fractions"] == [0.6, 0.2, 0.2]
    assert first["maximum_observed_class_fraction_deviation"] <= .02
    assert validate_split_manifest(first, source_manifest_sha256=source, expected_inventory=records) == first["content_hash"]


def test_source_audit_streams_per_file_class_counts_for_stratification(tmp_path):
    import uproot

    sample = tmp_path / "signal"; sample.mkdir()
    source = sample / "part.root"
    with uproot.recreate(source) as output:
        output["tree"] = {
            "jet_tightId": np.ones(4, np.int32),
            "jet_no": np.zeros(4, np.int32),
            "scoutfj_pt": np.full(4, 300, np.float32),
            "scoutfj_sdmass": np.full(4, 100, np.float32),
            "n_scoutpfcands": np.ones(4, np.int32),
            "scoutfj_label": np.array([0, 17, 20, 999], np.int32),
            "scoutfj_gen_pid": np.array([0, 0, -37, 0], np.int32),
            "fj_label": np.array([309, 0, 0, 0], np.int32),
        }
    report = audit_source_inventory(tmp_path, (source,), strict_reference=False)
    assert report["schema_version"] == 2
    assert report["baseline_selected_entries"] == 4
    assert report["mapped_entries"] == 3
    assert report["class_counts"] == [1, 1] + [0] * 12 + [1]
    assert report["files"][0]["class_counts"] == report["class_counts"]


def test_projected_training_reads_round_robin_across_bounded_source_group(tmp_path):
    import uproot

    files = []
    for name, offset in (("a.root", 0), ("b.root", 10)):
        path = tmp_path / name
        with uproot.recreate(path) as output:
            output["tree"] = {"value": np.arange(offset, offset + 4, dtype=np.float32)}
        files.append(path)
    chunks = list(iterate_projected_chunks(
        files, ("value",), data_root=tmp_path, role="train",
        step_size=2, interleave_files=2,
    ))
    assert [chunk.source_path for chunk in chunks] == [
        "a.root", "b.root", "a.root", "b.root",
    ]


def test_split_fails_closed_when_whole_files_cannot_preserve_class_fractions():
    records = []
    for index in range(5):
        counts = [0] * 15
        counts[0] = 100 if index == 0 else 0
        for class_index in range(1, 15):
            counts[class_index] = 20
        records.append(SourceFileRecord(
            f"sample/{index}.root", "sample", 500, _digest(f"imbalanced-{index}"),
            sum(counts), tuple(counts),
        ))
    manifest = build_split_manifest(records, source_manifest_sha256=_digest("imbalanced-source"))
    assert manifest["maximum_observed_class_fraction_deviation"] > .02
    with pytest.raises(ValueError, match="cannot meet class-balance tolerance"):
        validate_split_manifest(
            manifest, source_manifest_sha256=_digest("imbalanced-source"),
            expected_inventory=records,
        )


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


def test_ram_assignment_join_and_cross_chunk_batch_packing():
    assignments = np.full((2, 200), -1, np.int16)
    assignments[0, 0] = 4; assignments[1, 0] = 7
    table = EphemeralAssignmentTable.create(
        ("a", "b"), assignments, parents={"matcher": _digest("matcher")},
        matcher_id=_digest("model"), threshold=.99, matcher_variant="M5",
        eligible_categories=(0, 1, 2, 3, 4),
    )
    assert table.join(("b", "a"))[:, 0].tolist() == [7, 4]
    with pytest.raises(KeyError): table.join(("missing",))

    def part(rows, offset):
        return {
            "labels": np.arange(offset, offset + rows),
            "identity_keys": np.asarray([f"k{index}" for index in range(offset, offset + rows)]),
            "hlt": ParticleInputs(
                np.full((rows, 21, 2), offset, np.float32),
                np.zeros((rows, 4, 2), np.float32),
                np.ones((rows, 1, 2), np.bool_), np.full(rows, 2, np.int32),
            ),
        }
    packed = _concat_batches((part(2, 0), part(3, 2)))
    full = _slice_batch(packed, 0, 4); tail = _slice_batch(packed, 4, 5)
    assert full["labels"].tolist() == [0, 1, 2, 3]
    assert tail["labels"].tolist() == [4]


def test_bounded_ram_assignment_miniature_never_scans_the_full_role(monkeypatch):
    import hlt_classification.scouting.pmard_stream as pmard_stream

    record = SourceFileRecord(
        "sample/a.root", "sample", 10, _digest("source"), 10,
        tuple([10] + [0] * 14),
    )
    monkeypatch.setattr(pmard_stream, "role_records", lambda *_args, **_kwargs: (record,))
    observed = {}

    def bounded_batches(*_args, assignment_collector, max_rows, **_kwargs):
        observed["max_rows"] = max_rows
        assignment_collector.append((
            np.asarray([f"jet-{index}" for index in range(max_rows)]),
            np.full((max_rows, 200), -1, np.int16),
        ))
        if False:
            yield None

    monkeypatch.setattr(pmard_stream, "iterate_pmard_batches", bounded_batches)
    table = pmard_stream.build_ephemeral_assignment_table(
        {}, data_root=".", role="train", matcher_model=object(),
        matcher_id=_digest("matcher"), parents={"split": _digest("split")},
        max_rows=3,
    )
    assert observed["max_rows"] == 3
    assert table.header["rows"] == 3


def test_source_balanced_matcher_ordinals_are_deterministic_and_spread():
    records = tuple(
        SourceFileRecord(
            f"sample/f{index}.root", "sample", 100, _digest(str(index)), 100,
            tuple([100] + [0] * 14),
        ) for index in range(3)
    )
    targets = build_even_source_ordinals(records, total_rows=8)
    assert [len(targets[record.path]) for record in sorted(records)] == [3, 3, 2]
    assert targets[records[0].path].tolist() == [16, 50, 83]


def test_p4_closure_gate_selects_best_formula_and_fails_above_tolerance():
    report = authorize_p4_closure({
        "objects": 10000, "unweighted_failures": 4, "weighted_failures": 1,
    })
    assert report["selected_formula"] == "puppiw_times_unweighted"
    assert report["selected_failure_fraction"] == 1e-4
    with pytest.raises(ValueError):
        authorize_p4_closure({"objects": 9999, "unweighted_failures": 1})
