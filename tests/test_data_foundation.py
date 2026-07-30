from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.data.identity import (
    FileRecord,
    JetIdentity,
    normalize_source_file,
)
from hlt_classification.data.root_reader import (
    _retry_root_operation,
    discover_file_records,
    load_offline_view,
)
from hlt_classification.data.schema import (
    CLASS_LABELS,
    FILENAME_PREFIX_TO_LABEL,
    LABEL_BRANCHES,
    MAX_CONSTITUENTS,
    RAW_FEATURE_BRANCHES,
    RAW_TOKEN_FEATURES,
    RAW_TOKEN_DIM,
    label_from_filename,
)
from hlt_classification.data.splits import (
    DEFAULT_BASE_SEED,
    DEFAULT_SPLIT_SEEDS,
    ROW_ORDERING_CONTRACT,
    SPLIT_ROLES,
    SplitManifest,
    audit_split_manifest,
    build_balanced_split_manifest,
    load_split_manifest,
    save_split_manifest,
)


PREFIXES_BY_LABEL = {
    label: prefix for prefix, label in FILENAME_PREFIX_TO_LABEL
}


def _particle_rows(branch: str, label: int, entries: int) -> ak.Array:
    rows: list[list[float]] = []
    for entry in range(entries):
        length = 130 if entry == 0 else entry % 3 + 1
        if branch == "part_px":
            values = [float(3 + entry)] + [0.0] * (length - 1)
        elif branch == "part_py":
            values = [4.0] + [0.0] * (length - 1)
        elif branch == "part_pz":
            values = [0.0] + [float(label + 1)] * (length - 1)
        elif branch == "part_energy":
            values = [float(10 + label + entry + index) for index in range(length)]
        elif branch == "part_charge":
            values = [float((index % 3) - 1) for index in range(length)]
        elif branch.startswith("part_is"):
            values = [float(index % 2) for index in range(length)]
        else:
            offset = RAW_FEATURE_BRANCHES.index(branch) / 100.0
            values = [float(label + entry + index) + offset for index in range(length)]
        rows.append(values)
    return ak.Array(rows)


def _write_root(
    path: Path,
    *,
    label: int,
    entries: int = 8,
    mismatch_branch: str | None = None,
    wrong_one_hot: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        branch: _particle_rows(branch, label, entries)
        for branch in RAW_FEATURE_BRANCHES
    }
    if mismatch_branch is not None:
        bad = ak.to_list(payload[mismatch_branch])
        bad[1] = bad[1][:-1]
        payload[mismatch_branch] = ak.Array(bad)
    for branch_label, branch in enumerate(LABEL_BRANCHES):
        values = np.zeros(entries, dtype=np.int8)
        if branch_label == label:
            values[:] = 1
        payload[branch] = values
    if wrong_one_hot:
        payload[LABEL_BRANCHES[label]] = np.zeros(entries, dtype=np.int8)
        payload[LABEL_BRANCHES[(label + 1) % len(CLASS_LABELS)]] = np.ones(
            entries, dtype=np.int8
        )
    with uproot.recreate(path) as root_file:
        root_file["tree"] = payload


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for label in range(len(CLASS_LABELS)):
        nested = root / f"part_{label % 2}" / f"class_{label}"
        _write_root(
            nested / f"{PREFIXES_BY_LABEL[label]}_fixture.root",
            label=label,
        )
    return root


def _tiny_records(entries: int = 20) -> tuple[FileRecord, ...]:
    return tuple(
        FileRecord(
            file=f"nested/{PREFIXES_BY_LABEL[label]}_sample.root",
            label=label,
            num_entries=entries,
        )
        for label in range(len(CLASS_LABELS))
    )


def _tiny_sizes(value: int = 10) -> dict[str, int]:
    return {role: value for role in SPLIT_ROLES}


def test_frozen_schema_and_longest_prefix_matching() -> None:
    assert CLASS_LABELS == (
        "QCD", "Hbb", "Hcc", "Hgg", "H4q",
        "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
    )
    assert RAW_FEATURE_BRANCHES == (
        "part_px", "part_py", "part_pz", "part_energy", "part_charge",
        "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
        "part_isElectron", "part_isMuon", "part_d0val", "part_d0err",
        "part_dzval", "part_dzerr",
    )
    assert RAW_TOKEN_DIM == 14
    assert RAW_TOKEN_FEATURES == (
        "pt", "eta", "phi", "energy", "charge", "isChargedHadron",
        "isNeutralHadron", "isPhoton", "isElectron", "isMuon", "d0val",
        "d0err", "dzval", "dzerr",
    )
    assert MAX_CONSTITUENTS == 128
    assert label_from_filename("TTBarLep_42.root") == 9
    assert label_from_filename("TTBar_42.root") == 8
    with pytest.raises(ValueError, match="unrecognized"):
        label_from_filename("mystery.root")


def test_canonical_identity_binds_label_but_leakage_key_does_not() -> None:
    identity = JetIdentity(r"part0\HToBB.root", 7, 1)
    relabeled = JetIdentity("part0/HToBB.root", 7, 2)
    assert identity.file == "part0/HToBB.root"
    assert identity.key != relabeled.key
    assert identity.location_key == relabeled.location_key
    assert len(identity.sha256) == 64
    assert normalize_source_file("x/./y.root") == "x/y.root"
    for invalid in ("", "../x.root", "/absolute.root", r"C:\absolute.root"):
        with pytest.raises(ValueError):
            normalize_source_file(invalid)


def test_recursive_discovery_is_deterministic_and_strict(
    synthetic_root: Path,
) -> None:
    first = discover_file_records(synthetic_root)
    second = discover_file_records(synthetic_root)
    assert first == second
    assert len(first) == len(CLASS_LABELS)
    assert [record.label for record in first] == list(range(len(CLASS_LABELS)))
    assert all(not Path(record.file).is_absolute() for record in first)
    assert all(record.num_entries == 8 for record in first)


def test_discovery_fails_closed_on_missing_branch(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = root / "ZJetsToNuNu_bad.root"
    _write_root(path, label=0)
    with uproot.open(path) as source:
        arrays = source["tree"].arrays(library="ak")
    payload = {field: arrays[field] for field in arrays.fields if field != "part_dzerr"}
    with uproot.recreate(path) as destination:
        destination["tree"] = payload
    with pytest.raises(RuntimeError, match="invalid JetClass ROOT"):
        discover_file_records(root, require_all_classes=False)
    with pytest.raises(ValueError, match="diagnostic_only"):
        discover_file_records(
            root,
            require_all_classes=False,
            skip_unreadable=True,
        )


def test_root_retry_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def transient() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary")
        return "ok"

    monkeypatch.setattr("hlt_classification.data.root_reader.time.sleep", lambda _: None)
    assert _retry_root_operation(
        transient, description="fixture", attempts=3, retry_seconds=0
    ) == "ok"
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        _retry_root_operation(
            lambda: (_ for _ in ()).throw(OSError("persistent")),
            description="fixture",
            attempts=2,
            retry_seconds=0,
        )


def test_chunked_root_read_restores_requested_order_and_schema(
    synthetic_root: Path,
) -> None:
    relative = "part_0/class_0/ZJetsToNuNu_fixture.root"
    identities = (
        JetIdentity(relative, 5, 0),
        JetIdentity(relative, 1, 0),
        JetIdentity(relative, 4, 0),
    )
    view = load_offline_view(
        identities,
        data_root=synthetic_root,
        max_constituents=4,
        read_chunk_size=2,
    )
    assert view.identities == identities
    assert view.tokens.shape == (3, 4, 14)
    assert view.tokens.dtype == np.float32
    assert view.mask.dtype == np.bool_
    assert view.labels.dtype == np.int64
    assert view.stats.chunks_read == 2
    assert view.stats.files_read == 1
    np.testing.assert_allclose(
        view.tokens[:, 0, 0],
        [np.hypot(8, 4), np.hypot(4, 4), np.hypot(7, 4)],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(view.mask.sum(axis=1), [3, 2, 2])
    assert np.all(view.tokens[~view.mask] == 0)
    np.testing.assert_allclose(
        view.tokens[1, 0],
        [
            np.hypot(4, 4), np.float32(0), np.pi / 4, 11, -1,
            0, 0, 0, 0, 0, 1.10, 1.11, 1.12, 1.13,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    # The second constituent has pt=0, so eta is deterministically zero.
    assert view.tokens[1, 1, 0] == 0
    assert view.tokens[1, 1, 1] == 0
    truncated = load_offline_view(
        (JetIdentity(relative, 0, 0),),
        data_root=synthetic_root,
        max_constituents=MAX_CONSTITUENTS,
        read_chunk_size=2,
    )
    assert truncated.mask.sum() == MAX_CONSTITUENTS


def test_reader_rejects_length_mismatch_and_wrong_one_hot(tmp_path: Path) -> None:
    root = tmp_path / "data"
    mismatch = root / "ZJetsToNuNu_mismatch.root"
    _write_root(mismatch, label=0, mismatch_branch="part_dzerr")
    with pytest.raises(RuntimeError, match="length mismatch"):
        load_offline_view(
            (JetIdentity(mismatch.name, 1, 0),),
            data_root=root,
            read_chunk_size=2,
        )
    wrong = root / "HToBB_wrong.root"
    _write_root(wrong, label=1, wrong_one_hot=True)
    with pytest.raises(RuntimeError, match="one-hot label mismatch"):
        load_offline_view(
            (JetIdentity(wrong.name, 0, 1),),
            data_root=root,
            read_chunk_size=2,
        )


def test_balanced_split_algorithm_is_deterministic_disjoint_and_mixed() -> None:
    records = _tiny_records()
    sizes = _tiny_sizes()
    first = build_balanced_split_manifest(
        records,
        data_root="/fixture",
        split_sizes=sizes,
    )
    second = build_balanced_split_manifest(
        tuple(reversed(records)),
        data_root="/fixture",
        split_sizes=sizes,
    )
    assert first.to_dict() == second.to_dict()
    assert first.payload_without_hash()["row_ordering_contract"] == ROW_ORDERING_CONTRACT
    audit = audit_split_manifest(first)
    assert audit["ok"]
    locations: set[str] = set()
    for role in SPLIT_ROLES:
        identities = first.splits[role]
        assert len(identities) == 10
        assert sorted(identity.label for identity in identities) == list(range(10))
        assert [identity.label for identity in identities] != list(range(10))
        assert not locations.intersection(identity.location_key for identity in identities)
        locations.update(identity.location_key for identity in identities)


def test_split_selection_matches_frozen_per_role_rng() -> None:
    records = _tiny_records()
    manifest = build_balanced_split_manifest(
        records,
        data_root="/fixture",
        split_sizes=_tiny_sizes(),
    )
    remaining = np.arange(20, dtype=np.int64)
    expected: dict[str, int] = {}
    for role in SPLIT_ROLES:
        rng = np.random.RandomState(DEFAULT_SPLIT_SEEDS[role])
        position = int(rng.choice(len(remaining), size=1, replace=False)[0])
        expected[role] = int(remaining[position])
        remaining = np.delete(remaining, position)
    actual = {
        role: next(identity.entry for identity in manifest.splits[role] if identity.label == 0)
        for role in SPLIT_ROLES
    }
    assert actual == expected
    assert manifest.base_seed == DEFAULT_BASE_SEED


def test_split_validation_rejects_capacity_sizes_and_bad_identity() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_balanced_split_manifest(
            _tiny_records(), data_root="/fixture", split_sizes={**_tiny_sizes(), "model_val": 11}
        )
    with pytest.raises(ValueError, match="are required"):
        build_balanced_split_manifest(
            _tiny_records(entries=4), data_root="/fixture", split_sizes=_tiny_sizes()
        )
    good = build_balanced_split_manifest(
        _tiny_records(), data_root="/fixture", split_sizes=_tiny_sizes()
    )
    bad_splits = {role: list(good.splits[role]) for role in SPLIT_ROLES}
    original = bad_splits["model_train"][0]
    bad_splits["model_train"][0] = JetIdentity(
        original.file, 999, original.label
    )
    bad = SplitManifest.create(
        data_root=good.data_root,
        tree_name=good.tree_name,
        max_constituents=good.max_constituents,
        files=good.files,
        split_sizes=good.split_sizes,
        split_seeds=good.split_seeds,
        base_seed=good.base_seed,
        splits=bad_splits,
    )
    assert not audit_split_manifest(bad)["ok"]


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_manifest_roundtrip_authentication_and_immutable_publish(
    tmp_path: Path,
    suffix: str,
) -> None:
    manifest = build_balanced_split_manifest(
        _tiny_records(), data_root="/fixture", split_sizes=_tiny_sizes()
    )
    path = tmp_path / f"manifest{suffix}"
    assert save_split_manifest(manifest, path) == manifest.content_hash
    assert load_split_manifest(path).to_dict() == manifest.to_dict()
    assert save_split_manifest(manifest, path) == manifest.content_hash

    different = build_balanced_split_manifest(
        _tiny_records(entries=21), data_root="/fixture", split_sizes=_tiny_sizes()
    )
    with pytest.raises(FileExistsError, match="immutable"):
        save_split_manifest(different, path)


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    manifest = build_balanced_split_manifest(
        _tiny_records(), data_root="/fixture", split_sizes=_tiny_sizes()
    )
    payload = copy.deepcopy(manifest.to_dict())
    payload["splits"]["model_train"][0]["entry"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_split_manifest(path)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_and_build_clis(
    synthetic_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight = _load_script("preflight_data.py")
    assert preflight.main(
        ["--data-root", str(synthetic_root), "--required-per-class", "8"]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["classes"]["QCD"]["available_events"] == 8
    assert preflight.main(
        ["--data-root", str(synthetic_root), "--required-per-class", "9"]
    ) == 2
    assert not json.loads(capsys.readouterr().out)["ok"]

    builder = _load_script("build_splits.py")
    output = tmp_path / "split_manifest.json.gz"
    arguments = ["--data-root", str(synthetic_root), "--output", str(output)]
    for role in SPLIT_ROLES:
        arguments.extend([f"--{role.replace('_', '-')}", "10"])
    assert builder.main(arguments) == 0
    build_report = json.loads(capsys.readouterr().out)
    loaded = load_split_manifest(output)
    assert build_report["manifest_content_hash"] == loaded.content_hash
    assert audit_split_manifest(loaded)["ok"]
