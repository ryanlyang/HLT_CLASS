"""Frozen nested subsets: native synthetic ROOT metadata, no test particles."""
import base64
import copy
from pathlib import Path
import shutil

import numpy as np
import pytest
import uproot

from test_jetclass2_delphes import snapshot
from hlt_classification.data.cache_contracts import load_json, with_content_hash
from hlt_classification.jetclass2_delphes import production
from hlt_classification.jetclass2_delphes.cache import prepare_cache
from hlt_classification.jetclass2_delphes.campaign import build_campaign_plan
from hlt_classification.jetclass2_delphes.contracts import row_identity
from hlt_classification.jetclass2_delphes.foundation import (
    build_foundation_spec, validate_foundation_spec, build_assignment_shard, load_assignments,
)
from hlt_classification.jetclass2_delphes.reader import DatasetReader
from hlt_classification.jetclass2_delphes.split_registry import (
    build_design, build_registry, select_profile, validate_registry, validate_split_profile,
    nested_quotas, pack_entries, unpack_entries, publish_registry,
)
from hlt_classification.jetclass2_delphes.splits import validate_splits


@pytest.fixture
def registered(snapshot):
    data, inventory, reservoirs = snapshot
    registry = build_registry(data, inventory, reservoirs, training_sizes=(11, 22, 33, 44),
                              validation_size=11, test_size=11, step_size=3)
    return data, inventory, reservoirs, registry


def test_exact_nested_profiles_and_shared_evaluation(registered):
    data, inventory, reservoirs, registry = registered
    assert validate_registry(registry, inventory) == registry["content_hash"]
    previous = set()
    eval_hashes, eval_ids = set(), []
    for size in (11, 22, 33, 44):
        profile = select_profile(registry, inventory, f"TRAIN_{size}")
        assert validate_splits(profile, inventory) == profile["content_hash"]
        assert profile["groups"] == reservoirs["groups"]
        assert profile["role_counts"] == {"train": size, "validation": 11, "final_test": 11}
        assert sum(profile["role_class_counts"]["train"]) == size
        train = list(DatasetReader(data, inventory, profile, role="train", step_size=5))
        ids = {jet.identity for jet in train}
        assert len(ids) == size and previous <= ids
        previous = ids
        assert np.bincount([j.label for j in train], minlength=11).tolist() == profile["role_class_counts"]["train"]
        evaluation = list(DatasetReader(data, inventory, profile, role="validation", step_size=7))
        eval_ids.append([j.identity for j in evaluation])
        assert not set(eval_ids[-1]) & ids
        eval_hashes.add(tuple(profile["memberships"][r]["content_hash"] for r in ("validation", "final_test")))
        # Metadata membership identities can be inspected without opening any
        # final-test particle branches or supplying a test-reader capability.
        test_ids = set()
        sources = {f["path"]: f for f in inventory["files"]}
        for member in profile["memberships"]["final_test"]["files"]:
            source = sources[member["path"]]
            for entry in unpack_entries(member["entry_mask"], source["entries"]):
                test_ids.add(row_identity(inventory["content_hash"], source["path"], source["tree_key"], int(entry)))
        assert len(test_ids) == 11 and not test_ids & (ids | set(eval_ids[-1]))
        with pytest.raises(PermissionError, match="final_test"):
            DatasetReader(data, inventory, profile, role="final_test")
    assert len(eval_hashes) == 1
    assert all(x == eval_ids[0] for x in eval_ids)


def test_registry_replay_relocation_and_metadata_only(registered, tmp_path, monkeypatch):
    data, inventory, reservoirs, registry = registered
    copied = tmp_path / "relocated"
    shutil.copytree(data, copied)
    arrays = uproot.behaviors.TBranch.HasBranches.arrays
    requested = []
    def metadata_only(self, expressions=None, *args, **kwargs):
        requested.append(expressions)
        assert expressions == ["jet_label", "hlt_matched"]
        return arrays(self, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", metadata_only)
    replay = build_registry(copied, inventory, reservoirs, training_sizes=(11, 22, 33, 44),
                            validation_size=11, test_size=11, step_size=7)
    assert requested and registry == replay
    changed = build_registry(copied, inventory, reservoirs, training_sizes=(11, 22, 33, 44),
                             validation_size=11, test_size=11, seed=20260912)
    assert changed["content_hash"] != registry["content_hash"]
    assert changed["memberships"]["TRAIN_11"]["files"] != registry["memberships"]["TRAIN_11"]["files"]


def test_design_capacity_and_monotonic_quota_allocation(snapshot):
    _, inventory, reservoirs = snapshot
    with pytest.raises(ValueError, match="capacity"):
        build_design(inventory, reservoirs)  # never silently downsize on tiny data
    for sizes in ((11, 11), (22, 11), (True, 22), (10, 22)):
        with pytest.raises(ValueError):
            nested_quotas([100] * 11, sizes)
    caps = [4190225, 185106, 184905, 184948, 185420, 190199, 19980, 30823, 43453, 46018, 115030]
    sizes = (500000, 1000000, 1500000, 2000000)
    counts = nested_quotas(caps, sizes)
    for index, (size, row) in enumerate(zip(sizes, counts)):
        assert sum(row) == size and all(0 < n <= c for n, c in zip(row, caps))
        assert np.max(np.abs(np.array(row) / size - np.array(caps) / sum(caps))) < 3e-5
        if index:
            assert all(a <= b for a, b in zip(counts[index - 1], row))


def test_mask_canonicality_and_invalid_padding():
    value = pack_entries(np.array([0, 3, 8]), 10)
    assert unpack_entries(value, 10).tolist() == [0, 3, 8]
    for entries in (np.array([3, 3]), np.array([-1]), np.array([10]), np.array([1.5])):
        with pytest.raises(ValueError):
            pack_entries(entries, 10)
    with pytest.raises(ValueError, match="padding"):
        unpack_entries(base64.b64encode(bytes([0, 128])).decode(), 10)
    with pytest.raises(ValueError):
        unpack_entries("!!!!", 10)


def test_reader_rejects_ineligible_membership_and_keeps_hlt_only_capability(registered, monkeypatch):
    data, inventory, _, registry = registered
    profile = select_profile(registry, inventory, "TRAIN_22")
    original = uproot.behaviors.TBranch.HasBranches.arrays
    requested = []
    def hlt_only(self, expressions=None, *args, **kwargs):
        requested.extend(expressions)
        assert not any(name.startswith("part_") or name == "jet_nparticles" for name in expressions)
        return original(self, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", hlt_only)
    assert len(list(DatasetReader(data, inventory, profile, role="train"))) == 22
    assert requested
    bad = copy.deepcopy(profile)
    member = bad["memberships"]["train"]
    row = next(r for r in member["files"] if r["rows"] > 0)
    source = next(f for f in inventory["files"] if f["path"] == row["path"])
    entries = unpack_entries(row["entry_mask"], source["entries"])
    entries[0] = source["entries"] - 1  # fixture's unmatched dummy row
    row["entry_mask"] = pack_entries(entries, source["entries"])
    bad["memberships"]["train"] = with_content_hash(member)
    bad = with_content_hash(bad)
    with pytest.raises(ValueError, match="ineligible/unmatched"):
        list(DatasetReader(data, inventory, bad, role="train"))


def test_registry_tamper_nesting_and_role_escape(registered):
    _, inventory, _, registry = registered
    bad = copy.deepcopy(registry)
    small, large = (bad["memberships"][f"TRAIN_{n}"] for n in (11, 22))
    found = False
    for lower, upper in zip(small["files"], large["files"]):
        file = next(r for r in inventory["files"] if r["path"] == lower["path"])
        a = unpack_entries(lower["entry_mask"], file["entries"])
        b = unpack_entries(upper["entry_mask"], file["entries"])
        other = sorted(set(range(file["entries"])) - set(b))
        if len(a) and other:
            a[0] = other[0]
            lower["entry_mask"] = pack_entries(a, file["entries"])
            found = True
            break
    assert found
    bad["memberships"]["TRAIN_11"] = with_content_hash(small)
    with pytest.raises(ValueError, match="not nested"):
        validate_registry(with_content_hash(bad), inventory)
    profile = select_profile(registry, inventory, "TRAIN_22")
    bad = copy.deepcopy(profile)
    bad["groups"][0]["role"] = "final_test" if bad["groups"][0]["role"] != "final_test" else "train"
    with pytest.raises(ValueError):
        validate_split_profile(with_content_hash(bad), inventory)
    bad = copy.deepcopy(profile)
    bad["role_counts"]["train"] += 1
    with pytest.raises(ValueError, match="totals"):
        validate_split_profile(with_content_hash(bad), inventory)
    with pytest.raises(ValueError, match="Unregistered"):
        select_profile(registry, inventory, "TRAIN_FULL")


def test_subset_foundation_assignment_cache_and_cross_profile_rejection(registered, tmp_path):
    data, inventory, _, registry = registered
    small = build_foundation_spec(inventory, select_profile(registry, inventory, "TRAIN_11"))
    large = build_foundation_spec(inventory, select_profile(registry, inventory, "TRAIN_22"))
    assert small["contract"] == "JETCLASS2_DELPHES_FOUNDATION_SPEC/v2"
    assert validate_foundation_spec(small) == small["content_hash"]
    assert sum(t["rows"] for t in small["assignment_tasks"]) == 22
    assert all(t["role"] != "final_test" and t["rows"] > 0 for t in small["assignment_tasks"])
    for task in small["assignment_tasks"]:
        build_assignment_shard(small, data_root=data, output_root=tmp_path / "foundation", file_index=task["file_index"])
    common = next(t for t in small["assignment_tasks"] if t["role"] == "train")
    with pytest.raises(ValueError, match="lineage"):
        load_assignments(large, root=tmp_path / "foundation", file_index=common["file_index"])
    kwargs = dict(data_root=data, foundation_root=tmp_path / "foundation", role="train", max_ram_bytes=10**8)
    hlt = prepare_cache(small, coordinate_name="D000", workers=1, **kwargs)
    mixed = prepare_cache(small, coordinate_name="U050", workers=2, **kwargs)
    assert len(hlt) == len(mixed) == 11
    np.testing.assert_array_equal(hlt.identities, mixed.identities)
    np.testing.assert_array_equal(hlt.labels, mixed.labels)
    plan = build_campaign_plan(small)
    assert plan["contract"] == "JETCLASS2_DELPHES_CAMPAIGN_PLAN/v2"
    assert plan["role_counts"] == dict(train=11, validation=11, final_test=11)
    assert plan["split_registry_sha256"] == registry["content_hash"]
    assert plan["fresh_fit_count"] == 31 and not plan["imported_models"]


def test_profile_bundle_publication_and_legacy_production_refusal(registered, tmp_path, monkeypatch):
    data, inventory, reservoirs, registry = registered
    root = tmp_path / "registry"
    publish_registry(root, registry, inventory)
    assert load_json(root / "registry.json") == registry
    for p in registry["design"]["profiles"]:
        assert load_json(root / "profiles" / f"{p['name']}.json") == select_profile(registry, inventory, p["name"])
    with pytest.raises(FileExistsError):
        publish_registry(root, registry, inventory)
    monkeypatch.setattr(production, "_source", lambda *args: None)
    legacy = build_foundation_spec(inventory, reservoirs)
    assert legacy["contract"] == "JETCLASS2_DELPHES_FOUNDATION_SPEC/v1"
    validate_foundation_spec(legacy)
    with pytest.raises(ValueError, match="full-data reservoir"):
        production.create_campaign(legacy, foundation_root=tmp_path / "foundation", data_root=data,
                                   campaign_root=tmp_path / "no_campaign", project=tmp_path,
                                   source_commit="a" * 40, profile={})
    assert not (tmp_path / "no_campaign").exists()
