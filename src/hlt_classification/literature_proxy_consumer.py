"""Read frozen NOISE_V3 copies without rewriting producer paths or rerunning it.

The complete manifest is an explicit trust anchor. Only ordinary roles can be
decoded; final-test byte-copy verification is a separate transport operation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, require_sha256, sha256_file, with_content_hash,
)
from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.cms2jc2_response.readers import authenticated_open
from hlt_classification.jetclass2_delphes.inventory import latest_tree, validate_inventory
from hlt_classification.jetclass2_delphes.selection import selected_mask
from hlt_classification.jetclass2_delphes.split_registry import validate_split_profile
from hlt_classification.literature_proxy.population import BRANCHES, JC2_FIELDS, from_columns
from hlt_classification.literature_proxy_production import output, population
from hlt_classification.literature_proxy_production.contracts import safe, validate
from hlt_classification.literature_proxy_v2.kernel import validate_calibration
from hlt_classification.literature_proxy_v3.kernel import recipe
from hlt_classification.literature_proxy_v3.contracts import validate as validate_pilot


MANIFEST_SHA256 = "26c902d04988b3f32be98a8d2ccbed6d48dd8c3f49c71ddc2c660da0f2ec5deb"
PROXY_DIRECTORY = "jetclass2_literature_noise_v3_2250k_ebd5bc1a_r1"
SNAPSHOT_DIRECTORY = "jetclass2_10M_20260918_dzfix_partial_v1"
OSCAR_ROOT = Path("/oscar/home/rlyang/datasets/literature_noise_v3_2250k_ebd5bc1a_r1")
CONTRACT = "JC2_LITERATURE_RELOCATED_READER/v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _role(role):
    if role not in ("train", "validation"):
        raise PermissionError("Relocated reader exposes train/validation only; final test is sealed")


def _check_file(path, reference):
    if "bytes" in reference:
        _require(path.stat().st_size == reference["bytes"], f"Artifact size differs: {path}")
    _require(sha256_file(path) == reference["sha256"], f"Artifact checksum differs: {path}")
    return path


@dataclass(frozen=True)
class PairedJet:
    """Identity/path/entry/label are metadata or targets, never model features."""
    identity: str
    role: str
    source_file: str
    tree_key: str
    entry: int
    offline: Particles
    proxy: Particles
    label: int | None


class RelocatedDataset:
    """Authenticate copied metadata, then verify each requested bank on use.

    Construction opens JSON only. Original absolute paths are mapped only for
    the declared references; there is no fallback to the original RIT files.
    Use one instance per process; no global validation cache or monkeypatching.
    """

    def __init__(self, *, proxy_root, offline_root, provenance_root,
                 expected_manifest_sha256=MANIFEST_SHA256):
        require_sha256(expected_manifest_sha256, name="expected_manifest_sha256")
        self.proxy_root = Path(proxy_root).resolve(strict=True)
        self.offline_root = Path(offline_root).resolve(strict=True)
        self.provenance_root = Path(provenance_root).resolve(strict=True)
        for root in (self.proxy_root, self.offline_root, self.provenance_root):
            _require(root.is_dir(), f"Not a directory: {root}")
        self.manifest = load_json(safe(self.proxy_root, "dataset_manifest.json"))
        validate(self.manifest, "MANIFEST", test=True)
        _require(self.manifest["content_hash"] == expected_manifest_sha256,
                 "Unexpected frozen manifest; never adopt an unreviewed copied hash")
        self.study = load_json(safe(self.proxy_root, "study_spec.json"))
        study = self.study
        validate(study, "STUDY", parents=dict(pilot=study["pilot_hash"],
            population=study["population"]["content_hash"]), test=False)
        population.validate_population(study["population"], "POPULATION", test=False)
        population.validate_shards(study["population"], study["shards"])
        _require(self.manifest["parents"] == dict(study=study["content_hash"],
                 population=study["population"]["content_hash"]), "Manifest parent lineage differs")
        _require(self.manifest["recipe"] == "NOISE_V3"
                 and self.manifest["dataset_kind"] == study["kind"] == "controlled_literature_synthetic_proxy"
                 and self.manifest["native_hlt_accessed"] is False
                 and study["physics_production_qualified"] is False,
                 "Not the registered synthetic literature dataset")
        _require(self.manifest["counts"] == study["counts"] ==
                 study["population"]["counts"] == population.COUNTS, "Dataset counts differ")
        expected = [f"shards/{s['shard_id']}.json" for s in study["shards"]]
        _require([r["relative"] for r in self.manifest["shards"]] == expected,
                 "Full manifest shard coverage/order differs")
        self._references = dict(zip((s["shard_id"] for s in study["shards"]), self.manifest["shards"]))
        self.local_references = {}

        # Do not invoke producer.validate_study: it binds an execution worktree,
        # architecture and RIT absolute paths, not a read-only portability claim.
        self.bundle = self._reference("bundle", study["bundle"], area="proxy")
        validate(self.bundle, "BUNDLE", parents=dict(pilot=study["pilot_hash"]), test=False)
        _require(self.bundle["recipe"] == recipe() and self.bundle["source"] == study["source"],
                 "Frozen source/recipe differs")
        validate_calibration(self.bundle["calibration"], self.bundle["calibration_parent"])
        self.inventory = self._reference("inventory", study["inventory"], area="provenance")
        validate_inventory(self.inventory)
        profiles = {}
        for name in ("profile", "donor_profile"):
            profiles[name] = self._reference(name, study[name], area="snapshot")
            validate_split_profile(profiles[name], self.inventory)
        _require(study["population"]["parents"] == dict(
            inventory=self.inventory["content_hash"], profile=profiles["profile"]["content_hash"],
            donor_profile=profiles["donor_profile"]["content_hash"]), "Population inventory/profile differs")
        pilot = self._reference("pilot", self.bundle["pilot"], area="provenance")
        validate_pilot(pilot, "SPEC")
        _require(pilot["content_hash"] == study["pilot_hash"]
                 and pilot["recipe"] == self.bundle["recipe"]
                 and pilot["calibration"] == self.bundle["calibration"], "Pilot recipe/calibration differs")
        # Byte references are anchored by the frozen bundle. These are evidence,
        # not instructions to follow the pilot's external block dependencies.
        for name in ("pilot_receipt", "pilot_report"):
            self._reference(name, self.bundle[name], area="provenance")
        self._inventory_files = {r["path"]: r for r in self.inventory["files"]}
        for row in study["population"]["files"]:
            original = self._inventory_files[row["path"]]
            _require(all(row[k] == original[k] for k in ("sha256", "tree_key", "source"))
                     and row["raw_entries"] == original["entries"], "Offline file provenance differs")

    @classmethod
    def from_oscar_copy(cls, root=OSCAR_ROOT, *, expected_manifest_sha256=MANIFEST_SHA256):
        root = Path(root)
        return cls(proxy_root=root / PROXY_DIRECTORY,
                   offline_root=root / SNAPSHOT_DIRECTORY / "jetclass2",
                   provenance_root=root / "oscar_provenance_v1",
                   expected_manifest_sha256=expected_manifest_sha256)

    def _reference(self, name, reference, *, area):
        original = Path(reference["path"])
        if area == "proxy":
            relative = original.relative_to(Path(self.study["root"])).as_posix()
            root = self.proxy_root
        elif area == "snapshot":
            relative = original.relative_to(Path(self.study["data_root"]).parent).as_posix()
            root = self.offline_root.parent
        else:
            # Only direct, explicitly named producer references reach this route.
            relative = f"{original.parent.name}/{original.name}"
            root = self.provenance_root
        path = _check_file(safe(root, relative), reference)
        self.local_references[name] = str(path)
        return load_json(path)

    def describe(self):
        """Metadata-only report; does not assert current full-bank verification."""
        return with_content_hash(dict(contract=CONTRACT, schema_version=1,
            parents=dict(manifest=self.manifest["content_hash"], study=self.study["content_hash"]),
            recipe="NOISE_V3", counts=self.study["counts"], proxy_root=str(self.proxy_root),
            offline_root=str(self.offline_root), references=self.local_references,
            allowed_roles=["train", "validation"], class_names=self.inventory["selection"]["class_names"],
            units=dict(p4="GeV", tracking="mm"), metadata_authenticated=True,
            physical_banks_verified=False, final_test_accessed=False, final_test_evaluated=False,
            input_manifest_test_materialized=self.manifest["final_test_materialized"],
            producer_metadata_modified=False, generator_rerun=False))

    def _shards(self, role, shard_ids):
        _role(role)  # Before opening any role receipt or bank.
        rows = [s for s in self.study["shards"] if s["role"] == role]
        if shard_ids is not None:
            ids = tuple(shard_ids)
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= {s["shard_id"] for s in rows}:
                raise PermissionError("Shard subset escapes requested ordinary role")
            rows = [s for s in rows if s["shard_id"] in ids]
        release = load_json(safe(self.proxy_root, f"releases/{role}.json"))
        validate(release, "ROLE_MANIFEST", parents=self.manifest["parents"], test=False)
        expected = [self._references[s["shard_id"]] for s in self.study["shards"] if s["role"] == role]
        _require(release["role"] == role and release["counts"] == {role: self.study["counts"][role]}
                 and release["shards"] == expected and release["recipe"] == "NOISE_V3",
                 "Role release differs from frozen complete manifest")
        return rows

    def _receipt(self, shard):
        reference = self._references[shard["shard_id"]]
        path = _check_file(safe(self.proxy_root, reference["relative"]), reference)
        receipt = load_json(path)
        _require(receipt["content_hash"] == reference["content_hash"], "Shard receipt identity differs")
        attempt_name = receipt["attempt"]
        attempt = load_json(safe(self.proxy_root, f"attempts/{attempt_name}/attempt_spec.json"))
        validate(attempt, "ATTEMPT", parents=dict(study=self.study["content_hash"]), test=False)
        _check_file(safe(self.proxy_root, "study_spec.json"), attempt["study"])
        _require(attempt["name"] == attempt_name and shard["shard_id"] in attempt["shards"],
                 "Shard not registered in producing attempt")
        preflight = load_json(safe(self.proxy_root, f"attempts/{attempt_name}/preflight.json"))
        validate(preflight, "PREFLIGHT", parents=dict(study=self.study["content_hash"],
            attempt=attempt["content_hash"]), test=False)
        _require(preflight["exact_replay"] and preflight["resource_envelope_ok"], "Producer preflight differs")
        validate(receipt, "SHARD", parents=dict(study=self.study["content_hash"],
            attempt=attempt["content_hash"], shard=canonical_sha256(shard)), test=False)
        _require(all(receipt[k] == shard[k] for k in ("shard_id", "role", "jets", "ordered_identities"))
                 and receipt["physical_schema_verified"] is True and receipt["test_lock"] is None,
                 "Shard registration/physical scope differs")
        blocks = receipt["blocks"]
        prefix = f"attempts/{attempt_name}/shards/{shard['shard_id']}/"
        _require(blocks and all(b["relative"] == prefix+f"block_{i:05d}.npz"
                               and 1 <= b["jets"] <= 1000 for i, b in enumerate(blocks))
                 and sum(b["jets"] for b in blocks) == shard["jets"]
                 and sum(b["bytes"] for b in blocks) == receipt["output_bytes"], "Block coverage differs")
        return receipt

    def _proxy_blocks(self, shard, entries):
        source, _ = population.entries_for(self.study["population"], shard)
        expected_ids = list(population.ids(self.inventory["content_hash"], source, entries))
        receipt = self._receipt(shard)
        cursor = particles = 0
        for block in receipt["blocks"]:
            path = _check_file(safe(self.proxy_root, block["relative"]), block)
            values = output.arrays(path)
            _check_file(path, block)
            stop = cursor + block["jets"]
            _require([bytes(i).hex() for i in values["jet_identity"]] == expected_ids[cursor:stop],
                     "Proxy/offline canonical identity join differs")
            particles += int(values["offsets"][-1])
            yield entries[cursor:stop], values
            cursor = stop
        _require(cursor == len(entries) and particles == receipt["particles"], "Physical shard counts differ")

    def iter_proxy(self, role, *, shard_ids=None):
        """Yield (jet identity, Particles), with no ROOT/label/native-HLT reads."""
        for shard in self._shards(role, shard_ids):
            _, entries = population.entries_for(self.study["population"], shard)
            for _, values in self._proxy_blocks(shard, entries):
                yield from output.particles(values)

    def iter_pairs(self, role, *, labels=False, shard_ids=None):
        """Yield original-entry joined endpoints; no particle-index correspondence.

        Closing this generator early closes/authenticates the open ROOT file.
        Contextlib.closing is recommended for bounded diagnostics.
        """
        _role(role)
        if type(labels) is not bool:
            raise ValueError("labels must be an explicit boolean")
        for shard in self._shards(role, shard_ids):
            source, entries = population.entries_for(self.study["population"], shard)
            path = safe(self.offline_root, source["path"])
            with authenticated_open(path, source["sha256"]) as handle:
                key, tree = latest_tree(handle)
                _require(key == source["tree_key"] and tree.num_entries == source["raw_entries"],
                         "Offline ROOT tree identity differs")
                for block_entries, values in self._proxy_blocks(shard, entries):
                    proxies = list(output.particles(values))
                    # Bound sparse reads to 512-entry buckets, not a whole file
                    # or an unbounded span between selected raw entries.
                    buckets = {}
                    for index, entry in enumerate(block_entries):
                        buckets.setdefault(int(entry)//512, []).append((index, int(entry)))
                    for bucket, selected in buckets.items():
                        start = bucket*512
                        branches = list(BRANCHES) + (["jet_label", "hlt_matched"] if labels else [])
                        raw = tree.arrays(branches, entry_start=start,
                            entry_stop=min(start+512, source["raw_entries"]), library="ak", how=dict)
                        if labels:
                            keep, targets = selected_mask(ak.to_numpy(raw["jet_label"]),
                                ak.to_numpy(raw["hlt_matched"]), source["source"], self.inventory["selection"])
                        for index, entry in selected:
                            local = entry-start
                            columns = {f: ak.to_numpy(raw["part_"+f][local]) for f in JC2_FIELDS}
                            n = int(raw["jet_nparticles"][local])
                            _require(n >= 0 and all(len(v) == n for v in columns.values()),
                                     "Offline jagged counts differ")
                            if labels:
                                _require(bool(keep[local]), "Frozen row no longer eligible")
                            identity, proxy = proxies[index]
                            yield PairedJet(identity, role, source["path"], key, entry,
                                from_columns(columns), proxy, int(targets[local]) if labels else None)
