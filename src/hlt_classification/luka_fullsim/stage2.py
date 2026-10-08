"""Immutable stage-2 audit, explicit conventions and physical view preparation."""
from pathlib import Path
import hashlib
import itertools
import subprocess
import time

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, load_json, load_npz_arrays,
    require_sha256, sha256_file, write_immutable_json,
)
from hlt_classification.provenance import validate_source_snapshot, validate_source_snapshot_payload
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.cms_proxy_ladder import views
from hlt_classification.literature_context.transform import build_inputs, input_contract
from .contracts import artifact, validate
from .foundation import _source, load_foundation
from .particle_audit import scan
from .particles import ParticleReader, physical

COORDINATES = ("OFFLINE", "U000", "U050", "U100", "D066", "D033", "D000")
REQUIRED_SOURCE = (
    *(f"src/hlt_classification/luka_fullsim/{n}.py" for n in
      ("particles", "particle_audit", "stage2", "stage2_cache", "preflight")),
    "scripts/luka_fullsim_stage2.py", "sbatch/run_luka_fullsim_stage2.sh",
    "docs/plans/LUKA_FULLSIM_STAGE2_PLAN.md", "docs/contracts/LUKA_FULLSIM_STAGE2.md",
)


def source(project, commit):
    result = _source(Path(project), commit)
    tracked = subprocess.run(["git", "-C", str(project), "ls-files"], check=True,
                             text=True, capture_output=True).stdout.splitlines()
    if not set(REQUIRED_SOURCE) <= set(tracked):
        raise ValueError("Commit stage-2 implementation/contracts before execution")
    if Path(__file__).resolve() != Path(project).resolve() / "src/hlt_classification/luka_fullsim/stage2.py":
        raise ValueError("Stage-2 import escapes pinned project")
    return result


def fresh(output, *protected):
    output = Path(output).resolve()
    for location in protected:
        p = Path(location).resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError("Stage-2 output overlaps protected source/artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    return output


def audit(foundation_root, output, *, project, expected_commit):
    pinned = source(project, expected_commit)
    f, inventory, splits = load_foundation(foundation_root)
    output = fresh(output, foundation_root, f["input_container"], project)
    started = time.monotonic()
    result = artifact("PARTICLE_AUDIT", parents=dict(foundation=f["content_hash"], source=pinned["content_hash"]),
                      source_snapshot=pinned, **scan(f["input_container"], inventory, splits),
                      seconds=time.monotonic()-started)
    validate_source_snapshot(pinned, repository=project, require_clean=True)
    write_immutable_json(output / "audit.json", result)
    return result


def check_audit(report, foundation):
    validate(report, "PARTICLE_AUDIT")
    validate_source_snapshot_payload(report["source_snapshot"])
    if (report["parents"] != dict(foundation=foundation["content_hash"], source=report["source_snapshot"]["content_hash"])
            or report["counts"] != {r: foundation["counts"][r] for r in ("train", "validation")}
            or report["final_test_accessed"] is not False or report["scientific_admission"] is not False
            or report["units"] != "stored_unconfirmed_no_conversion"
            or report["source_snapshot"]["worktree_clean"] is not True):
        raise ValueError("Audit lineage/coverage/semantics differ")


def conventions(*, audit_sha256, offline_unit, hlt_unit, zero_error, evidence, authority):
    result = artifact("CONVENTIONS", parents=dict(audit=audit_sha256),
        length_units=dict(offline=offline_unit, hlt=hlt_unit), zero_error=zero_error,
        momentum_units="GeV", neutral_tracking="unavailable", signed_values="as_stored",
        other_sentinels="none_all_finite_charged_values_meaningful",
        label_codes="provisional_stage1_mapping_explicitly_accepted",
        cross_file_event_independence="unverified_explicitly_acknowledged",
        authority=authority, evidence=evidence, final_test_accessed=False)
    validate_conventions(result)
    return result


def validate_conventions(value):
    digest = validate(value, "CONVENTIONS")
    require_sha256(value["parents"]["audit"], name="audit")
    if (set(value["parents"]) != {"audit"} or set(value["length_units"]) != {"offline", "hlt"}
            or any(v not in ("mm", "cm") for v in value["length_units"].values())
            or value["zero_error"] not in ("error_only_unavailable", "value_and_error_unavailable")
            or value["momentum_units"] != "GeV" or value["neutral_tracking"] != "unavailable"
            or value["signed_values"] != "as_stored"
            or value["other_sentinels"] != "none_all_finite_charged_values_meaningful"
            or value["label_codes"] != "provisional_stage1_mapping_explicitly_accepted"
            or value["cross_file_event_independence"] != "unverified_explicitly_acknowledged"
            or value["authority"] not in ("producer_confirmed", "operator_provisional")
            or not isinstance(value["evidence"], str) or len(value["evidence"].strip()) < 20
            or value["final_test_accessed"] is not False):
        raise ValueError("Explicit convention/evidence required; no inferred units or sentinels")
    return digest


def contracts(capacity=512):
    if type(capacity) is not int or capacity < 16:
        raise ValueError("Invalid nontruncating capacity")
    return dict(
        views=artifact("VIEWS", kernel=views.view_contract(),
            coordinates=list(COORDINATES), switch_domain="unchanged_donor_cms_proxy_views_v1",
            D000="native_Luka_HLT", OFFLINE="native_Luka_offline", synthetic_degradation=False),
        inputs=artifact("INPUTS", kernel=input_contract(), capacity=capacity,
            minimum_padding=16, truncation=False, model_inputs=["features", "vectors", "mask"],
            tracking_units="mm", source_deta_dphi="audited_not_model_input"))


class ViewMeter:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.rows = self.particles = self.maximum = 0

    def add(self, jet, value):
        self.rows += 1
        n = len(value.features)
        self.particles += n
        self.maximum = max(self.maximum, n)
        self.digest.update(bytes.fromhex(jet.identity))
        self.digest.update(np.asarray([jet.label, n], dtype="<i8").tobytes())
        for a in (value.features, value.vectors):
            self.digest.update(np.asarray(a, dtype="<f4").tobytes())

    def report(self):
        return dict(rows=self.rows, particles=self.particles, maximum=self.maximum,
                    input_sha256=self.digest.hexdigest(), resident_bytes=self.particles*84+self.rows*88+8)


def prepare_rows(container, inventory, splits, config, output, *, capacity=512):
    """Full ordinary populations, bounded one-source-file assignment publication."""
    validate_conventions(config)
    summaries, assignments = {}, {}
    for role in ("train", "validation"):
        meters = {c: ViewMeter() for c in COORDINATES}
        refs = []
        reader = ParticleReader(container, inventory, splits, role=role, include_offline=True)
        for index, rows in itertools.groupby(reader, key=lambda row: row.file_index):
            offsets, mappings, identities, labels = [0], [], [], []
            for jet in rows:
                hlt, off = physical(jet.hlt, config, "hlt"), physical(jet.offline, config, "offline")
                mapping = views.match_particles(hlt, off)
                mappings.append(mapping)
                offsets.append(offsets[-1]+len(mapping))
                identities.append(np.frombuffer(bytes.fromhex(jet.identity), np.uint8))
                labels.append(jet.label)
                for coord, meter in meters.items():
                    view = views.build_view(identity=jet.identity, proxy=hlt, offline=off,
                                            coordinate=coord, mapping=mapping)
                    meter.add(jet, build_inputs(view, capacity=capacity))
            arrays = dict(offsets=np.asarray(offsets, np.int64), mapping=np.concatenate(mappings).astype(np.int32),
                          identities=np.asarray(identities, np.uint8), labels=np.asarray(labels, np.int64))
            path = Path(output) / "assignments" / f"{role}_{index:04d}.npz"
            atomic_publish_bytes(path, deterministic_npz_bytes(arrays))
            refs.append(dict(path=path.relative_to(output).as_posix(), sha256=sha256_file(path),
                             bytes=path.stat().st_size, file_index=index, rows=len(labels)))
            print(f"LUKA prepare role={role} files={len(refs)} rows={meters['D000'].rows}", flush=True)
        summaries[role] = {c: m.report() for c, m in meters.items()}
        assignments[role] = refs
    return dict(summaries=summaries, assignments=assignments)


def prepare(foundation_root, audit_path, config_path, output, *, project, expected_commit):
    pinned = source(project, expected_commit)
    f, inventory, splits = load_foundation(foundation_root)
    report, config = load_json(audit_path), load_json(config_path)
    check_audit(report, f)
    validate_conventions(config)
    if config["parents"] != dict(audit=report["content_hash"]):
        raise ValueError("Conventions reference a different audit")
    output = fresh(output, foundation_root, f["input_container"], project, audit_path, config_path)
    started = time.monotonic()
    products = prepare_rows(f["input_container"], inventory, splits, config, output)
    result = artifact("PREPARED", parents=dict(foundation=f["content_hash"], audit=report["content_hash"],
        conventions=config["content_hash"], source=pinned["content_hash"]),
        source_snapshot=pinned, foundation_root=str(Path(foundation_root).resolve()),
        audit=report, conventions=config, **contracts(), **products,
        counts={r: f["counts"][r] for r in ("train", "validation")},
        seconds=time.monotonic()-started, final_test_accessed=False, science_authorized=False)
    validate_source_snapshot(pinned, repository=project, require_clean=True)
    write_immutable_json(output / "prepared.json", result)
    return result


def load_prepared(root):
    root = Path(root)
    p = load_json(root / "prepared.json")
    validate(p, "PREPARED")
    f, inventory, splits = load_foundation(p["foundation_root"])
    check_audit(p["audit"], f)
    validate_conventions(p["conventions"])
    validate_source_snapshot_payload(p["source_snapshot"])
    if (p["parents"] != dict(foundation=f["content_hash"], audit=p["audit"]["content_hash"],
            conventions=p["conventions"]["content_hash"], source=p["source_snapshot"]["content_hash"])
            or p["conventions"]["parents"] != dict(audit=p["audit"]["content_hash"])
            or p["source_snapshot"]["worktree_clean"] is not True
            or p["final_test_accessed"] is not False or p["science_authorized"] is not False
            or p["counts"] != {r: f["counts"][r] for r in ("train", "validation")}
            or any(p[k] != v for k, v in contracts().items())):
        raise ValueError("Prepared lineage/contracts differ")
    for role in ("train", "validation"):
        expected = [m["file_index"] for m in splits["memberships"][role]]
        if ([r["file_index"] for r in p["assignments"][role]] != expected
                or sum(r["rows"] for r in p["assignments"][role]) != p["counts"][role]
                or set(p["summaries"][role]) != set(COORDINATES)):
            raise ValueError("Prepared assignment coverage differs")
        for record in p["assignments"][role]:
            load_assignment(root, record)
        for meter in p["summaries"][role].values():
            if meter["rows"] != p["counts"][role] or not 0 < meter["maximum"] <= 512:
                raise ValueError("Prepared view coverage/capacity differs")
    return p, f, inventory, splits


def load_assignment(root, record):
    path = relative_file(Path(root), record["path"])
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("Assignment bytes differ")
    a = load_npz_arrays(path)
    n = record["rows"]
    if (set(a) != {"offsets", "mapping", "identities", "labels"}
            or a["offsets"].dtype != np.int64 or a["offsets"].shape != (n+1,)
            or a["mapping"].dtype != np.int32 or a["mapping"].ndim != 1
            or a["identities"].dtype != np.uint8 or a["identities"].shape != (n, 32)
            or a["labels"].dtype != np.int64 or a["labels"].shape != (n,)
            or a["offsets"][0] != 0 or a["offsets"][-1] != len(a["mapping"])
            or np.any(np.diff(a["offsets"]) <= 0)):
        raise ValueError("Assignment shapes/offsets differ")
    return a
