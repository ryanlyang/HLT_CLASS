"""Bounded local HCWDL behavioral smoke over synthetic authenticated fixtures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, load_json, sha256_file, with_content_hash, write_immutable_json,
)

from .evaluation import classification_metrics
from .hcwdl_ladder import GRAPH_SHA256, NODE_REGISTRY
from .hcwdl_recipe import example_recipe
from .hcwdl_training import node_training_config, select_checkpoint
from .highcov_cache import (
    DenseAssignmentStore, publish_assignment_manifest, publish_assignment_shard,
    sampled_recomputation_audit,
)
from .highcov_data import Particles
from .highcov_matcher import HighCoverageMatcher
from .highcov_resources import load_highcov_resources, resource_validation_report
from .repair import build_alpha_repaired_inputs
from .schema import HLT_FEATURE_SPECS
from .training import pmard_loss


SMOKE_REPORT_CONTRACT: Final = "HCWDL_LOCAL_SMOKE_REPORT/v1"


def _p4(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.column_stack((
        pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta),
        1.1 * pt * np.cosh(eta),
    ))


def _matcher_fixture() -> tuple[Particles, Particles]:
    hlt = Particles(
        _p4(np.asarray([50., 30., 15., 5.]), np.asarray([0., .1, -.2, .5]), np.asarray([0., .2, -.3, 1.])),
        np.asarray([0, 1, 3, 2]), np.asarray([-1, 1, 0, 1]),
        np.zeros((4, 7)), np.zeros((4, 7), bool),
    )
    offline = Particles(
        _p4(np.asarray([52., 29., 14., 4.8, 8.]), np.asarray([.001, .101, -.195, .49, -1.]), np.asarray([.002, .198, -.29, 1.01, 2.])),
        np.asarray([0, 1, 4, 2, 3]), np.asarray([-1, 1, 0, 1, 0]),
        np.zeros((5, 7)), np.zeros((5, 7), bool), np.asarray([0, 1, 4, 2, 5]),
    )
    return hlt, offline


def _repair_fixture():
    arrays = {spec.branch: [np.zeros(2, np.float32)] for spec in HLT_FEATURE_SPECS}
    for branch, value in {
        "scoutpfcand_px": [10, 0], "scoutpfcand_py": [0, 5],
        "scoutpfcand_pz": [0, 0], "scoutpfcand_energy": [10.1, 5.1],
    }.items():
        arrays[branch] = [np.asarray(value, np.float32)]
    hlt = {
        "scoutpfcand_quality": [1, 2], "scoutpfcand_charge": [1, 0],
        "scoutpfcand_isEl": [0, 0], "scoutpfcand_isMu": [0, 0],
        "scoutpfcand_isChargedHad": [1, 0], "scoutpfcand_isGamma": [0, 1],
        "scoutpfcand_isNeutralHad": [0, 0], "scoutpfcand_phirel": [3.1, -2.8],
        "scoutpfcand_etarel": [.1, -.2], "scoutpfcand_abseta": [.1, .2],
        "scoutpfcand_pt_log": [1., .5], "scoutpfcand_normchi2": [2, 0],
        "scoutpfcand_dz": [.01, 0], "scoutpfcand_dxy": [-.02, 0],
        "scoutpfcand_dxysig": [-2, 0], "scoutpfcand_btagEtaRel": [.4, 0],
        "scoutpfcand_btagPtRatio": [.6, 0], "scoutpfcand_btagPParRatio": [.7, 0],
        "scoutpfcand_dzsig": [1.5, 0], "scoutpfcand_e_log": [1.1, .6],
        "scoutpfcand_lostInnerHits": [1, 0],
    }
    for name, value in hlt.items(): arrays[name] = [np.asarray(value, np.float32)]
    charged = {
        "px": 20, "py": 0, "pz": 1, "energy": 20.1, "quality": 5, "charge": -1,
        "isEl": 1, "isMu": 0, "isChargedHad": 0, "phirel": -3., "etarel": .4,
        "abseta": .8, "pt_log_nopuppi": 2.3, "normchi2": 4, "dz": .03,
        "dxy": -.04, "dxysig": -3, "btagEtaRel": .9, "btagPtRatio": .8,
        "btagPParRatio": .6, "dzsig": 2.5, "e_log_nopuppi": 2.5, "lostInnerHits": 2,
    }
    neutral = {
        "px": 0, "py": 12, "pz": -1, "energy": 12.1, "isGamma": 0,
        "isNeutralHad": 1, "phirel": 3., "etarel": -.3, "abseta": .7,
        "pt_log_nopuppi": 1.7, "e_log_nopuppi": 1.8,
    }
    for name, value in charged.items(): arrays[f"cpfcandlt_{name}"] = [np.asarray([value], np.float32)]
    for name, value in neutral.items(): arrays[f"npfcand_{name}"] = [np.asarray([value], np.float32)]
    offline = [np.asarray([
        [charged[name] for name in ("px", "py", "pz", "energy")],
        [neutral[name] for name in ("px", "py", "pz", "energy")],
    ], np.float32)]
    assignment = np.full((1, 200), -1, np.int16); assignment[0, :2] = [1, 0]
    confidence = np.zeros((1, 200), np.float32); confidence[0, :2] = [.2, .95]
    return arrays, offline, assignment, confidence


def _torch_bytes(value: object) -> bytes:
    import torch
    stream = BytesIO(); torch.save(value, stream); return stream.getvalue()


def run_local_smoke(output_root: str | Path) -> dict[str, Any]:
    import torch
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("HCWDL local smoke output must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    resources = load_highcov_resources(); resource_report = resource_validation_report()
    hlt, offline = _matcher_fixture()
    matcher = HighCoverageMatcher(resources.empirical, resources.calibration)
    match = matcher.match(hlt, offline)
    if not match.accepted.all():
        raise RuntimeError("HCWDL smoke matcher fixture did not complete the shell")

    parents = {"split_manifest_sha256": "1" * 64, "row_selection_sha256": "2" * 64,
               "matcher_resources_sha256": resource_report["content_hash"]}
    manifests = {}
    for role in ("train", "validation"):
        role_root = root / "assignments" / role
        publish_assignment_shard(
            role_root / "shard_0000", source_path=f"fixture/{role}.root", role=role,
            source_fold=0 if role == "train" else None, entries=[0, 1],
            hlt_categories=[hlt.category, hlt.category], results=[match, match],
            parents={**parents, "source_file_sha256": "3" * 64},
        )
        manifest_path = role_root / "manifest.json"
        publish_assignment_manifest(
            manifest_path, role=role,
            shard_metadata_paths=[role_root / "shard_0000.json"],
            expected_mapped_jets=2, parents=parents,
        )
        sampled_recomputation_audit(
            manifest_path, recompute=lambda source, entry: match, sample_size=2, seed=7,
        )
        DenseAssignmentStore(manifest_path).get(f"fixture/{role}.root", 1)
        manifests[role] = manifest_path

    arrays, offline_p4, assignment, confidence = _repair_fixture()
    repaired = {
        alpha: build_alpha_repaired_inputs(
            arrays, offline_p4, assignment, alpha=alpha,
            repair_family="HIGHCOV_SHELL_EXACT/v1", confidence_weights=confidence,
            offline_arrays=arrays, identity_keys=("fixture.root::tree::0",), discrete_seed=19,
        ) for alpha in (0.0, .25, .5, .75, 1.0)
    }
    if repaired[0.0].features.tobytes() == repaired[1.0].features.tobytes():
        raise RuntimeError("HCWDL smoke repair endpoints unexpectedly coincide")

    recipe = example_recipe()
    labels = torch.arange(15, dtype=torch.long).repeat(2)
    generator = torch.Generator().manual_seed(8181)
    hlt_features = torch.randn((len(labels), 12), generator=generator)
    shifts = {"hlt": 0.0, "d25": .025, "d50": .05, "d75": .075, "d100": .1, "toff": .15}
    domain_features = {name: hlt_features + value for name, value in shifts.items()}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(12, 15)
        def forward(self, value):
            return self.linear(value)

    models: dict[str, Tiny] = {}
    reports: dict[str, dict[str, Any]] = {}
    for node_id, node in NODE_REGISTRY.items():
        seed = 1000 + list(NODE_REGISTRY).index(node_id)
        torch.manual_seed(seed); model = Tiny()
        if node.initialization == "warm":
            model.load_state_dict(models[node.initialization_parent].state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = node_training_config(
            node_id, recipe, train_rows=len(labels), replicate_seed=seed,
            require_authorized_recipe=False,
        )
        records = []
        states = []
        for update in (1, 2):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(domain_features[node.student_domain])
            teacher_logits = {}
            for teacher in node.teachers:
                models[teacher.node_id].eval()
                with torch.no_grad():
                    teacher_logits[teacher.role] = models[teacher.node_id](domain_features[teacher.domain]).detach()
            hlt_target = teacher_logits.get("predecessor")
            privileged_target = teacher_logits.get("privileged")
            if node.loss_kind == "ce_kd":
                only = teacher_logits["sole"]
                if node.teachers[0].domain == "hlt": hlt_target = only
                else: privileged_target = only
            parts = pmard_loss(
                logits, labels, class_weights=torch.ones(15), configuration=config.loss,
                hlt_teacher_logits=hlt_target, privileged_teacher_logits=privileged_target,
            )
            parts["total"].backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): validation_logits = model(domain_features[node.student_domain]).numpy()
            metrics = classification_metrics(validation_logits, labels.numpy())
            records.append({"update": update, **metrics}); states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        selection = select_checkpoint(records)
        selected_index = [row["update"] for row in records].index(selection["selected_update"])
        model.load_state_dict(states[selected_index]); models[node_id] = model
        node_root = root / "training" / node_id; node_root.mkdir(parents=True, exist_ok=True)
        selected_path = node_root / "selected_model.pt"; final_path = node_root / "final_model.pt"
        atomic_publish_bytes(selected_path, _torch_bytes({"model": states[selected_index]}))
        atomic_publish_bytes(final_path, _torch_bytes({"model": states[-1]}))
        report = with_content_hash({
            "contract": "HCWDL_SMOKE_NODE_REPORT/v1", "schema_version": 1,
            "node_id": node_id, "node": node.payload(), "updates": 2,
            "validation_history": records, "selection": selection,
            "selected_checkpoint_sha256": sha256_file(selected_path),
            "final_checkpoint_sha256": sha256_file(final_path),
            "teacher_nodes": [teacher.node_id for teacher in node.teachers],
            "warm_parent": node.initialization_parent,
            "loss_arity": len(node.teachers), "finite": True,
        })
        write_immutable_json(node_root / "training_report.json", report); reports[node_id] = report
    if set(reports) != set(NODE_REGISTRY):
        raise RuntimeError("HCWDL local smoke did not complete every graph node")
    result = with_content_hash({
        "contract": SMOKE_REPORT_CONTRACT, "schema_version": 1,
        "graph_sha256": GRAPH_SHA256, "matcher_resources_sha256": resource_report["content_hash"],
        "matcher_native_index": match.native_offline_index.tolist(),
        "matcher_confidence_hex": [float(value).hex() for value in match.confidence],
        "assignment_manifests": {
            role: load_json(path)["content_hash"] for role, path in manifests.items()
        },
        "repair_alphas": [0.0, .25, .5, .75, 1.0],
        "repair_endpoint_exact": True, "nodes_completed": list(reports),
        "updates_per_node": 2, "final_test_accessed": False,
        "scientific_result": False,
    })
    write_immutable_json(root / "smoke_report.json", result)
    return result


__all__ = ["SMOKE_REPORT_CONTRACT", "run_local_smoke"]
