from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from pathlib import Path

from hlt_classification.models import scouting_particle_transformer as scouting_part
from hlt_classification.scouting.hcwdl_architecture_ablation import (
    AGGREGATE_CONTRACT, ARCHITECTURE_CHECK_CONTRACT, CAMPAIGN_CONTRACT,
    CELLS, COMMAND_PLAN_CONTRACT, COMPLETION_CONTRACT, GRAPH_CONTRACT,
    GRAPH_SHA256, NODE_CONTRACT, REPORT_CONTRACT, effect_rows, validate_graph,
)
from hlt_classification.scouting.loaders import scouting_model_factory_for_report
from hlt_classification.scouting import hcwdl_architecture_campaign as campaign


class FakeWeaver(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config["num_classes"])))
        self.projection = nn.Linear(
            int(config["input_dim"]) + 4, int(config["num_classes"]),
        )

    def forward(self, features, v=None, mask=None):
        valid = mask.to(features.dtype)
        denominator = valid.sum(-1).clamp_min(1)
        pooled = torch.cat((
            (features * valid).sum(-1) / denominator,
            (v * valid).sum(-1) / denominator,
        ), dim=1)
        return self.projection(pooled) + self.cls_token[:, 0]


def test_split_architecture_exhaustively_partitions_without_dropping(monkeypatch) -> None:
    monkeypatch.setattr(scouting_part, "_weaver_class", lambda: FakeWeaver)
    features = torch.zeros(2, 21, 7)
    vectors = torch.randn(2, 4, 7)
    mask = torch.tensor([
        [[True, True, True, True, True, False, False]],
        [[True, True, True, True, True, True, True]],
    ])
    features[0, 4, [0, 3]] = 1
    features[0, 6, [1, 4]] = 1  # token 2 is unknown and must be retained
    features[1, 2, [0, 2, 4, 6]] = 1
    charged = scouting_part._compact_partition(features, vectors, mask, charged=True)
    other = scouting_part._compact_partition(features, vectors, mask, charged=False)
    assert torch.equal(
        charged[2][:, 0].sum(1) + other[2][:, 0].sum(1),
        mask[:, 0].sum(1),
    )
    assert charged[2][0, 0].sum() == 2
    assert other[2][0, 0].sum() == 3
    assert torch.equal(charged[0][0, :, 0], features[0, :, 0])
    assert torch.equal(charged[0][0, :, 1], features[0, :, 3])
    model = scouting_part.build_split_scouting_particle_transformer()
    output = model(features, vectors, mask)
    assert output.shape == (2, 15)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    # Cached particle inputs remain FP32 while Weaver and the classifier emit
    # BF16 under the production autocast policy.  The two stream embeddings
    # must be stitched without a cross-dtype indexed assignment.
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed_precision_output = model(features, vectors, mask)
    assert mixed_precision_output.dtype == torch.bfloat16
    assert torch.isfinite(mixed_precision_output).all()
    mixed_precision_output.float().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    # A wholly absent branch is represented by an exact zero embedding, not
    # an invented constituent and not an all-masked Weaver call.
    all_charged = features.clone(); all_charged[:, 2:7] = 0
    all_charged[:, 4] = mask[:, 0]
    assert torch.isfinite(model(all_charged, vectors, mask)).all()


def test_factorial_graph_is_exact_and_json_native() -> None:
    assert validate_graph() == GRAPH_SHA256
    assert tuple(CELLS) == ("H_U", "H_S", "O_U", "O_S")
    assert CELLS["H_U"].seed_alias == CELLS["O_U"].seed_alias
    assert CELLS["H_S"].seed_alias == CELLS["O_S"].seed_alias
    assert json.loads(json.dumps([cell.payload() for cell in CELLS.values()])) == [
        cell.payload() for cell in CELLS.values()
    ]
    assert {
        CAMPAIGN_CONTRACT, GRAPH_CONTRACT, NODE_CONTRACT,
        ARCHITECTURE_CHECK_CONTRACT, REPORT_CONTRACT, AGGREGATE_CONTRACT,
        COMPLETION_CONTRACT, COMMAND_PLAN_CONTRACT,
    } == {
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_SPEC/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_GRAPH/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_NODE/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_CHECK/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_TRAINING_REPORT/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_AGGREGATE/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMPLETION/v1",
        "HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMMAND_PLAN/v1",
    }


def test_factorial_effect_definition() -> None:
    rows = {
        "H_U": {"macro_ovr_auc": 1.0},
        "H_S": {"macro_ovr_auc": 1.1},
        "O_U": {"macro_ovr_auc": 1.2},
        "O_S": {"macro_ovr_auc": 1.5},
    }
    effects = effect_rows(rows, "macro_ovr_auc")
    assert effects["architecture_effect_hlt"] == pytest.approx(.1)
    assert effects["architecture_effect_offline"] == pytest.approx(.3)
    assert effects["input_effect_unified"] == pytest.approx(.2)
    assert effects["input_effect_split"] == pytest.approx(.4)
    assert effects["interaction"] == pytest.approx(.2)


def test_split_report_selects_split_factory() -> None:
    factory = scouting_model_factory_for_report({
        "config": {"model_input": "hlt", "representation_arm": "R0"},
        "scientific_config": {"model_architecture": "split_21x2_v1"},
    })
    assert factory is scouting_part.build_split_scouting_particle_transformer


def test_campaign_plan_has_stable_identity_and_exact_resources(
    monkeypatch, tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    evidence = {
        "mode": "smoke",
        "parent": {"content_hash": "a" * 64, "data_root": "/data"},
        "parent_path": tmp_path / "parent/campaign_spec.json",
        "root": tmp_path / "parent",
        "recipe_path": tmp_path / "recipe.json", "recipe_sha256": "b" * 64,
        "split_path": tmp_path / "split.json", "split_sha256": "c" * 64,
        "selection_path": tmp_path / "selection.json", "selection_sha256": "d" * 64,
        "toff": {
            "report_path": str(tmp_path / "toff.json"),
            "report_sha256": "e" * 64, "checkpoint_sha256": "f" * 64,
        },
    }
    monkeypatch.setattr(campaign, "authenticate_parent", lambda _path: evidence)
    root = tmp_path / "factorial"
    spec = campaign.create_campaign(
        parent_campaign_spec=evidence["parent_path"], campaign_root=root,
        project_dir=repository, source_commit="1" * 40,
        authorize_live_submission=True,
        authorization_phrase="AUTHORIZE HCWDL ARCHITECTURE INPUT FACTORIAL EXACT SPEC",
    )
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert spec["command_plan_sha256"] == campaign.build_command_plan(spec)["content_hash"]
    assert len(spec["tasks"]) == 7
    assert spec["resources"]["training"] == {
        "cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1",
    }
    plan = campaign.build_command_plan(spec)
    training = [row for row in plan["commands"] if row["task_id"].startswith("train_")]
    assert len(training) == 4
    assert all("--cpus-per-task=8" in row["command"] for row in training)
    assert all("--mem=96G" in row["command"] for row in training)
    assert all("--time=06:00:00" in row["command"] for row in training)
    assert all("--gres=gpu:gh200:1" in row["command"] for row in training)
    assert all(not any(arg.startswith("--array") for arg in row["command"]) for row in training)
