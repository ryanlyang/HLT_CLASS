"""Task dispatcher for the HCWDL architecture-input factorial."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import (
    load_json, validate_content_hash, with_content_hash, write_immutable_json,
)

from .hcwdl_architecture_ablation import (
    AGGREGATE_CONTRACT, ARCHITECTURE_CHECK_CONTRACT, CELLS,
    COMPLETION_CONTRACT, build_aggregate,
)
from .hcwdl_architecture_runner import node_output_dir, run_factorial_node


def build_architecture_check(spec: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    import torch
    from hlt_classification.models.scouting_particle_transformer import (
        _compact_partition, build_scouting_particle_transformer,
        build_split_scouting_particle_transformer,
    )

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("factorial architecture check requires its requested GPU")
    torch.manual_seed(20260811)
    features = torch.randn(4, 21, 16, device=target)
    vectors = torch.randn(4, 4, 16, device=target)
    vectors[:, 3] = vectors[:, :3].square().sum(1).add(1).sqrt()
    mask = torch.ones(4, 1, 16, dtype=torch.bool, device=target)
    mask[0, :, -3:] = False
    # Exact, exhaustive identities: charged in alternating active slots;
    # neutral/unknown in the remainder. Padded values are deliberately noisy.
    features[:, 2:7] = 0
    features[:, 4, 0::2] = 1
    features[:, 6, 1::4] = 1
    charged = _compact_partition(features, vectors, mask, charged=True)
    noncharged = _compact_partition(features, vectors, mask, charged=False)
    original = mask[:, 0].sum(1)
    partitioned = charged[2][:, 0].sum(1) + noncharged[2][:, 0].sum(1)
    if not torch.equal(original, partitioned):
        raise RuntimeError("split architecture token conservation failed")
    unified = build_scouting_particle_transformer().to(target)
    split = build_split_scouting_particle_transformer().to(target)
    unified.train(); split.train()
    unified_logits = unified(features, vectors, mask)
    split_logits = split(features, vectors, mask)
    (unified_logits.float().sum() + split_logits.float().sum()).backward()
    if (
        unified_logits.shape != (4, 15) or split_logits.shape != (4, 15)
        or not torch.isfinite(unified_logits).all()
        or not torch.isfinite(split_logits).all()
        or any(parameter.grad is None for parameter in split.parameters())
    ):
        raise RuntimeError("split architecture forward/backward check failed")
    counts = {
        "unified_21_v1": sum(parameter.numel() for parameter in unified.parameters()),
        "split_21x2_v1": sum(parameter.numel() for parameter in split.parameters()),
    }
    return with_content_hash({
        "contract": ARCHITECTURE_CHECK_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": spec["content_hash"], "device": str(target),
        "dtype": "float32", "parameter_counts": counts,
        "parameter_ratio_split_over_unified": counts["split_21x2_v1"] / counts["unified_21_v1"],
        "transformer_blocks_per_encoder": {
            "unified_21_v1": [8], "split_21x2_v1": [8, 8],
        },
        "capacity_is_part_of_architecture_factor": True,
        "token_partition_conserved": True, "unknown_policy": "noncharged_stream",
        "native_toff_architecture_claimed": False, "final_test_accessed": False,
    })


class ArchitectureFactorialWorkflow:
    def __init__(self, spec: Mapping[str, Any], *, repository: str | Path) -> None:
        from .hcwdl_architecture_campaign import (
            semantic_source_hashes, validate_campaign,
        )
        validate_campaign(spec, executable=False)
        if spec.get("semantic_source_sha256") != semantic_source_hashes(repository):
            raise ValueError("factorial worker semantic source differs")
        self.spec = dict(spec); self.root = Path(spec["campaign_root"])

    def run(self, task_id: str) -> list[Path]:
        tasks = {row["task_id"]: row for row in self.spec["tasks"]}
        if task_id not in tasks:
            raise ValueError("unknown architecture-factorial task")
        task = tasks[task_id]; kind = task["kind"]
        if kind == "architecture_check":
            output = self.root / "locks/architecture_check.json"
            if output.exists():
                value = load_json(output)
                validate_content_hash(
                    value, expected_contract=ARCHITECTURE_CHECK_CONTRACT,
                    expected_schema_version=1,
                )
                if value.get("campaign_spec_sha256") != self.spec["content_hash"]:
                    raise ValueError("reused architecture check lineage differs")
            else:
                write_immutable_json(output, build_architecture_check(self.spec, device="cuda"))
            return [output]
        if kind == "train_node":
            node_id = str(task["node_id"])
            run_factorial_node(spec=self.spec, node_id=node_id, device="cuda")
            output = node_output_dir(self.root, node_id)
            return [
                output / "training_report.json", output / "hcwdl_training_report.json",
                output / "runtime.json",
            ]
        if kind == "aggregate":
            output = self.root / "reports/validation_aggregate.json"
            if not output.exists():
                write_immutable_json(output, build_aggregate(self.spec))
            value = load_json(output)
            validate_content_hash(value, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1)
            return [output]
        if kind == "campaign_complete":
            aggregate = load_json(self.root / "reports/validation_aggregate.json")
            aggregate_hash = validate_content_hash(
                aggregate, expected_contract=AGGREGATE_CONTRACT, expected_schema_version=1,
            )
            output = self.root / "reports/campaign_complete.json"
            payload = with_content_hash({
                "contract": COMPLETION_CONTRACT, "schema_version": 1,
                "campaign_spec_sha256": self.spec["content_hash"],
                "aggregate_sha256": aggregate_hash, "fit_count": len(CELLS),
                "mode": self.spec["mode"], "scientific_result_does_not_control_completion": True,
                "final_test_accessed": False,
            })
            if not output.exists():
                write_immutable_json(output, payload)
            elif load_json(output) != payload:
                raise FileExistsError("factorial completion artifact differs")
            return [output]
        raise RuntimeError("unhandled architecture-factorial task kind")


__all__ = ["ArchitectureFactorialWorkflow", "build_architecture_check"]
