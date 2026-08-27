"""Task dispatch for the TRI60 M1 greedy ensemble diagnostic."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

from hlt_classification.data.cache_contracts import load_json, write_immutable_json

from .hcwdl_tri60_m1_greedy_ensemble import (
    build_campaign_complete, build_result, run_prediction_shard, shard_paths,
    validate_result, validate_source_lock,
)
from .hcwdl_tri60_m1_greedy_ensemble_campaign import (
    campaign_tasks, validate_campaign,
)
from .hcwdl_tri60_m1_greedy_ensemble_contracts import (
    CAMPAIGN_COMPLETE_CONTRACT, validate_artifact,
)


def _task(task_id: str) -> Mapping[str, Any]:
    try:
        return next(row for row in campaign_tasks() if row["task_id"] == task_id)
    except StopIteration as error:
        raise KeyError("unknown TRI60 M1 greedy task") from error


def task_outputs(spec: Mapping[str, Any], task_id: str) -> list[Path]:
    task = _task(task_id)
    root = Path(spec["campaign_root"])
    if task["kind"] == "authenticate":
        return [Path(spec["artifact_paths"]["source_lock"])]
    if task["kind"] == "inference_shard":
        return list(shard_paths(root, int(task["shard_index"])))
    if task["kind"] == "greedy_reduce":
        return [root / "reports/greedy_ensemble.json"]
    return [root / "reports/campaign_complete.json"]


class M1GreedyEnsembleWorkflow:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self.root = Path(spec["campaign_root"])

    def run(self, task_id: str, *, device: str = "cuda") -> dict[str, Any]:
        validate_campaign(self.spec, executable=False)
        task = _task(task_id)
        if shutil.disk_usage(self.root).free < int(self.spec["minimum_free_disk_bytes"]):
            raise OSError("TRI60 M1 greedy free disk is below its reserve")
        if task["kind"] == "authenticate":
            source = load_json(self.spec["artifact_paths"]["source_lock"])
            if validate_source_lock(source) != self.spec["parents"]["source_lock"]:
                raise ValueError("TRI60 M1 greedy source changed")
            return source
        if task["kind"] == "inference_shard":
            return run_prediction_shard(
                spec=self.spec, shard_index=int(task["shard_index"]), device=device,
            )
        if task["kind"] == "greedy_reduce":
            result = build_result(self.spec)
            validate_result(result, spec=self.spec)
            write_immutable_json(self.root / "reports/greedy_ensemble.json", result)
            return result
        result = load_json(self.root / "reports/greedy_ensemble.json")
        complete = build_campaign_complete(self.spec, result)
        validate_artifact(complete, contract=CAMPAIGN_COMPLETE_CONTRACT)
        write_immutable_json(self.root / "reports/campaign_complete.json", complete)
        return complete


__all__ = ["M1GreedyEnsembleWorkflow", "task_outputs"]
