#!/usr/bin/env python3
"""Cache frozen teacher jet outputs and optional dense relation/bias tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.data.cache_contracts import load_json  # noqa: E402
from hlt_classification.contracts import validate_final_test_execution_lock  # noqa: E402
from hlt_classification.models.prad_particle_transformer import build_prad_particle_transformer  # noqa: E402
from hlt_classification.prad.artifacts import build_prad_teacher_output_cache  # noqa: E402
from hlt_classification.prad.cache import PradCacheDataset  # noqa: E402
from hlt_classification.prad.engine import _tensor_inputs  # noqa: E402
from hlt_classification.prad.loaders import load_selected_prad_model  # noqa: E402
from hlt_classification.prad.splits import load_prad_split_manifest  # noqa: E402
from hlt_classification.prad.streaming import build_in_memory_paired_views  # noqa: E402
from hlt_classification.prad.teacher_engine import PRAD_TEACHER_REPORT_CONTRACT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--paired-cache", type=Path)
    parser.add_argument("--streaming-inputs", action="store_true")
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--final-evaluation-lock", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dense-pairs", action="store_true")
    parser.add_argument("--max-new-shards", type=int)
    args = parser.parse_args()
    report = load_json(args.teacher_report)
    relation_dim = int(report["config"]["relation_dim"])
    teacher, _, teacher_hash = load_selected_prad_model(
        args.teacher_report,
        model_factory=lambda: build_prad_particle_transformer(relation_dim=relation_dim),
        expected_report_contract=PRAD_TEACHER_REPORT_CONTRACT,
        map_location=args.device,
    )
    target = torch.device(args.device)
    teacher.to(target).eval()
    split = load_prad_split_manifest(args.split_manifest)
    if args.streaming_inputs:
        if args.paired_cache is not None:
            raise ValueError("streaming teacher outputs forbid a durable paired cache")
        paired_dataset = build_in_memory_paired_views(
            split,
            logical_role=args.role,
            replica_ids=(0,),
            source_snapshot_sha256=args.source_snapshot_sha256,
        )[0]
    else:
        if args.paired_cache is None:
            raise ValueError("durable teacher outputs require --paired-cache")
        paired_dataset = PradCacheDataset(args.paired_cache)

    def infer(arrays):
        inputs, labels = _tensor_inputs(arrays, view="offline", device=target)
        with torch.no_grad():
            output = teacher.forward_with_relations(**inputs)
            confidence = torch.softmax(output.logits.float(), dim=-1)[
                torch.arange(len(labels), device=target), labels
            ]
        result = {
            "teacher_logits": output.logits.float().cpu().numpy().astype(np.float32),
            "teacher_true_class_confidence": confidence.cpu().numpy().astype(np.float32),
        }
        if args.dense_pairs:
            result["teacher_relation"] = (
                output.relation.float().cpu().numpy().astype(np.float16)
            )
            result["teacher_bias"] = (
                output.privileged_bias.float().cpu().numpy().astype(np.float16)
            )
        return result

    lock_hash = None
    if args.final_evaluation_lock is not None:
        lock = load_json(args.final_evaluation_lock)
        lock_hash = validate_final_test_execution_lock(lock)
        if args.role != "test":
            raise PermissionError("a final-evaluation lock is valid only for test")
        if (
            lock["final_test_cache_manifest_sha256"]
            != paired_dataset.manifest_sha256
            or lock["source_snapshot_sha256"] != args.source_snapshot_sha256
        ):
            raise PermissionError("final-evaluation lock lineage differs")
    elif args.role == "test":
        raise PermissionError("test teacher outputs require the execution lock")
    result = build_prad_teacher_output_cache(
        split,
        paired_dataset,
        logical_role=args.role,
        output_dir=args.output_dir,
        source_snapshot_sha256=args.source_snapshot_sha256,
        teacher_checkpoint_sha256=teacher_hash,
        infer=infer,
        dense_pairs=args.dense_pairs,
        final_evaluation_lock_sha256=lock_hash,
        shard_size=args.batch_size,
        max_new_shards=args.max_new_shards,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
