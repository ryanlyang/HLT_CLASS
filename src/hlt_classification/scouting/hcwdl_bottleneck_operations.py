"""Bounded operational inventories for bottleneck foundation/campaign tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hlt_classification.data.cache_contracts import (
    sha256_file, with_content_hash, write_immutable_json,
)


OUTPUT_INVENTORY_CONTRACT = "HCWDL_FULLCARD_BOTTLENECK_TASK_OUTPUT_INVENTORY/v1"


def publish_output_inventory(
    *, root: str | Path, task_id: str, array_index: int | None,
    outputs: Sequence[str | Path],
) -> Path:
    campaign_root = Path(root).resolve()
    rows = []
    for raw in outputs:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"bottleneck task output is absent: {path}")
        rows.append({
            "path": str(path), "bytes": path.stat().st_size,
            "byte_sha256": sha256_file(path),
        })
    suffix = "single" if array_index is None else f"{array_index:06d}"
    output = campaign_root / "output_inventories" / task_id / f"{suffix}.json"
    payload = with_content_hash({
        "contract": OUTPUT_INVENTORY_CONTRACT, "schema_version": 1,
        "task_id": task_id, "array_index": array_index,
        "outputs": rows, "total_bytes": sum(row["bytes"] for row in rows),
        "dense_pair_matrices_persisted": False,
        "rolling_resume_persisted": False,
        "final_test_accessed": False,
    })
    write_immutable_json(output, payload)
    return output


__all__ = ["OUTPUT_INVENTORY_CONTRACT", "publish_output_inventory"]
