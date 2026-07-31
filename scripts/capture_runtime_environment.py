#!/usr/bin/env python3
"""Publish source-bound Python/package/CUDA runtime evidence."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hlt_classification.campaign import validate_campaign_spec  # noqa: E402
from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json,
    with_content_hash,
    write_immutable_json,
)
from hlt_classification.provenance import validate_campaign_source  # noqa: E402

RUNTIME_ENVIRONMENT_CONTRACT = "hlt_classification_runtime_environment_v1"


def _package_record(
    import_name: str,
    *,
    distribution_name: str | None = None,
) -> dict[str, Any]:
    module = importlib.import_module(import_name)
    version = getattr(module, "__version__", None)
    if version is None and distribution_name is not None:
        version = metadata.version(distribution_name)
    module_file = getattr(module, "__file__", None)
    return {
        "version": None if version is None else str(version),
        "module_file": (
            None if module_file is None else str(Path(module_file).resolve())
        ),
    }


def build_runtime_environment(
    *,
    campaign_spec: dict[str, Any],
    require_cuda: bool,
) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for the training runtime attestation")
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return with_content_hash(
        {
            "contract": RUNTIME_ENVIRONMENT_CONTRACT,
            "schema_version": 1,
            "campaign_spec_sha256": campaign_spec["content_hash"],
            "source_snapshot_sha256": campaign_spec[
                "source_snapshot_sha256"
            ],
            "python": {
                "version": platform.python_version(),
                "major_minor": [
                    int(sys.version_info.major),
                    int(sys.version_info.minor),
                ],
                "executable": str(Path(sys.executable).resolve()),
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "packages": {
                "awkward": _package_record("awkward"),
                "numpy": _package_record("numpy"),
                "torch": _package_record("torch"),
                "uproot": _package_record("uproot"),
                "weaver": _package_record(
                    "weaver",
                    distribution_name="weaver-core",
                ),
            },
            "cuda": {
                "required": require_cuda,
                "available": cuda_available,
                "torch_cuda_version": (
                    None if torch.version.cuda is None else str(torch.version.cuda)
                ),
                "cudnn_version": (
                    None
                    if torch.backends.cudnn.version() is None
                    else int(torch.backends.cudnn.version())
                ),
                "devices": devices,
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    campaign_spec = load_json(args.campaign_spec)
    validate_campaign_spec(campaign_spec)
    validate_campaign_source(campaign_spec, repository=args.repository)
    report = build_runtime_environment(
        campaign_spec=campaign_spec,
        require_cuda=args.require_cuda,
    )
    write_immutable_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
