"""Plot validation Hbb/Hcc rejection curves for dense C25P75 MHPE."""

from __future__ import annotations

import argparse
from pathlib import Path

from hlt_classification.scouting.hcwdl_mhpe_roc import build_dense_c25p75_roc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = build_dense_c25p75_roc(
        args.campaign_spec, args.output_dir, device=args.device,
    )
    print(report["figures"]["pdf"]["path"])
    print(report["figures"]["png"]["path"])
    print(report["figures"]["progression_pdf"]["path"])
    print(report["figures"]["progression_png"]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
