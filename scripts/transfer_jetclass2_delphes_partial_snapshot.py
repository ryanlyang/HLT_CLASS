"""Build or safely extract a planned JetClass2 partial-snapshot archive."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.partial_snapshot_transfer import (
    build_transfer_archive,
    extract_transfer_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--inventory", type=Path, required=True)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--transfer-manifest", type=Path, required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--transfer-manifest", type=Path, required=True)
    extract.add_argument("--inventory", type=Path, required=True)
    extract.add_argument("--plan", type=Path, required=True)
    extract.add_argument("--output-parent", type=Path, required=True)
    extract.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    inventory = load_json(args.inventory)
    plan = load_json(args.plan)
    if args.mode == "build":
        if args.transfer_manifest.exists():
            parser.error("Transfer manifest already exists")
        result = build_transfer_archive(
            data_root=args.data_root,
            inventory=inventory,
            plan=plan,
            archive_path=args.archive,
        )
        write_immutable_json(args.transfer_manifest, result)
    else:
        result = extract_transfer_archive(
            archive_path=args.archive,
            transfer=load_json(args.transfer_manifest),
            inventory=inventory,
            plan=plan,
            output_parent=args.output_parent,
            receipt_path=args.receipt,
        )
    print(result["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
