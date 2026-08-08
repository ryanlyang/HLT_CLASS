#!/usr/bin/env python3
"""Build authenticated seed-12345 source-file-disjoint Scouting splits."""

from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hlt_classification.data.cache_contracts import (  # noqa: E402
    load_json, validate_content_hash, write_immutable_json,
)
from hlt_classification.scouting.audit import (  # noqa: E402
    SOURCE_MANIFEST_CONTRACT, SOURCE_MANIFEST_VERSION,
)
from hlt_classification.scouting.splits import (  # noqa: E402
    build_split_manifest, role_records, source_file_record_from_manifest_row,
    validate_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    source = load_json(args.source_manifest)
    source_hash = validate_content_hash(
        source, expected_contract=SOURCE_MANIFEST_CONTRACT,
        expected_schema_version=SOURCE_MANIFEST_VERSION,
    )
    records = [source_file_record_from_manifest_row(row) for row in source["files"]]
    manifest = build_split_manifest(records, source_manifest_sha256=source_hash, seed=args.seed)
    validate_split_manifest(
        manifest, source_manifest_sha256=source_hash, expected_inventory=records,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_immutable_json(args.output_dir / "split_manifest.json", manifest)
    for role, filename in (("train", "train.txt"), ("validation", "val.txt"), ("final_test", "test.txt")):
        (args.output_dir / filename).write_text("".join(f"{item.path}\n" for item in role_records(manifest, role)), encoding="utf-8")
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "split", "sample", "file", "raw_entries", "mapped_entries",
            "class_counts", "sha256",
        ))
        for role in ("train", "validation", "final_test"):
            for item in role_records(manifest, role):
                writer.writerow((
                    role, item.stratum, item.path, item.raw_entries,
                    item.mapped_entries, ";".join(map(str, item.class_counts)), item.sha256,
                ))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
