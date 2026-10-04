#!/usr/bin/env python3
"""Read-only relocated NOISE_V3 inspection and bounded paired-reader smoke."""
import argparse
from contextlib import closing
from itertools import islice
import json

from hlt_classification.literature_proxy_consumer import (
    MANIFEST_SHA256, OSCAR_ROOT, RelocatedDataset,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "smoke"))
    parser.add_argument("--root", default=str(OSCAR_ROOT))
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    parser.add_argument("--jets", type=int, default=8)
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.jets <= 128:
        parser.error("--jets must be between 1 and 128 for this bounded smoke")
    print("Authenticating copied metadata; original artifacts remain unchanged.", flush=True)
    data = RelocatedDataset.from_oscar_copy(args.root, expected_manifest_sha256=args.manifest_sha256)
    print(json.dumps(data.describe(), indent=2, sort_keys=True))
    if args.command == "smoke":
        count = 0
        with closing(data.iter_pairs(args.role, labels=args.labels)) as rows:
            for row in islice(rows, args.jets):
                count += 1
                print(json.dumps(dict(identity=row.identity, source_file=row.source_file,
                    tree_key=row.tree_key, entry=row.entry, label=row.label,
                    offline_particles=len(row.offline), proxy_particles=len(row.proxy)), sort_keys=True), flush=True)
        if count != args.jets:
            raise ValueError("Requested smoke rows not available")
        print(f"PAIRED READER SMOKE PASS: {count} {args.role} jets; final test untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
