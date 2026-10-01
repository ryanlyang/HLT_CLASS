"""Audit raw HLT/offline jet pairing on a bounded registered sample (CPU only)."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.contracts import validate
from hlt_classification.jetclass2_delphes.pairing_audit import print_report, run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-spec", type=Path, help="Existing salience MT20 spec; used only for its data membership")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--split-profile", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    parser.add_argument("--files-per-source", type=int, default=16)
    parser.add_argument("--rows-per-file", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new diagnostic path")
    if args.campaign_spec:
        if args.inventory or args.split_profile:
            parser.error("Use a campaign spec OR inventory and split profile")
        spec = load_json(args.campaign_spec)
        validate(spec, "SALIENCE_MT20_CAMPAIGN_SPEC")
        foundation = spec["foundation"]
        validate(foundation, "SALIENCE_FOUNDATION_SPEC")
        inventory, profile = foundation["inventory"], foundation["splits"]
        data_root = args.data_root or Path(spec["data_root"])
        if args.output.resolve().is_relative_to(Path(spec["campaign_root"]).resolve()):
            parser.error("Place the diagnostic outside the existing campaign root")
    else:
        if not (args.inventory and args.split_profile and args.data_root):
            parser.error("Need --campaign-spec, or --inventory --split-profile --data-root")
        inventory, profile = load_json(args.inventory), load_json(args.split_profile)
        data_root = args.data_root
    if args.output.resolve().is_relative_to(data_root.resolve()):
        parser.error("Output must be outside the raw dataset")
    report = run_audit(inventory, profile, data_root=data_root, role=args.role,
                       files_per_source=args.files_per_source, rows_per_file=args.rows_per_file,
                       seed=args.seed, repeats=args.repeats)
    if args.campaign_spec:
        from hlt_classification.data.cache_contracts import with_content_hash
        report = with_content_hash(dict(report, source_campaign_sha256=spec["content_hash"]))
    write_immutable_json(args.output, report)
    print_report(report)
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
