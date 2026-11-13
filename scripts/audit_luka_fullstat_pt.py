#!/usr/bin/env python3
"""Standalone NumPy-only CLI for saved FullStat TRAIN membership audits."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from hlt_classification.operations import fullstat_pt_audit as audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", default=audit.DEFAULT_SPEC)
    run.add_argument("--output", required=True)
    run.add_argument("--widths", nargs="+", type=float, default=[50, 100, 200])
    run.add_argument("--pt-range", nargs=2, type=float, metavar=("LOW", "HIGH"))
    run.add_argument("--draws", type=int, default=3000000)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--min-cell", type=int, default=100)
    show = sub.add_parser("show", help="Read a copied report without Oscar data or source paths")
    show.add_argument("report")
    a = p.parse_args()
    if a.command == "show":
        report = audit.load_report(a.report)
    else:
        report = audit.run(a.spec, a.output, widths=a.widths, pt_range=a.pt_range,
                           draws=a.draws, min_cell=a.min_cell, epochs=a.epochs)
    print(audit.text_report(report), end="")
    if a.command == "run":
        print(f"Saved report: {Path(a.output).absolute()/'report.json'}")
        print(f"Copy/share this directory: {Path(a.output).absolute()}")


if __name__ == "__main__":
    main()
