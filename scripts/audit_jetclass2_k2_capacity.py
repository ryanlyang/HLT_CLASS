"""Read-only K=2 particle-count census; no matching or job submission."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hlt_classification.jetclass2_delphes.capacity_audit import run_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "inventory-path", "splits-path", "partial-plan-path", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--step-size", type=int, default=100_000)
    args = p.parse_args()
    report = run_audit(**vars(args))
    overall = report["groups"]["all"]["overall"]
    print(f"AUDIT COMPLETE: {overall['overflow_jets']:,}/{overall['jets']:,} jets "
          f"overflow ({100*overall['overflow_fraction']:.6f}%).")
    print(f"Report: {args.output_root / 'report.json'}")


if __name__ == "__main__":
    main()
