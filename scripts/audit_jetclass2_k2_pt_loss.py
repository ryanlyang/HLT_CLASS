"""Diagnose the minimum scalar-pT cost of K=2 cropping on an audited scope."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hlt_classification.jetclass2_delphes.capacity_pt_audit import run_pt_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("count-root", "inventory-path", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--step-size", type=int, default=20_000)
    args = p.parse_args()
    result = run_pt_audit(**vars(args))
    print("PT AUDIT COMPLETE:", result["groups"]["all"])


if __name__ == "__main__":
    main()
