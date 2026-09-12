"""Bounded genuine-Weaver CE/oracle/KD check, not a scientific fit or submission."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.acceptance import run_acceptance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foundation-spec", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rows-per-class", type=int, default=16)
    a = p.parse_args()
    result = run_acceptance(load_json(a.foundation_spec), data_root=a.data_root, output_root=a.output_root,
                            device=a.device, rows_per_class=a.rows_per_class)
    print(result["content_hash"])


if __name__ == "__main__":
    main()
