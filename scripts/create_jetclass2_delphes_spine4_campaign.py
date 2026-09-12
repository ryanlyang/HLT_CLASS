"""Materialize the new scientific graph (non-submitting, acceptance-gated preview)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes.campaign import build_campaign_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foundation-spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = build_campaign_plan(load_json(a.foundation_spec))
    write_immutable_json(a.output, plan)
    print(f"Fresh fits: {plan['fresh_fit_count']}; teacher publications: {plan['probability_publication_count']}")
    print("Plan materialized. Not executable until new-data Tigris acceptance is measured.")


if __name__ == "__main__":
    main()
