#!/usr/bin/env python3
"""Reopen the bounded validation-only HCWDL-RKD action proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from _hcwdl_representation_common import artifact
from hlt_classification.scouting.hcwdl_representation_acceptance_evidence import (
    build_validation_proxy_proof,
)
from hlt_classification.scouting.hcwdl_representation_nonfinal_acceptance import (
    validate_nonfinal_acceptance_authority_static,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    artifact_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    authority = artifact(args.authority)
    proof = build_validation_proxy_proof(
        result_reference=artifact_reference(args.result),
        authority=authority,
        authority_validator=validate_nonfinal_acceptance_authority_static,
    )
    if proof != artifact(args.result):
        raise ValueError("validation-proxy result is not its canonical proof")
    print(proof["content_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
