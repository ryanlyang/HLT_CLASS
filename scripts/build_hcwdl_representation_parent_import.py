#!/usr/bin/env python3
"""Run the registered HCWDL-RKD parent-import or attestation task."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("tap_schema", "surface_parity", "architecture_attestation", "parent_loss_attestation", "parent_import")))
