#!/usr/bin/env python3
"""Run the sole registered final label-selection and escrow task."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("final_selection",)))
