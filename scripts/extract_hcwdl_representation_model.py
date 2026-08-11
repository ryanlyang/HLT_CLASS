#!/usr/bin/env python3
"""Extract a deployable HLT-only state through its registered training task."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("train_node", "confirmation")))
