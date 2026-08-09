#!/usr/bin/env python3
"""Train one registered HCWDL-RKD node, control, or confirmation row."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("train_node", "train_control", "confirmation", "smoke_probe")))
