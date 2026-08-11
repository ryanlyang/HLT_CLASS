#!/usr/bin/env python3
"""Run the registered HCWDL-RKD resource or recipe task."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("kernel_resources", "representation_recipe", "numerical_acceptance")))
