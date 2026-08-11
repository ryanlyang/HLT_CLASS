#!/usr/bin/env python3
"""Run the registered population audit, disposition, and reservation task."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("reservation",)))
