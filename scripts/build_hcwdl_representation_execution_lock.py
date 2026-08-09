#!/usr/bin/env python3
"""Publish the combined prediction/metric execution lock."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("execution_lock", "finalist_lock")))
