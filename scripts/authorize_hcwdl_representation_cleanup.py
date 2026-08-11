#!/usr/bin/env python3
"""Authorize cleanup for one exact registered target generation."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("target_cleanup",)))
