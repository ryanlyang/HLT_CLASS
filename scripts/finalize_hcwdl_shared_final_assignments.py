#!/usr/bin/env python3
"""Finalize and audit all registered final assignment shards."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("assignment_finalize",)))
