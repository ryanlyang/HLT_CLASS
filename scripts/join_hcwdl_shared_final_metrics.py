#!/usr/bin/env python3
"""Perform the single registered locked label/metric join."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("metric_join", "final_aggregate", "validation_only_aggregate")))
