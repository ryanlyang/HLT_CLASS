#!/usr/bin/env python3
"""Produce one label-free registered final prediction shard."""
from run_hcwdl_representation_task import main
if __name__ == "__main__": raise SystemExit(main(allowed_kinds=("prediction_shard",)))
