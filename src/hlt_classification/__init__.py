"""Standalone HLT-only JetClass classification research package."""

from __future__ import annotations

__version__ = "0.1.0"

FOUNDATION_STATUS = {
    "transfer_block": 4,
    "implemented_out_of_order_blocks": (5,),
    "scientific_pipeline_implemented": False,
    "next_transfer_block": 5,
    "authoritative_weaver_parity_passed": False,
}

__all__ = ["FOUNDATION_STATUS", "__version__"]
