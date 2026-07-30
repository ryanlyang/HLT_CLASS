"""Deployable HLT-only model definitions."""

from .particle_transformer import (
    PARTICLE_TRANSFORMER_CONTRACT,
    CanonicalParticleTransformer,
    build_particle_transformer,
    canonical_particle_transformer_config,
    validate_weaver_bf16_finiteness,
    validate_weaver_fp32_parity,
)

__all__ = [
    "PARTICLE_TRANSFORMER_CONTRACT",
    "CanonicalParticleTransformer",
    "build_particle_transformer",
    "canonical_particle_transformer_config",
    "validate_weaver_bf16_finiteness",
    "validate_weaver_fp32_parity",
]
