"""Deployable HLT-only model definitions."""

from .particle_transformer import (
    PARTICLE_TRANSFORMER_CONTRACT,
    CanonicalParticleTransformer,
    build_particle_transformer,
    canonical_particle_transformer_config,
    validate_weaver_bf16_finiteness,
    validate_weaver_fp32_parity,
)
from .prad_particle_transformer import (
    PRAD_PARTICLE_TRANSFORMER_CONTRACT,
    PradForwardOutput,
    PradParticleTransformer,
    build_prad_particle_transformer,
    validate_prad_runtime,
)

__all__ = [
    "PARTICLE_TRANSFORMER_CONTRACT",
    "PRAD_PARTICLE_TRANSFORMER_CONTRACT",
    "CanonicalParticleTransformer",
    "PradForwardOutput",
    "PradParticleTransformer",
    "build_particle_transformer",
    "build_prad_particle_transformer",
    "canonical_particle_transformer_config",
    "validate_weaver_bf16_finiteness",
    "validate_weaver_fp32_parity",
    "validate_prad_runtime",
]
