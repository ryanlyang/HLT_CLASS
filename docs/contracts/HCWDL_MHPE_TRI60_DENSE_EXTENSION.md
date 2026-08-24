# HCWDL MHPE TRI60 Dense Extension Contracts

The additive dense extension uses the
`HCWDL_MHPE_TRI60_DENSE_EXTENSION_* /v1` contract family defined in
`hcwdl_mhpe_tri60_dense_contracts.py`.

The immutable graph binds exact rational coordinates, 48 fresh `DX_` nodes,
15 expanded uniform probability reducers, 23 read-only source components,
track-local representation carriers, and `DX_M1E -> DX_M2` terminal
compression. The source lock binds the original campaign, graph, recipe,
foundation, completed early reports/checkpoints, compact U-stage probability
locks, and—while incomplete—the exact source campaign-complete Slurm job ID.
It also provides a new read-only consumer adapter that binds each immutable
source probability lock to an exact set of dense consumers; the old source
manifest itself is not changed.
The later source gate binds the completed source report and all lower imported
reports/checkpoints.

Dense training reports and checkpoints use graph-specific contracts and may
not be loaded as ordinary TRI60 artifacts without their additive authority.
Dense probability shards contain only canonical 32-byte identities and FP32
15-class probabilities. Manifests bind component order, equal weights,
consumer registry, temperature, and component report/checkpoint/logit hashes.

No contract grants final-test access. No representation target, particle
view, hidden state, component logits, rolling checkpoint, or duplicate
deployable checkpoint is a durable artifact.
