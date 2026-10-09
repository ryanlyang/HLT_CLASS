# Luka FullSim stage-2 contracts

Authority: [stage-2 plan](../plans/LUKA_FULLSIM_STAGE2_PLAN.md).

`LUKA_FULLSIM_PARTICLE_AUDIT/v1` binds the stage-1 foundation and the current
clean source snapshot. It records selected train/validation coverage and raw
stored-unit statistics, explicitly without physical-unit or sentinel inference.
Negative/nonfinite tracking entries are diagnostic counts, not silently replaced
measurements. There is no test reader or admission flag in this report.

`LUKA_FULLSIM_CONVENTIONS/v1` binds that audit and explicit operator declarations:
per-side mm/cm length units; zero errors invalidate only error or value-and-error;
signed d0/dz kept as stored; neutral tracking non-applicable; GeV momentum;
finite values otherwise meaningful. Evidence must be recorded. A provisional
operator declaration is labelled as such, not producer-confirmed physics.

`LUKA_FULLSIM_VIEWS/v1` and `LUKA_FULLSIM_INPUTS/v1` wrap the intentionally reused
salience/persistent-shell algorithm and asinh/log1p feature kernel. Historical
contracts are not changed or impersonated. Assignment NPZs contain ordered row
identities, labels, offsets and injective HLT-to-offline maps; none are model
features. Preparation authenticates source files before/after reads, every
selected row and every registered view, and records byte digests of model inputs.

`LUKA_FULLSIM_PREPARED/v1` binds foundation, audit, conventions, current source,
all matching payload bytes, counts, per-view sizes and model-input fingerprints.
Cache reuse rechecks these bindings, raw bytes and ordered identities. D000 cache
requests no offline particle branches and reproduces the prepared native HLT
fingerprint. `LUKA_FULLSIM_GPU_PREFLIGHT/v1` records exact allocation/environment,
installed parity, mask/offload checks, full-population technical passes, worst
TRAIN batch stress, checkpoint/probability round trips and measured resources.
It cannot stand in for a scientific training report or a campaign authorization.

Hash verification is not a claim of cryptographic signing or producer approval.
Readiness never follows merely from file existence. Unknown conventions remain
an explicit stop between audit and physical preparation.

## Operational recovery: GPU preflight v2

`LUKA_FULLSIM_GPU_PREFLIGHT/v2` / schema 2 changes execution order/resources,
not prepared inputs or the science recipe. V1 artifacts retain their meaning.

`LUKA_FULLSIM_PREFLIGHT_REUSE/v1` binds the prepared hash, distinct preparation
and execution source hashes, authenticated worktree locations, and SHA256 of
every byte-identical existing tracked `src/` file. Any deletion/change, or any
addition outside `luka_fullsim/{preflight_reuse,preflight_cache,preflight_v2}.py`,
rejects reuse. Both worktrees pass the ordinary clean-source verifier before and
after execution. CLI/worker/docs changes are covered by the new complete source
snapshot. Neither arbitrary source mismatch nor dirty worktrees are waived.

Small probes retain the first four TRAIN rows and first longest witness for
each of U000/D000/U050. Candidate upper bounds use mapping cardinalities and
raw HLT/offline scalar counts; particle reads remain selected TRAIN entries.
Witnesses must attain prepared maxima. Unchanged parity/stress kernels retain
batch256, three steps, padding/support, BN population, offload and optimizer.
Small probes never replace full-population technical passes. Afterwards, full
U000/D000/U050 caches are built in one ROOT pass per role, and each original
model-input fingerprint is verified before use. D000-only requests still decode
no offline branches. Caches stay in RAM, never persisted as a disk cache.

V2 allocation: 6 CPUs / 256000 MiB / one SPORC A100. Small-stress peak plus the
conservative future-cache reservation must stay below 85% host RAM; GPU peak
below 90% capacity. Resource checks may reject allocation, never poor metrics.
The report binds both sources, reuse, witnesses, phase start/end/failure records,
cache-build timings, Linux RSS/GPU peaks, weights and TRAIN-bank readbacks.
An OOM may leave only a phase start record/log heartbeat; it cannot publish
`preflight.json`. Actual v2 Slurm success is still required to establish GPU
correctness and whether the larger host-memory allocation is sufficient.
