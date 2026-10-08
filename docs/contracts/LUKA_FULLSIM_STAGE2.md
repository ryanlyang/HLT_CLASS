# Luka FullSim stage-2 contracts v1

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
