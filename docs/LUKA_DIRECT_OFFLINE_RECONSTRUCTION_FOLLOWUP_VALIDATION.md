# R8 follow-up v2 implementation validation

## Scope and source

Eighteen fresh fits: two matched control/R8 seed pairs, eight weights through10,
and six strategy variants. Implementation baseline:
`7bab93c4256fa97d71b51e112ac630d66acb5b88`, preserving pre-existing uncommitted
follow-up work and unrelated edits. Runtime donor:
`b5464e73c94b322a81a33032baea18f1615a5a66`; exact module provenance is in
[LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md).

Changed campaign registry/contracts, reconstruction target dispatch, normalized
field weighting and exact matching, particle taps, coefficient schedules,
gradient diagnostics, resource acceptance, submission/release/report routing,
CLI/queue wrapper, focused tests, plan, runbook, README and handoff. No external
runtime donor was copied. FOLLOWUP_SPEC/RECIPE v2 and execution R5 identify the
new scientific grid; base R3 and Group A/B/C recipe identities are retained.

## Local verification

Local environment: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`,
PyTorch2.5.1 with CUDA, authoritative Weaver0.5.3 installed from the cached wheel
into `tmp/r8-followup-weaver`. PYTHONPATH points to repository src and that
installation; PYTHONDONTWRITEBYTECODE=1. No package installed globally.

Before changes, the focused suite had94 passes and3 failures in166.77s:
the existing partial follow-up's replica comparison/default keys, a missing
`reused` fixture field, and the old registry test expecting R13 to be unknown.
The initial system Python lacked dependencies; the recorded suite uses the
complete CUDA environment above.

The new focused miniatures passed all control/replica/high-weight/strategy
training and native-export cases. A tap test initially expected an unused
concatenated half to receive None rather than an explicit zero gradient; the
assertion now accepts the actual zero-gradient representation.

Final combined regression: **183 passed in947.44s** across
`test_luka_reconstruction_followup.py`, `test_luka_reconstruction_batched.py`,
`test_luka_direct_reconstruction.py`, `test_luka_reconstruction_release.py`
and `test_luka_next_supervision_hydra.py`. No skips. Log:
`tmp/r8-final-tests.log` (local evidence, not a reusable scientific artifact).
Latest durable-release/registration checks:2 passed in20.45s.
Latest report-label/exact-release checks:2 passed in43.82s, covering the final
reporting and report-command changes independently of the long suite.

The installed full-depth Weaver/CUDA checks exercised weight10 and block4+8
memory with298 decoder slots, full128 and partial2 batches, live gradients,
CE/U100/reconstruction diagnostics and native-only deployment export. These
are synthetic local checks, not Brown A6000 performance qualification.

Coverage includes scalar/batched assignment, loss and gradient parity; 1:1:2
weighting in matching and loss; preserved original hashes and controls; exact
new-seed replicas; schedule boundaries/fade; live two-depth gradients without
extra parameters; finite high weights; full setup/receipt acceptance; immutable
registration and quota; and simulated Slurm dry review, exact afterok release,
eighteen independent requests, afterany report, setup failure and idempotence.
The inherited suite exercises ambiguous-submission rejection and CPU allocation
rounding. Real submission is mocked in local scheduler tests.

AST parsing37 Python files, Bash syntax of two runbook blocks and two scripts,
CLI help, documentation links and scoped whitespace checks passed. Local
synthetic resource tests stub host/GPU accounting; their values are not Brown
performance measurements.

## Remote boundary and next action

No Brown/Tigris job was submitted, and no commit or push was made. No scientific
AUC/recovery benefit is claimed. Real Hydra qualification remains pending:
authenticated full data, original A6000/software profile, installed-Weaver
parity, all13 maximum-weight/loss/memory paths, full/partial batches and real
multiplicity/validity tails, unchanged48-hour/85%-memory gates.

Commit/push the scoped implementation, then use the
[runbook](LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_RUNBOOK.md) to register a
fresh v2 root and submit setup plus durable release. Successful setup releases
all18 fits and a CPU report automatically. A6000 availability must be checked
on Brown; this implementation does not change GPU precision/hardware silently.
