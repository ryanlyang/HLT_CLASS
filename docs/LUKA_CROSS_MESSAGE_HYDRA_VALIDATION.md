# Cross-attention message Hydra validation

## A6000 runtime correction after job144751

The supplied log from commit `c2bdfcc08b6d478165f83ccf59641e4afd34e463`
reached the first completed production miniature after passing control/encoder
numerical checks. Its resource diagnostic recorded1759 minutes, GPU peak
6761658880 bytes and host peak14607953920 bytes. Only the1440-minute ceiling
was exceeded. No A6000 full fit or scientific result was produced by that setup.

New A6000 SPEC/v3 raises the reviewed fit ceiling to2880 minutes, preserving
the existing memory bounds, individual measured times,2x margin and science.
RESOURCE_FAILURE/v2 records explicit limits, violations and projection pieces.
Legacy A6000 v2 remains1440 minutes; L40 v1 remains unchanged.

- Focused pre-change baseline: **52 passed in70.16s**.
- Final cross-message/A6000 plus Hydra/portability regression:
  **125 passed in90.78s**.
- The exact job144751 measurements reject as walltime under v2 and pass under
  v3. Tests independently reject excess runtime, host RAM, GPU memory, simultaneous
  violations and nonfinite measurements. The29h19m rendered request fits the48h
  reviewed envelope but fails the old24h envelope. Schema/site combinations,
  exact submission and the real production miniature path remain covered.
- Changed Python syntax, migration Bash blocks and scoped whitespace checks pass.

No remote change, commit or push was performed. Actual completion of all A6000
miniatures remains required in a fresh root. Keep the failed evidence and old
pending L40 jobs while following the updated A6000 setup-only runbook.

## 2026-11-04 A6000 execution amendment

User-provided L40 acceptance reports14 task miniatures with host peak13.3GiB
and GPU peak6.2-6.3GiB. Scheduler evidence reports six unallocated RTX A6000s.
These motivate the new32000MiB profile; they do not establish A6000 acceptance.

- Before changes: **44 passed in93.66s**.
- New profile and existing focused tests: **50 passed in83.95s**.
- Final profile/campaign/registration/setup-only/replay plus existing Hydra and
  portability suites: **117 passed in108.30s**.
- Tests execute actual installed-Weaver/local CUDA full-control replay and
  BF16/FP32 probes, then authenticate saved outputs and reject corruption.
  A6000 device identity is simulated locally; actual hardware must pass remotely.
- Profile tests preserve the scientific recipe/DAG, distinguish SPEC/v2 from
  v1, reject wrong GPU/RAM and mixed requests, and verify setup-only submits
  exactly one job, with idempotent15-job release through `finish`.

No remote job changes or A6000 runtime measurements have been performed here.
The [A6000 runbook](LUKA_CROSS_MESSAGE_A6000_RUNBOOK.md) provides scoped commit,
existing-input reuse, setup-only qualification and exact-pending-job migration.
The original implementation evidence below remains historical.

This is an additive, locally tested queue implementation. Real Brown L40
acceptance runs inside shared setup before science submission; this agent has
not executed it. No training outcome or high-data recovery improvement is claimed.

## Local evidence

- Before edits: joint/frozen-adapter baseline **95 passed in111.37s**.
- First message tensor/fit suite: **27 passed in36.54s**.
- Expanded transfer, cache, resource and submission suite: **38 passed in48.36s**.
  A nonfatal pytest-cache permissions warning was avoided in later runs using
  `-p no:cacheprovider`; no scientific test failed.
- Complete focused message suite: **42 passed in93.34s**.
- Message, joint/frozen-adapter, Hydra execution/portability and transfer
  regression: **229 passed in221.28s**.
- After preserving native CUDA RNG during CPU-only message-head initialization,
  final focused suite: **44 passed in82.90s**. The added teacher/student tests
  compare both CPU and CUDA random states against the native auxiliary model.
- Python AST/CLI help, both Bash entry points, all four runbook Bash blocks,
  new documentation links and whitespace checks passed. The exact rendered
  science plan has14 fits plus report, with only genuine teacher dependencies.

The focused suite uses installed Weaver, actual CPU/CUDA-BF16 forward/backward
and full production architecture identity checks. End-to-end synthetic tests
exercise all14 one-epoch production miniatures, 30-epoch teacher/student/control
workers, immutable publication, strict deployment reload, frozen U100 extraction
and read-only authenticated results. Small synthetic populations do not measure
full Brown memory, runtime or scientific quality.

Additional checks cover zero-expansion/native identity, native CPU/CUDA random
state preservation during extra-head initialization, nonzero teacher and HLT
predictor gradients, detached teacher targets, same-seed equal student/control
initialization, auxiliary gradients reaching blocks0-3, valid-slot masking and
permutation equivariance, equal weighting of jets, TRAIN-only RMS, privileged
input rejection, controls never loading oracle features or weights, ragged cache
alignment/budgets, corrupt-transfer rejection, source-bound campaign registration,
all genuine dependencies, idempotent completed submission and partial-claim
refusal. Selected teacher disabled-message rows use the same selected epoch;
student architecture deltas use the matching control. The Bash test verifies the
new authorization and detached process handoff; `setsid` is stubbed on Windows,
so it does not prove survival of Brown login-node restarts or cleanup policies.

## Files, provenance and execution

New package `src/hlt_classification/luka_cross_message_hydra/`, thin CLI/queue
helper/Slurm worker, focused tests, scientific plan, artifact contract, runbook
and this evidence record. README, HANDOFF and LEGACY_SOURCE_MAP have additive
entries. Existing donor runtime, data, results, jobs and unrelated edits remain
unchanged. No Fresh_check runtime import. Donor commit and individual reused
files are listed in [LEGACY_SOURCE_MAP](LEGACY_SOURCE_MAP.md).

New contracts: `LUKA_CROSS_MESSAGE_HYDRA_*/v1`. Source baseline is
`a40ae47ba87ee23020b482c17d467e2a42282e8f`; existing imported artifact namespaces
remain unchanged. No authoritative real-dataset L40/Tigris miniature, remote
data operation, job submission, commit or push was performed here.

Next: commit/push the scoped files, authenticate/export the original U100 encoder
and saved validation predictions on SPORC (CPU only), copy/verify the small
delta on Brown, then follow the [runbook](LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md).
The original teacher checkpoint must still exist; no substitution or retraining
is assumed. The shared setup enforces real frozen-feature extraction and all14
production miniatures before scientific jobs are submitted. This Hydra workflow
follows the user's requested site rather than the repository's legacy Tigris
deployment path.
