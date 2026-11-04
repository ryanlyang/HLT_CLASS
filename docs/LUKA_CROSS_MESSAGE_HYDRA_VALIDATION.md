# Cross-attention message Hydra validation

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
