# Hydra adapter validation and donor record

## Job 131484 completed; v2 campaign qualification implemented (2026-11-03)

User-supplied readout authenticates completed precision job 131484 from source
`ad55d3cd32d7826d6e16a475228fece3c6099da4`, root
`/cs/home/rlyang/hlt_u100_pairs/runs/pair_precision_ad55d3cd_r1`.
Accounting reports COMPLETED/0:0, elapsed 1m43s. Full validation AUC is
0.96209461 on Hydra versus 0.96209280 saved on SPORC; exact difference is
1.813499919e-6, approximately 0.04574 historical recovery percentage points.
CPU/GPU FP32 on the fixed 527 rows differs by maximum 1.3113022e-6 and RMS
5.4447115e-8; repeated original BF16 batches are identical. Cross-site BF16
max/RMS remain 0.01554453373/0.001017804247. The local BF16 backend comparison
itself differs by maximum 0.01215446, supporting a backend/precision explanation.
QCD rejection differences are retained, including X_bb @30% passing counts
28 versus 29. There is no A100 FP32 reference or independent test.

After discussing these results, the user requested the campaign. Implemented
the explicitly versioned qualification in
[the active plan](plans/LUKA_U100_PAIR_HYDRA_PLAN.md) and
[contract](contracts/LUKA_U100_PAIR_HYDRA.md). New SPEC/RESULTS and
PORTABILITY_POLICY/EVIDENCE/PORTABILITY artifacts use v2. V1 diagnostics and
strict historical gates remain unchanged. Local FP32 max/RMS limits are
1e-5/1e-6, repeated BF16 batches must match exactly, and the original 5e-5 AUC
drift bound remains. Cross-site BF16 max/RMS are report-only under v2.

The new `portability.py` replays validation in the original diagnostic source,
binds exact job/spec/report/receipt/prediction/submission hashes, recomputes the
saved numerical evidence, and checks the shared setup's actual GPU/backend and
validation population. It never performs new control inference. Shared setup
still requires every original miniature/resource check. The new report adds
the Hydra baseline, retains SPORC references, and labels historical recovery
anchors. The existing queue helper/CLI and exact DAG use v2 authorization.
See [the current commit/queue runbook](LUKA_U100_PAIR_HYDRA_V2_RUNBOOK.md).

Donor commit: `ad55d3cd32d7826d6e16a475228fece3c6099da4`.
Changed `luka_u100_pair_hydra/{campaign,contracts,submission,worker}.py`, its CLI
and queue helper; added `portability.py` and portability tests. Reused unchanged
`precision.py`, original-source `luka_attention_weight/source.py:replay`,
`luka_fullsim/foundation.py:_source`, data/cache and file-reference contracts,
the original exact-DAG helper, and all original correction model/training/target/
resource code. No external or Fresh_check code was migrated.

Before changes: **59 tests passed in 30.32s**. Initial new qualification suite:
**18 passed in 10.93s**. After the added AUC-bound test and final changes:
**79 combined Hydra/precision/portability tests passed in 34.24s**, no skips.
After adding saved-metric scalar-roundoff handling and its regression, the final
suite passed **80 tests in 32.30s**, no skips. This allows only 1e-12 absolute/
relative recomputation roundoff and retains original authenticated scalars in
reusable hashes; it does not relax probability-array or identity checks.
Coverage includes v1 strict parity, v2 original-source/receipt binding, changed
arrays/identity/settings rejection, same-GPU setup, diagnostic reuse without
control inference, resource measurement still invoked, explicit Hydra reporting,
and successful/failed exact dispatcher paths for both versions. Existing local
Weaver CPU/GPU diagnostic tests also pass; synthetic qualification fixtures do
not replace real L40 production miniatures. Queue shell syntax, CLI help and
whitespace checks pass. The helper prints/saves its full dry plan and binds the
already-authorized submission automatically; the remaining prompt is the user's
available output-space allowance. No remote submission, commit or push was performed.
Next: push the scoped files and invoke the v2 helper on Brown. That command
authenticates the real saved diagnostic before submitting new shared setup.

## Separate Hydra precision diagnostic implemented (2026-11-02)

The user authorized a Hydra-only check after the 131410 mismatch below. Added
`precision.py` and `precision_campaign.py` in the Hydra package, a dedicated
`queue_luka_u100_pair_hydra_precision.sh`, CLI subcommands and focused tests.
Updated the additive source allowlist, plan/contract and
[diagnostic runbook](LUKA_U100_PAIR_HYDRA_PRECISION_RUNBOOK.md). The queue helper
performs dry review and submits exactly one one-hour L40 inference job. It uses
the existing bundle/environment/checkpoint, opens only validation D000 payloads,
and cannot dispatch scientific fits. The original science gate is unchanged.

Donor commit: `d7589b312fe5cea7ebb177db0ede70388f0daf45`.
Existing `luka_u100_pair_hydra/{campaign,submission,worker,site,inputs}.py`
provide source/site authentication, exact-job ownership and control-replay
patterns. `jetclass2_delphes/{model,runner}.py` and
`luka_u100_contrastive/worker.py:extended_metrics` are imported unchanged for
native inference and metrics. `scouting/hcwdl_exact_dag_submission.py` and
`cms_proxy_ladder/contracts.py` supply unchanged journals and file references.
The existing absolute-project Slurm wrapper is reused unchanged. No external
code, Fresh_check dependency or scientific-model migration was introduced.

New distinct `PRECISION_*` artifact kinds are v1 under the Hydra prefix;
existing scientific contracts/tolerances remain unchanged. A diagnostic receipt
binds the spec, exact job, report and immutable output bytes. It is not a TASK
or ACCEPTANCE artifact. Finite discrepancies remain reportable outcomes.

Before implementation, the focused Hydra suite passed **43 tests in 29.15s**.
Afterward, the new diagnostic suite passed **16 tests in 8.93s** and the combined
adapter/diagnostic suite passed **59 tests in 29.74s**, with no skips. These use
the actual Weaver 0.5.3 wheel sources and local PyTorch 2.5.1. Actual CPU/GPU
FP32/BF16 replay was exercised on synthetic jets with a compact native model
on an NVIDIA RTX 2000 Ada Generation Laptop GPU; it is not a production L40
or transferred-checkpoint measurement. Backend restoration (including errors),
unchanged checkpoint bytes, validation-only reads, all four modes, same-batch
repeatability, probability validation, finite failed-tolerance reporting,
exact one-job submission, duplicate prevention, lineage and output corruption
checks passed. Full historical source/site guard tests also remain green.

The queue helper passes Git Bash syntax checking; CLI help and whitespace checks
pass. No remote jobs were submitted or changed. Next: commit/push the scoped
files, invoke the diagnostic helper on Brown, and inspect its full-validation
and fixed-batch evidence. Real Hydra diagnostic execution, any explicit revised
portability policy, and the four scientific fits remain outstanding.

## Job 131410: allocation passed, numerical portability failed (2026-11-02)

Executed source: `d7589b312fe5cea7ebb177db0ede70388f0daf45`.
User-supplied worker log confirms the `(null)` cluster export was resolved
through the controller to hydra. The worker admitted the registered L40 and
environment, then loaded 299,234 TRAIN and 74,895 validation jets with original
fingerprints. It reached native control inference, before particle target
export or miniatures; no scientific fit was released.

The saved CONTROL_PARITY/v1 report (hash
`f42058b210ca6c4d04a20eaa7c74a1fc312436c02522de7a4a113d8d2acccd96`)
uses the expected checkpoint and predictions, exact software identity, and an
NVIDIA L40 with compute capability 8.9. All population/order checks passed.

| Comparison | Observed absolute difference | Registered limit | Outcome |
| --- | ---: | ---: | --- |
| Full validation macro AUC | 0.0000018134999187902068 | 0.00005 | Pass |
| Maximum individual probability | 0.015544533729553223 | 0.005 | Fail |
| RMS over probabilities | 0.001017804247275696 | 0.0002 | Fail |

The AUC difference is about 0.046 percentage points of the historical recovery
gap, but aggregate AUC agreement does not establish per-jet or tail equivalence.
The original producer and Hydra both use the same native checkpoint reload,
batch128 and BF16 prediction path. Cross-device reduced-precision variation is
a plausible explanation, not a demonstrated diagnosis. PyTorch's
[numerical accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
does not guarantee bitwise results across devices/backends.

No tolerance, scientific runtime, source guard or saved artifact was changed
in this diagnostic step. The existing local control-parity regression passes
(1 test, 7.06s); it is synthetic and cannot establish cross-GPU agreement.
Next required evidence is a controlled replay of fixed original validation
batches with FP32 and BF16 (including a fixed attention backend), using the same
authenticated checkpoint. A revised portability policy would need explicit
documentation/versioning; do not silently raise limits to fit this observation.
No commit, push, remote mutation, new training or submission was performed.

## Literal null cluster export: reproduced job 131385 (2026-11-02)

Donor commit: `17a1719ea5d19ae8e2befb1cfc963f31565e2bf6`;
changed runtime: `src/hlt_classification/luka_u100_pair_hydra/site.py`.
The job log proves that the controller's job ID, RUNNING state, partition,
account, QoS, node/task counts, CPUs and memory all matched. The only rejected
value was the exported literal `SLURM_CLUSTER_NAME="(null)"`. The previous
fallback tested only for an absent variable, so it never queried the controller.
This evidence supersedes the uncertain diagnosis for this new attempt; it
does not recover the missing record of earlier job 131366.

The minimal runtime change treats absent and literal `(null)` cluster exports
as unspecified in both fallback selection and comparison. It requires a
successful controller configuration query identifying `hydra`; wrong/unknown
names and query failure still reject. No export is spoofed, and the raw value
remains in the log. Resource, scientific and artifact contracts remain v1;
the plan/contract only clarify this observed metadata representation.

Pre-change focused suite: **26 passed in 7.23s**. A new regression using the exact
131385 scheduler evidence then reproduced the identical two-field rejection
before editing runtime. After repair: **31 focused tests passed in 5.50s**,
including the regression, controller names sporc/unknown/empty, failed controller
query, existing allocation conflict checks, environment equality and GPU guards.
CUDA/installed-environment checks are mocked in allocation tests; this is not
real Hydra GPU acceptance. No remote changes, commit or push were performed.

Preserve both failed roots and reuse the existing environment and extracted
bundle. Next: commit/push the scoped repair and queue a new root using the
[retry commands](LUKA_U100_PAIR_HYDRA_RUNBOOK.md#retry-after-setup-allocation-rejection).
Real L40 parity/miniatures remain pending; successful setup still gates the
four fits. No new donor files or external code were copied.

## Allocation compatibility repair after job 131366 (2026-11-02)

Repair donor: `63f31c2ffd3606508a789609e04e2c1f1c5ad38f`, specifically
`src/hlt_classification/luka_u100_pair_hydra/site.py` and its allocation tests.
No additional external code or scientific donor was copied. This section
supersedes the initial implementation's unexecuted-site status below.

User-supplied evidence: setup 131366 ran 75 seconds and failed at the opaque
allocation guard before data loading/miniatures; no science jobs were released.
`sacct` recorded hydra/gpus/default/normal, six requested/allocated CPUs,
120000M and one allocated GPU. Its task-count field was blank. The subsequent
`scontrol` query returned Invalid job id, so neither the original controller
values nor batch environment can be recovered from that evidence. An exact
offending field is **not established**.

The repair accepts unspecified controller task metadata only with explicit
batch evidence, checks every available value for conflicts, and can authenticate
missing cluster/memory environment exports from controller evidence. Exact job,
running state, resources, venv, software and L40 checks remain enforced. Allowed
scheduler fields are printed before checks; each mismatch names expected and
observed values. No scientific code, requested resources or artifact schema
changed. Original source, failed outputs, transferred bundle and environment
are preserved; retry uses a newly pushed commit and fresh output root.

Before edits, three allocation/site/environment tests passed (5.66s). After
edits, all 26 focused guard tests passed (5.67s), covering unspecified metadata,
missing exports, conflicting counts/site/memory, malformed evidence, exact
memory units, safe diagnostic logging, and unchanged GPU/software checks.
The complete Hydra adapter suite then passed **38 tests in 22.55s**, including
portable input/target replay, unchanged four-arm fitting and exact dispatch.
Both edited Python files parse, the runbook's fresh-retry Bash block passes
`bash -n`, all four local documentation targets resolve, and scoped
`git diff --check` passes. Existing unrelated working-tree edits were retained.
Real L40 numerical parity and miniature acceptance still require a successful
shared setup. No remote jobs, commit or push were performed by the agent.

## Initial adapter implementation

Local implementation base: `65952ed3805f812c06d1ea54117c35b630cc684e`.
The user reported successful export and per-member verification on Brown of
2,287 files, 3,127,574,502 payload bytes. The new consumer pins that archive and
manifest. This record does not claim an executed Hydra environment or GPU job.

New files: the seven `luka_u100_pair_hydra` modules, CLI, environment installer,
queue helper, Slurm worker, focused tests, plan, execution contract and runbook.
Existing documentation gains pointers/status; original scientific code is unchanged.

Before edits,32 focused transfer/correction tests passed in8.39s. After initial
implementation,83 combined tests passed in59.29s using the PyPI Weaver0.5.3 wheel
in an isolated local test path. They cover archive corruption/exclusivity,
transferred synthetic ROOT-derived view fingerprints and matching joins, particle
target export/reload, native checkpoint probability replay, environment mismatch,
Hydra test-only requests, exact two-lane dispatch, failed setup and duplicate
submission, and the original CPU/CUDA correction/model production regressions.
These are synthetic local checks, not the recorded SPORC environment replay or
real L40 resource evidence. Two additional allocation/source-guard tests then
passed in5.27s (85 passing cases across the focused checks). AST parsing passed
for9 new Python files; all three shell scripts passed bash syntax checks, scoped
whitespace checks passed, and four local documentation links resolved.

Donors (all at the base commit): `luka_u100_pair_correction/{worker,submission}`
adapted for Hydra orchestration; `{targets,campaign,source}` for portable loading,
registration and source guards; `{model,geometry,losses,training,resources}`
imported unchanged. `luka_u100_pair_transfer/bundle` supplies transfer integrity.
`luka_fullsim/{regular_cache,alljets_data,foundation,regular_submission}` supplies
block/foundation/source and environment primitives. `luka_u100_embedding/targets`
supplies the unchanged bank validator. `jetclass2_delphes/{model,runner,reporting}`,
`luka_u100_contrastive/worker`, `cms_proxy_ladder/{contracts,submission}`,
`data/cache_contracts` and `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}`
provide existing model/metric/artifact/Slurm helpers. Original correction CLI,
queue helper, worker shell and transfer/correction test fixtures informed the
additive counterparts. No Fresh_check donor or runtime dependency.

Execution and extraction use new v1 contracts described in
[the contract](contracts/LUKA_U100_PAIR_HYDRA.md). Scientific contracts are reused
without changing their meaning. The remaining tasks are clean commit/push,
isolated Brown environment matching, extraction, full dry review and genuine
shared setup. Setup owns numerical portability and measured resource admission;
only its successful completion releases the four scientific fits.
