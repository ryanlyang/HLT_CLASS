# Balanced reconstruction Hydra: queue-readiness evidence

Evidence date2026-11-19. This implements the
[600k plan](plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md), not a completed
physics campaign. No remote jobs were submitted, changed or cancelled. Existing
datasets, scientific checkpoints and other working-tree edits were preserved.

## Population and storage

The downloaded verified new dataset was audited locally using scalar membership
metadata. New subset build and independent replay both produced
`21175a03adc3d847e62d84a447b04525fc813207685a37d0b0dcb630acafd1f1`.
Evidence artifacts are ignored local files under
`artifacts/audits/balanced_reconstruction_600k_v1/`; no dataset enters Git.

| Class | TRAIN | Validation |
| --- | ---: | ---: |
| QCD | 427186 | 106429 |
| X_bb | 35806 | 8937 |
| X_cc | 34645 | 8689 |
| X_ss | 33939 | 8682 |
| X_qq | 33774 | 8488 |
| X_gg | 34650 | 8775 |
| Total | 600000 | 150000 |

All120 TRAIN cells survive. Minimum cell1234; maximum expected appearances is
4.051863857 per epoch. TRAIN HLT/OFFLINE support280/316; validation221/247.
Decoder support316 is TRAIN-derived. Test and reduced-out roles stay unchanged.
This subset audit read no particles and performed no inference.

A separate discarded real-data technical probe replayed the same membership,
authenticated the referenced ROOT checksums and decoded12 selected TRAIN jets,
two per class. It ran a production R24 CUDA update with full decoder support316;
these rows had maximum HLT/OFFLINE multiplicity131/143. All losses were finite.
Normalization fitted only those12 TRAIN rows and the embedding target was a
technical synthetic substitute, not a scientific teacher bank. No test particles
or validation inference were used, and no scientific checkpoint was published.
Local evidence:`real_train_probe.json` in the same ignored audit directory.

Initialized production tensor/row inventory estimates conservative new outputs
at4,828,964,049 bytes (4.50GiB), within the10GiB allowance. Native selected
states617,306,112 bytes; prediction arrays691,200,000; outgoing teacher
banks739,200,000; largest full restart estimate142,007,988. The estimate includes
staging and extra metadata margin. It is an inventory estimate, not a measured
remote disk peak. Physical/declared free space,2GiB reserve and the unspent
existing ladder14GiB reservation are checked before launch and throughout fits.

Budget-sizing follow-up: the current initial output allowance is8GiB, replacing
the original10GiB above. The complete initialized316-support inventory remains
4,828,964,049 bytes, leaving over3GiB additional allowance plus the separate2GiB
reserve. The protected ladder allowance stays14GiB. The initial maximum is24GiB;
it is an estimate/admission rule, not physical storage reservation. A dedicated
test reinitializes all18 production models and verifies the inventory and exact
scientific recipe equality with1e602e33. No further deletion is required merely
to clear admission at the user's last reported24.539GiB free snapshot.

Operator update2026-11-19: the paragraph above describes the original release.
The revised source checks HOME admission at registration and explicit manual
resume only. Its versioned execution policy removes submission/release, GPU-job
startup and epoch checks. Genuine filesystem write errors propagate; the previous
committed checkpoint remains available. The scientific recipe and donor pins
are unchanged. See the updated runbook for a fresh-commit launch after the
original01324411 attempt stopped at admission before registration/submission.

Revision checks: **51 passed,1 skipped in75.99s** across the reconstruction and
new retirement suites (41 reconstruction tests plus10 cleanup tests). The skip
requires four real CUDA/NCCL GPUs. Tests make the HOME checker raise if invoked
during scientific training, GPU startup or automatic submission, and verify those
paths still complete. Simulated`ENOSPC` at epoch2 propagates while epoch1 remains
committed; resuming reproduces uninterrupted selected native tensors exactly.
Cleanup tests verify one-file-only deletion, unchanged retained model/prediction/
manifest bytes, content-hash/checksum/lineage rejection, alias-normalized transitive
dependencies, unknown jobs, newly arriving consumers and exclusive retirement
records. No remote deletion was performed by these local tests. Scientific
Weaver/CUDA and full-support checks remain as described below.

## Local automated checks

Before implementation, the focused existing reconstruction-stability,
EMA-control and six-class Oscar suites passed **130 tests,1 skipped** in509.21s.
The skip required four real CUDA/NCCL GPUs.

The final combined new-campaign and existing Hydra-tail suites passed
**65 tests,1 skipped** in154.99s:39 new campaign tests plus26 existing tail tests.
The new skip also requires four real CUDA/NCCL GPUs; this check is enforced by
the production Hydra qualification before science release.

Coverage includes proportional deterministic membership, exact replacement-draw
clock/EMA, TRAIN-only BN membership, synthetic ROOT reads that forbid test rows
and undeclared OFFLINE branches, and all nine arm objectives. An independent
full-graph reference checks chunked losses, gradients, AdamW moments and
parameters, including delayed/asymmetric decoder schedules. Four actual CPU/Gloo
processes verify gradient SUM for uneven shards and partial batches.

Production fit functions run all nine arm types in compact fixtures, including
new OFFLINE/U100 banks, all four native readout exports, checksum/row-order
tamper checks and compact retirement. Failure injection verifies exact selected
tensor reproduction after an interrupted epoch, and final teacher-export recovery
without repeated training updates. Submission tests exercise durable intent,
accepted-ID bindings, ambiguous replies, duplicates and teacher dependencies.
Retirement cannot delete outside its own fit; allocation guards reject wrong GPU
counts/requests. All531 canonical donor pins match their committed bytes.

Actual Weaver0.5.3 forward/input/parameter parity and all nine new native/CUDA
paths passed locally with FP32 and TF32 disabled, including live token taps,
full-graph/chunked decoder comparisons at both schedule positions, and exact
native deployable exports. Local environment: Python3.10.19, PyTorch2.5.1,
CUDA12.1, NVIDIA RTX2000 Ada laptop GPU, Weaver0.5.3 source under the existing
`tmp/r8-followup-weaver` harness. This is single-card evidence, not Pascal or
four-GPU acceptance. Four new shell scripts passed Git Bash`bash -n`.

## Required remote acceptance

The new plan adapts generic Tigris validation to this authorized Hydra workflow.
No Tigris or real Hydra miniature has been run from this workstation. The queued
four-GPU qualification uses actual selected TRAIN/validation views, installed
Weaver parity, four-rank SyncBN/full-graph comparisons, both-seed discarded
miniatures, native/EMA reload checks and complete-support stress. It measures
full validation (including gathered metrics), BN calibration and training, then
checks a conservative48h projection and measured CPU/GPU memory margins.

Shared CPU arrays use PSS accounting so mmap pages are counted once across
ranks, plus private transient-peak, elastic-agent and input-launcher allowances. Resource proof is
specific to the requested four GTX1080Ti GPUs,12CPUs,64000MiB and this population.
It cannot be substituted by local Ada timing or old small-dataset runs.

Qualification must pass before automatic release. Every scientific allocation
also runs a discarded miniature with its real frozen teacher bank. No miniature
loss threshold gates scientific performance. Poor recovery is retained; invalid
inputs, nonfinite quantities, lineage drift, ambiguous submissions, missing
resources and corrupt artifacts stop execution. No final-test API was added.
