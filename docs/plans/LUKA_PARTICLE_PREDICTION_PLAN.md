# Trial U100 particle-feature prediction: aligned versus unordered

User-approved comparison on SPORC debug. This is the original Luka trial
ALLJETS population, not Oscar FullStat: 299234 TRAIN / 74895 validation,
original50k report-only, historical test retired, NO independent test.

| Comparison | Prediction target | Association in the loss |
|---|---|---|
| Completed U100-w5 reference | One whole-jet teacher embedding | Same jet |
| New aligned arm | One 128-dimensional teacher particle feature per real token | Fixed persistent slot within the jet |
| New unordered arm | The same set of teacher particle features | Best one-to-one assignment within that jet |

For a jet with N real particles, the student produces N vectors through the
same small head applied to its N contextual particle tokens. The unordered
arm does not require predicted vector i to describe teacher particle i. It is
set supervision, not raw offline reconstruction, attention-matrix matching,
or forcing the student's own native hidden vectors to equal the teacher's.
The predictor is learned and removable; the native backbone receives its
training gradients. No extra set decoder or deploy-time work is introduced.

Two fresh HLT-only ParTs use the completed fixed30 regular COARSE_U100 teacher
from parent74aa58c5, also used by the completed direct U100 embedding-w5 run.
Both predict the OUTPUT of particle block 7 (eighth block, including residual
and feed-forward), before class attention. A shared 128-GELU-128 side head
predicts each teacher token, never feeding the classifier. Native deployment
has no predictor. No warm start, ladder, teacher retraining or logit KD.

Aligned arm: mean SmoothL1 beta1 per coordinate and real token, then mean per
jet. Set arm: exact per-jet Hungarian one-to-one assignment minimizing that
same mean SmoothL1 cost, then the identical differentiable loss on the chosen
pairs. Assignment is detached. No approximate transport, independent nearest
neighbors, cross-jet matching, geometry penalty or learned set decoder.
Full FP32 feature costs, teacher targets and predictor; normal BF16 backbone.
Deterministic SciPy tie behavior is environment-pinned. Both CE coefficients 1,
auxiliary coefficient 5, physical batch 128, fresh paired M0HLT model/sampler
seeds, exactly 30 epochs and original AdamW warmup/cosine. Selection is original
full-validation AUC/tie-break rule. Weak results never fail a job.

Teacher channel means/population SD are fitted over all real TRAIN tokens in
FP64 (floor1e-3). The two arms use identical normalized targets. This changes
the target/reduction relative to whole-jet-w5; weight5 is not a claim of equal
gradient strength. U100/D000 persistent slots, row IDs, labels, lengths,
prepared manifest and unchanged view builder are authenticated. These are
algorithmic offline assignments, not truth-level particle identity. No
correspondence/source metadata enters a deployable forward.

Export uses deterministic FP32 kernels, no TF32 or cuDNN benchmarking, restoring
the normal training backend afterwards. Each science job must regenerate the
same raw-token and normalization/index hashes as the accepted preflight.

One exact dry-reviewed DAG queues a new debug preflight and two afterok fits.
No login controller, manual follow-up, retries, job cancellation or holds.
Science also authenticates the exact successful preflight job and receipt.
Fixed reviewed allocation: A100,6CPU,90000MiB; preflight8h, each fit24h.
Preflight measures both full-population technical epochs, maximal real TRAIN
support backward at physical128, maximal validation forward, target export,
native parity/checkpoint readback and host/GPU peaks. It must demonstrate a
conservative <=24h projection for each fit and headroom; otherwise both fits
remain blocked. No scientific weights are reused from probes. Projection
includes source/input setup, target export,30 passes and final readback.

Large FP32 token banks exist only in uniquely-owned job-local directories under
registered `/tmp`, verified as node-local disk (not Ceph/NFS/tmpfs); no home
fallback or dependence on `SLURM_TMPDIR`. Cache exact targets once per job,
retain their hashes/moments/identities manifest, remove owned temporary payloads
on normal/error exit. Shared caches/teachers are read-only. Require2GiB
personally available persistent storage for two selected native checkpoints,
predictions, all epoch histories, reference snapshots and provenance. No
automatic deletion of models or old campaigns. Scratch requirement is checked
from actual token counts; a missing/insufficient scratch directory fails early.

Compare both arms to authenticated completed direct U100-w5, M0HLT, OFFLINE
and OFFLINE-w5; report full-validation/original50k AUC, accuracy, recovery and
rejection, with partial completion supported. Saved reports must not depend
on retired scratch banks. No test access. No new scientific run has happened
until exact pushed source, dry review and genuine remote preflight succeed.

Integration donor:01d45d6057e799b4e262f8c590b51706803ebfd1. Existing runtime
files remain unchanged. Focused tensor, corruption, actual installed-Weaver
and production-worker miniature tests precede queue readiness.
