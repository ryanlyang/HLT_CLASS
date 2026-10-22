# CE checkpoint continuation with U100 embedding supervision

Status: registered implementation plan; remote acceptance is still required.

## Question and matched comparison

Does auxiliary prediction of frozen regular COARSE_U100 embeddings improve an
already trained HLT classifier beyond simply continuing CE training?

Both arms independently restore the exact selected M0HLT native checkpoint
from the fixed30 campaign at source 74aa58c582ff6f497a2c6d91c503f5489fac3e9a.
Restore every native parameter and buffer, including the trained classifier.
Never initialize from a technical probe, an embedding-pretrained checkpoint,
the other continuation, or another teacher. Authenticate this checkpoint and
reproduce its saved validation probabilities before accepting the setup.

| Arm | Objective | Initial native weights |
| --- | --- | --- |
| CE_CONTINUE | CE | selected M0HLT |
| CE_EMBED_U100_W5 | CE + 5 * embedding SmoothL1 | the identical selected M0HLT |

The treatment adds a fresh Linear128-GELU-Linear128 prediction head to the
128-dimensional pre-classifier representation. Its output predicts frozen
regular COARSE_U100 representations, not teacher logits. The prediction head
is training-only; no auxiliary head or privileged inputs remain at inference.
The control does not read embedding targets or execute the auxiliary head.

## Fixed protocol

- Same authenticated ALLJETS population: 299,234 train / 74,895 validation.
  Original 50,000 validation rows are a report-only subset. The historical test
  was retired into training; there is NO independent test set.
- Native D000/HLT inputs only, identical identities and preprocessing in both
  arms. No provenance indices as features. No logit KD or attention KD.
- Fresh AdamW and scheduler per arm, all native parameters trainable. No
  optimizer moments restored. Predictor initialization does not perturb the
  native RNG stream; use matched initialization/dropout and sampler seeds.
- Exactly 30 additional epochs, batch 128, no early stopping. Scale the entire
  existing fixed30 warmup3/cosine LR curve by 0.1: peak 3e-5, floor 1.5e-6.
  Apply the same LR to every trainable parameter, including the fresh head.
- Preserve existing fixed30 AdamW decay/betas/epsilon and BF16 forward policy.
- Select among the 30 continuation checkpoints using the existing full-
  validation AUC/CE/rejection/update ordering. Do not fall back to epoch zero
  based on poor performance. All scientific results, including regressions,
  are retained. Original50k never selects checkpoints or hyperparameters.
- Frozen teacher: existing regular COARSE_U100 from the same fixed30 parent,
  not OFFLINE, D066, a fusion model, or a newly trained teacher.
- Reuse unchanged LUKA_U100_EMBEDDING_BANK/v1 semantics: TRAIN-only FP32 teacher
  representations, FP64 TRAIN mean/standard deviation, minimum scale 1e-3,
  mean SmoothL1 beta=1 against detached standardized targets. No validation
  embedding targets. Validation teacher inference is only a saved-probability
  readback, never normalization fitting or student supervision.

## Execution and acceptance

New additive source family relative to ed6d0af4be65f059356543f90583b30185702615.
Do not modify existing scientific runtimes or reuse another campaign's output
directory. This plan explicitly selects SPORC debug/A100, not Tigris, for its
genuine production-worker miniature and science runs.

One authorized exact DAG submits three jobs: setup, then independent CE and
embedding continuations with afterok(setup). The setup performs authenticated
teacher and M0HLT readbacks, one shared TRAIN target export, installed-Weaver
and auxiliary-head parity, maximum-support train/validation stresses, and one
disposable complete-population epoch per arm. Both probes also start from the
original M0HLT, never from each other. Science workers revalidate the setup's
receipt, payload hashes, measured admission, exact journal job and successful
Slurm accounting before starting. No login-node follow-up controller needed.

Resource ceilings: 6 CPUs, 90,000 MiB, one A100; setup 8h, each continuation 6h.
Admit measured host peak below 85% and GPU peak below 90%, with conservative
1.75x measured full-population projections fitting both science limits.
Failed acceptance blocks both arms; no automatic retries or partial resumes.
Dry review, exact pushed source, explicit plan-hash authorization required.

Reserve 2 GiB compact persistent outputs: shared TRAIN bank, byte-identical
native initialization, references, two technical probes, and one selected
native checkpoint plus predictions/report per arm. No new particle caches,
attention banks, per-epoch checkpoints, deletions, job cancellations, or edits
to Fresh_check. Keep the parent checkpoint and lineage intact.

The primary result is CE_EMBED_U100_W5 minus CE_CONTINUE on the full validation
population. Report original M0HLT/OFFLINE, OFFLINE-w5 and COARSE_U100 as context,
not compute-matched treatment controls. Report both arms even if either loses;
single-seed, repeatedly inspected development validation is exploratory only.
