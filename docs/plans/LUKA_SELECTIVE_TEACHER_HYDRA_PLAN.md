# Selective-information teacher campaign

Status: implemented and locally ready to queue Hydra setup. Scientific registration: v1. This plan
authorizes local queue preparation, not execution from this workstation.

## Question and fixed comparison

Can separate teachers exploiting restricted privileged fields provide useful,
complementary jet embeddings for a native HLT classifier to predict? Four fresh
CE-only teachers and eight fresh students are the entire twelve-fit campaign.
Every student retains CE + U100 embedding prediction weight 5. Its additional
embedding heads share a total weight of 5, divided equally between targets.
Equal loss coefficients do not imply equal encoder gradients; log both norms
and cosines with CE at epochs 1, 3, 10, 20 and 30.

All fits use the original 299234 TRAIN and 74895 development-validation jets,
native Particle Transformer, batch128, 30 epochs and original optimizer/LR
recipe. Select each fit by original validation-AUC tie rules. This is a
single-seed exploratory comparison, not an independent test. Historical
original50k overlaps validation. Final-test access remains sealed.

## Teacher fields

All teachers retain D000 particle slots/order and use the existing authenticated
algorithmic HLT-to-OFFLINE assignment, not truth particle correspondence. No
offline-only particles are added. Matched U100 slots supply selected field
groups; unmatched slots retain D000 fields.

| Teacher | Privileged group | Fields that stay D000 |
|---|---|---|
| K | p4 and every derived momentum/angular feature | identity, charge, tracking |
| I | identity categories and charge | p4/geometry, tracking |
| T | impact parameters, errors and their source validity convention | p4/geometry, identity/charge |
| IT | identity/charge and tracking together | p4/geometry |

The native17 frontend is separable into K columns 0–4,15–16 plus vectors, I
columns 5–10, and T columns 11–14. Copying these whole groups from authenticated
D000/U100 inputs exactly equals evaluating the native frontend on those raw
field groups. Tests must demonstrate this, including changed jet axes, unmatched
particles, invalid tracking and charge conflicts. Never mix partial K features
or retain a D000 axis after replacing p4. There are no new validity/matching
inputs. I-only/T-only charge/tracking conflicts are deliberate controlled hybrid
views; do not recouple groups or silently sanitize them using the other group.
These feature hybrids can violate the physical-particle applicability rules
at charge conflicts; they are controlled input ablations, not physical detector
endpoints. Original source particles and their validity rules are unchanged.

Teachers train from scratch with matched initialization/sampler seeds. Freeze
the selected teacher; export its clean native pooled128 preclassifier embedding
for TRAIN only in FP32. Normalize each bank independently by its TRAIN FP64
population mean/std (floor .001). Do not average axes across teachers.

## Students and submission order

Each extra target has an independent removable 128→128→128 prediction head
using the existing embedding-prediction architecture (the implementation must
use the donor head exactly). No attention modification or predicted-feature
feedback. Student inference is the original native HLT model, with all heads
removed. All students use the same original U100 student seed and sampler.

| Task | Prediction targets | Additional per-target weight |
|---|---|---:|
| S6 | U100 + K + I + T + IT + existing OFFLINE T00 | 1 |
| S5 | U100 + K + I + T + IT | 1.25 |
| S4 | U100 + K + I + T | 5/3 |
| S3 | U100 + I + T | 2.5 |
| S2IT | U100 + IT | 5 |
| S2T | U100 + T | 5 |
| S2I | U100 + I | 5 |
| S2K | U100 + K | 5 |

Reuse authenticated completed teacher-sweep T00 and S00, plus U100-only C0;
no extra control fits. Report delta C0 and delta S00, and oracle roles explicitly.
The four teacher jobs are independent and submitted first. Then submit students
in the exact table order, from more targets to fewer. Each waits only for its
own required new teacher(s). No student chains or artificial concurrency cap.
Slurm resource/priority decisions can change actual execution order.

## Data, lineage and execution

Reuse the existing Hydra bundle (D000 TRAIN/validation, U100 TRAIN, U100 bank)
and the existing frozen-adapter U100 validation delta located through the
authenticated Group A parent. Normally no new transfer is required.
If those files are missing, export only U100 validation input shards from SPORC, using original
assignments and raw source if deleted. Check every reconstructed shard against
its historical byte checksum and the full native fingerprint; never rewrite
historical sources or rerun matching. Only validation teacher selection uses
these privileged validation inputs. Student validation is always D000.

New versioned namespace: LUKA_SELECTIVE_TEACHER_*/v1. Original T00/S00 must be
authenticated by their original pinned code in a subprocess, avoiding cross-
worktree source guard failures. New banks name their selective view, selected
checkpoint, preparation, spec and recipe. Workers validate roles, ordered IDs,
labels, hashes, source and parents before reuse. Path existence is insufficient.

Hydra override to generic repository Tigris guidance: typed RTX A6000, gpus /
default / normal, one GPU, six CPUs, 32000MiB. Setup performs installed-Weaver
native parity, full input authentication, two-epoch production-worker miniatures
including export/reload, longest-support backward, warmed full-batch timing and
full-validation timing for all twelve profiles. Actual student targets receive
a disposable startup miniature after teachers finish. Scientific weights always
start fresh. Weak finite performance never prevents later registered rows.

Bootstrap submits GPU setup and durable compute-partition CPU release dependent
on successful setup. Release checks measured resources, exact accounting and
receipts, then submits the twelve fits and CPU report automatically. Bootstrap
reviews the full bounded DAG; fits may request only measured times within48h.
Report waits afterany. Exact-ID journals/claims prevent duplicate or ambiguous
retries; failed setup cannot release science. Nothing depends on a login process.
