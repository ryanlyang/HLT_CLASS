# LUKA_U100_PRETRAIN_*/v1

Authority: [active plan](../plans/LUKA_U100_PRETRAIN_PLAN.md).
All records carry canonical hashes, explicit parent hashes and exploratory
ALLJETS flags. SPEC pins source, fixed30 anchor, frozen COARSE_U100 checkpoint,
OFFLINE-w5 reference, exact paired population, seeds, two-stage recipe, debug
allocation and durable budget. BANK pins teacher/recipe/prepared data and
separate TRAIN/validation payloads with shared TRAIN-only normalization.

Stage1 REPORT selects minimum validation SmoothL1 then earliest pass. Its
selected.pt includes predictor AND native state; classifier remains unchanged.
Stage2 REPORT binds that exact full checkpoint through its task parent, resets
optimizer/schedule, and selects by the unchanged AUC/CE/rejection/update rule.
Stage2 selected.pt is native-only. Histories include objectives, epochs,
timings and the exact selected pass. Technical reports cannot initialize or
authorize scientific stage transitions.

TASK receipts authenticate explicit output inventories and exact dependency
receipt hashes. Setup admission verifies the real two-stage technical path,
resources, runtime environment/GPU, compact target bank and comparisons.
Successful setup accounting is required by science even when Slurm dependency
is manually altered. The second stage additionally requires successful
pretrain accounting and the selected full state. Retain intermediate state;
no implicit restart or cleanup. Results authenticate/recompute saved
probabilities, not rounded console numbers. No new final-test interpretation.
