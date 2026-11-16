# New FullSim balanced Oscar ladder with deployed-head KD

Status: implemented and locally validated; queue-ready after verified transfer
and source push. No jobs submitted. This plan records the
user-authorized new-data six-model experiment. It is separate from both the
running Hydra ladder and the old-data single-DE000 main-KD experiment.

## Population and evaluation

Use `luka_qcdrun2com_higgs2p_sb_v1`, authenticated transfer manifest
`7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4`.
Raw input lives at `/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1`.
Reuse the unchanged Hydra population builder, domain and POPULATION/v1 contract:
foundation `2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`,
balanced population `7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`.
No reassignment of any existing TRAIN/validation/test row; only the previously
approved unused-row extension. Label 206 stays quarantined.

TRAIN: 2,425,536 distinct eligible jets, 2,425,536 replacement draws per epoch.
Choose uniformly among 11 classes and 18 100-GeV offline-pT bins in
`200 <= pT < 2000`, then uniformly among rows in that cell. Preserve the Hydra
PCG64 stream and all seeds. No additional loss reweighting. Twenty epochs,
batch 128 (last batch 64), 18,950 updates/epoch, unchanged AdamW and two-epoch
warmup/cosine schedule, BF16 forwards and FP32 losses.

Select on the same natural, unweighted 500,000 in-range validation rows.
The 136,976 original outside-range validation rows are diagnostics evaluated
at that selected checkpoint. Report accuracy, macro AUC, R50/R30 with passing
counts, and per-bin metrics. Use this campaign's M0HLT/OFFLINE anchors.
The 1,200,000 in-range and 413,361 outside-range test rows remain sealed.
Row disjointness is verified; event independence and old-dataset overlap are
not claimed.

## Six fresh models

| Stage | Inputs | Main KD teacher | Embedding and auxiliary-logit teachers |
| --- | --- | --- | --- |
| OFFLINE | pure offline | none; CE only | none |
| M0HLT | native HLT | none; CE only | none |
| CONCAT | source-blind full HLT/offline particle union | none; CE only | none |
| UE100 | existing U100 view | OFFLINE | OFFLINE, CONCAT, M0HLT |
| DE050 | existing D050 view | UE100 | OFFLINE, CONCAT, UE100, M0HLT |
| DE000 | native HLT | DE050 | OFFLINE, CONCAT, UE100, DE050 |

Each student uses
`0.25 CE + 0.75 T² KL(predecessor || main) + (10/N) sum embedding SmoothL1
+ (1/N) sum auxiliary T² KL`, with T=2. The frozen predecessor intentionally
supervises both the deployed classifier and its separate auxiliary logit head.
Teacher probabilities are exported at T=2 and are not softened twice. CE-only
baselines retain coefficient 1. Native embeddings, unique-TRAIN FP64 moments,
std floor .001, predictor architecture, teacher order and auxiliary seeds are
unchanged. The exact OFFLINE -> UE100 -> DE050 -> DE000 chain is mandatory.

Only DE000 also uses OFFLINE contrastive matching, weight .3 with the existing
three-epoch ramp, temperature .1 and 256 hard / 256 same-class / 512 global
distinct negatives. No subclass loss or additional teachers. Auxiliary heads
are removed at deployment. M0HLT and DE000 inference consumes HLT only.

## Execution and recovery

Separate `luka_new_balanced_oscar` package and versioned execution/training
artifacts; donor population and sampling semantics stay unchanged. Six science
jobs, no standalone preflight or setup jobs. One L40S, six CPUs, 192000 MiB RAM,
48 hours per fit; all real resources are checked in each allocation. Dependencies
include every teacher. No historical model substitutes and no Hydra baseline
prerequisite. Persistent source/results in HOME, raw data in scratch, particle
views only in RAM. Reserve 14 GiB for new outputs and inspect current free quota.
Use an explicitly supplied healthy runtime, fingerprint it and reject drift.
New launches default to a persistent HOME runtime, not the scratch repair prefix.
The 2026-11-16 startup failures had missing Python standard-library files in that
scratch prefix. Runtime restoration may rebuild the exact recorded versions and
Conda builds in HOME, with downloads in scratch. It must match the original
runtime fingerprint before use. A changed environment path requires a fresh
execution spec; never rewrite the failed spec. The startup-only retry requires
all old jobs terminal and no saved training states. Science and population are
unchanged; there is no additional queued preflight.

Integrated acceptance inside each allocation performs installed-Weaver parity,
a two-epoch objective/reload miniature, main-classifier gradient checks, longest
input support, matching parity, memory measurements and timing projection with
headroom. Probe state is discarded before science. Local tests do not claim
measured Oscar readiness; this runtime acceptance remains required, without
another queued preflight.

Authenticate clean pushed source and immutable data before queueing. Full dry
review and `sbatch --test-only` precede actual submissions; journal each intent
and exact returned ID. Detached launch survives the watching SSH session.
Resume only after exact prior jobs are terminal; preserve ambiguous journals.
Complete epoch checkpoints include optimizer and CPU/CUDA RNG. Resume unfinished
fits at committed epoch boundaries; completed fits with failed export finalize
without repeating training. Poor scientific performance never gates a job.
