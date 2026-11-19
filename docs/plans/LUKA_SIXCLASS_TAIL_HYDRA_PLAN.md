# Six-class Oscar ladder continuation on Hydra L40

Status: implementation for transfer and queue review; no remote transfer or
submission performed by this implementation step. The user selected one L40.
This execution plan complements, and does not change, the
[six-class Oscar scientific plan](LUKA_SIXCLASS_BALANCED_OSCAR_PLAN.md).

## Scientific scope

Import the completed Oscar OFFLINE, M0HLT, CONCAT and UE100 artifacts associated
with jobs 9198207, 9198208, 9198209 and 9198210. Train **fresh DE050 then fresh
DE000**, each for 20 epochs. The names DE050/DE000 are the code's names for the
half-offline and HLT-only rungs described conversationally as UE050/UE000.
Keep Oscar's pending jobs and the older eleven-class Hydra ladder untouched.
No dataset re-upload, teacher retraining, checkpoint warm start, new seed or
loss-weight change is part of this continuation.

Reuse the frozen six-class population: QCD, X_bb, X_cc, X_ss, X_qq, X_gg;
600 <= offline jet pT < 2600 GeV; twenty 100 GeV bins. TRAIN has 1,658,937
eligible jets and that many replacement draws per epoch. The sampler gives each
of 120 class/bin cells equal probability. Batch size 128 gives 12,961 updates
per epoch, with a final batch of 57. It is equal probability sampling, not an
assertion that every realized cell count is exactly equal.

Primary validation has 328,589 jets with natural class/pT frequencies; sealed
in-range test has 822,313. The frozen extended split roles are preserved.
The 237,443 outside-range validation rows are evaluated only at the selected
checkpoint. Neither the 610,553 outside-range test rows nor the in-range test
are evaluated. No population is refitted or reassigned on Hydra.

The recipe, model, draw streams, initialization seeds, optimizer, schedule,
precision, target normalization, contrastive candidates, AUC selection and
metrics are imported unchanged from Oscar scientific contracts v1.

| Fresh fit | Input view | Main KD teacher | Auxiliary embedding/logit teachers |
| --- | --- | --- | --- |
| DE050 | Half-offline paired view | Imported UE100 | OFFLINE, CONCAT, UE100, M0HLT |
| DE000 | HLT only | New Hydra DE050 | OFFLINE, CONCAT, UE100, new Hydra DE050 |

Each main head uses 0.25 true-label CE + 0.75 T² forward KL at T=2. The four
embedding objectives retain total weight 10 (2.5 each), and the four separate
auxiliary logit objectives retain total weight 1 (0.25 each). Only DE000 adds
OFFLINE contrastive matching, weight 0.3 with its existing three-epoch ramp,
temperature 0.1 and 1,024 negatives. Its occurrence-aware sampler is unchanged.
Only the native classifier is deployed; DE000 inference consumes HLT only.

Report accuracy, macro one-vs-rest AUC, R50 and R30 at the same AUC-selected
checkpoint. Recovery uses the **imported six-class balanced Oscar M0HLT and
OFFLINE**, on identical ordered validation rows. It never uses the older
eleven-class historical anchors. Outside-range metrics remain diagnostics.
L40 and Hydra's recorded runtime may differ numerically from L40S/Oscar; these
are continuations with the same seeds, not independent-seed replications.

## Portable selected-artifact closure

Export authenticates the original clean source snapshot, pinned scientific
donor bytes, original runtime, original spec, original submission bindings,
four successful accounting records, all committed epoch state checksums,
selected reports/checkpoints/predictions, startup acceptance, banks, and the
replayed population. It writes only a new manifest, never changes the campaign.

Transfer the original campaign spec, population metadata, submission JSON,
four result receipts, 80 epoch JSON commits, selected native checkpoints,
reports, validation/outside-validation predictions and metrics, view/startup
proofs, and four complete TRAIN target banks. Do **not** transfer per-epoch
optimizer state payloads, raw ROOT data, unfinished stages or any test outputs.

Keep original JSON bytes and absolute Oscar references intact. A new hashed
bundle maps original campaign-relative paths to received files. Hydra resolves
each original ref through that inventory, checking size and SHA256. It validates
original teacher/selection/lineage metadata and joins every bank against the
canonical TRAIN identities and labels reconstructed using Hydra's existing
verified dataset. State payload checks at export are recorded evidence;
they cannot be rerun on Hydra because those payloads were deliberately omitted.

The binary SSH helper downloads this bundle to Windows, then uploads it to
Hydra using the user's existing private-key path. Private-key bytes never enter
the Python script or bundle. Every received file is hashed. Publication is
atomic, completion is last, and retries preserve matching complete files.
No shell text pipeline carries binary data; no generic tar extraction is used.

## Execution and storage

Two Slurm jobs on `gpus`, account `default`, QOS `normal`:
one `gpu:nvidia_l40:1`, six CPUs, 192000 MiB RAM, 2,880 minutes each.
DE000 has `afterok` on the newly submitted DE050. Cross-cluster dependencies
are represented by authenticated imported artifacts, never Oscar job IDs in
Hydra's dependency expression.

Raw input already exists at
`/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1`.
No persistent second particle cache is created: the existing resident-view
builder runs in each allocation. The four bank arrays alone are about 3.56 GiB;
the export manifest supplies the exact whole-bundle size. Require at least
14 GiB currently free **after** import for new results, plus headroom the user
chooses to reserve for unrelated running campaigns. Check actual filesystem
space as well as declared quota. Committed local output bytes can be credited
on an explicit resume; abandoned attempts cannot.

Registration records the Hydra environment and backend. Torch release and
Weaver version/source must match Oscar; CUDA/Python builds may differ and are
recorded. Each job rechecks its exact allocation, environment and source, then
performs installed-Weaver parity, connected-gradient checks, native deployment
reload, a real two-epoch miniature, full-support memory checks, and a conservative
timing estimate inside that same allocation. There are zero separate preflight
jobs. Failing acceptance preserves diagnostics and does not start the full fit.

Full optimizer/model/CPU-CUDA RNG state and sampled-order receipts are committed
at each new epoch. Explicit resume requires every prior Hydra submission to be
terminal; it skips fully published fits, resumes committed epochs, and can
finish export after epoch 20 without additional optimizer updates. Ambiguous
submission journals require reconciliation and never trigger automatic retries.
Poor metrics are results, never a reason to fail or suppress DE000.

## Implementation and acceptance

Execution and relocation code: `src/hlt_classification/luka_sixclass_tail_hydra/`.
Unchanged scientific workers: `src/hlt_classification/luka_sixclass_balanced_oscar/`.
The donor manifest pins their canonical Git bytes and inherited dependencies.
The [execution contract](../contracts/LUKA_SIXCLASS_TAIL_HYDRA.md),
[runbook](../LUKA_SIXCLASS_TAIL_HYDRA_RUNBOOK.md) and
[validation record](../LUKA_SIXCLASS_TAIL_HYDRA_VALIDATION.md) specify exact use.
