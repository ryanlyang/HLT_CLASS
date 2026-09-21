# JetClass2 dz-fix coarse fusion-to-fusion KD chain

Status: implementation plan; real SPORC/debug acceptance is required before
science submission. This is a new isolated experiment, not a recovery of CMS.

## Scientific registration

Use the authenticated `20260918_dzfix` partial snapshot and exact TRAIN_500K
membership: 500,000 training, 1,000,000 validation, and 1,000,000 sealed test
jets. Import the winner directly from the completed source-pinned v2 debug
salience screen. The old tier3 continuation was canceled. Do not rerun matching, choose a candidate
early, reuse September-10 artifacts, or import any CMS checkpoint or metric.

Views retain the persistent HLT skeleton. U removes unused offline tails;
matched U100 slots still have offline features. D transitions those matched
features to HLT. D000 is the native HLT endpoint. The input contract remains
17 features, 11 classes, and no source/matching metadata inputs. Capacity is
derived from authenticated inventory count metadata using the foundation
builder's round-up-to-16 rule (minimum 16), never a historical snapshot constant.
All three salience candidates and the bottleneck control must carry that exact
input contract, including forbidden truncation. The current dzfix foundations
have capacity **320**, as verified on SPORC; the previous 240 statement was a
stale assumption and is superseded. This does not rebuild matching or read test
particles. Cache bounds and real GPU stress use the selected foundation's capacity.

In the following notation, Fusion(context, primary) uses independent encoders
with one-way gated residual cross-attention into the primary encoder at blocks
2, 4, 6, 8. Fusion stays at alpha=1 throughout training and validation.
The CMS temporary-memory optimization is retained: compute the full Weaver
pair embedding (including its full BatchNorm population), then materialize the
rectangular cross-bias and merge padding once. No feature, loss, batch-size or
attention change; native forward/backward parity is acceptance evidence.

```
fresh U000 CE
  -> Fusion(U000,U050)             fresh; NOT the old CMS ACQUIRE_U050
  -> Fusion(U050,U100)
  -> Fusion(U100,D066)
  -> Fusion(D066,D033)
  -> Fusion(D033,D000)
       |-> single D000
       `-> Fusion(D000,D000) -> single D000
```

Every arrow is train-population logit KD from the selected parent checkpoint,
with C25/P75 and temperature 2, not warm initialization. Each fit starts fresh.
There is no context withdrawal, alpha-zero selection, intermediate compression,
mixture weight optimization, or probability ensemble. Same-view fusion has two
independently initialized branches but identical native HLT inputs. Both final
single models use identical initialization and shuffle seeds.

Fresh controls are M0HLT CE, pure OFFLINE CE, persistent U000 CE (the chain
anchor), and ordinary U000-to-D000 KD. No seed ensembles or other ablations
are added. Census: 12 fits, 7 train-probability reducers, 2 summaries; 21 science
tasks. References, anchor, direct comparator, and endpoint branches are allowed
to run independently when their exact dependencies have completed.

## Training and evaluation

Reuse the registered learned-handoff AdamW/BF16 implementation: batch 256,
warmup passes 1-3, LR 3e-4 through 45, cosine decay through 60 to 1.5e-5,
constant floor through 100, minimum 60, patience clock starting at 60,
patience 15 and minimum AUC improvement 5e-5. Restore selected weights;
selection uses AUC, CE, R50 and update tie-breakers. No rolling resume.

Partition validation deterministically by class and identity into 50%
checkpoint, 25% diagnostic (reserved), 25% report. Fit selection/early stopping
use checkpoint only; tables use report only. This partition is shared by all
new fits. Matching selection already used this validation reservoir: report is
not an untouched test or an unbiased estimate after matching selection.
Final-test input/inference remains unavailable to ordinary workers.

Report accuracy, macro AUC, linear R50 and recovery with M0HLT=0% and pure
OFFLINE=100%; label persistent U000 separately. Bad finite performance does
not fail or prune a fit. Corruption, invalid inputs or lineage fail closed.

## Execution and isolation

All new launchers, gates and science jobs target SPORC `debug`, one A100 per
GPU task, account reu-aisocial, qos_tier3, no requeue. No job exceeds debug's
24-hour limit. The campaign requests 8 CPUs/320000 MiB for GPU jobs; its own
real paired-view acceptance must demonstrate this is sufficient, not infer it
from the old CMS or single-view screen. RAM-only native/paired caches have
explicit population bounds; durable artifacts are weights, reports, compact
probability banks and hashes only. No particle/hidden-state cache is persisted.

Creation locks the active `screen_spec.json` and its exact eight-task live
ledger. The queue launcher depends on that ledger's `complete` job (currently
21748725), never on one fit, the canceled 21741416, or `afterany`. When the
durable screen is already complete, authenticate it and omit the expired Slurm
dependency. Foundation paths and identities come from the screen's candidate
registry and selection lock, never a guessed continuation subtree. No old
continuation receipt or 30-task production preview is required or reused.
After completion, authenticate screen selection and compact foundation; do a
full campaign dry run; submit authenticate/partition/preflight gates. A separate
success-dependent launcher validates their artifacts and submits science.
No parent jobs, roots, ledgers, matching outputs or CMS artifacts are modified.
The source screen deliberately records debug measurement and tier3 production;
that tier3 setting applies to its separate three-spine production consumer.
This independent fusion experiment keeps its own explicitly registered debug
site/resources and must pass its own real gate before scientific submission.

New execution evidence includes full-population paired cache construction,
native installed-Weaver single and paired optimizer steps, worst-length batch
stress at batch 256, train-bank publication/readback, finite inference and
checkpoint round trip. CPU/GPU peak headroom and epoch-runtime projection must
fit the registered debug resources. Failure stops science without changing
batch size, precision, schedule or population. An explicit new specification
is required if resources/science need to change.

Cross-commit preparation reuse has its own versioned source-import contract.
It authenticates the producer screen, exact ledger, completed screen, selector, foundation
and source files; it does not relax the old same-commit campaign validators.
Only preparation is reused. All scientific fits and GPU acceptance are fresh.

## Completion boundary

Local tests and source packaging establish queue tooling readiness, not measured
GPU science readiness. A live submission requires an exact pushed clean checkout,
explicit authorization and canonical dry-run ledgers. Real debug acceptance
is the automatic prerequisite to science, not a claim made by local tests.

## Queue entry point

After committing/pushing this implementation, make a clean detached worktree
at that exact pushed commit on SPORC. From that worktree:

```bash
bash scripts/queue_jetclass2_dzfix_fusion_chain.sh
bash scripts/queue_jetclass2_dzfix_fusion_chain.sh --execute
```

The first call is dry; the second explicitly authorizes the registered pipeline.
The helper defaults to the active screen
`checkpoints/jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json`
from producer commit `0d25a4a53aafb1348c8279d86bac7dbac82c8841` and the
unchanged dz-fix inventory. `SCREEN_SPEC`, `INVENTORY`, `LAUNCH_ROOT`,
`CAMPAIGN_ROOT` can explicitly override
locations. Their content and lineage are validated, not trusted by existence.
It never fetches into, updates, cancels or writes an existing campaign.

LAUNCH_SPEC, SOURCE_IMPORT and CAMPAIGN_SPEC advance to v3 to lock the
inventory-derived, no-truncation capacity policy. v2 introduced direct-screen
provenance but incorrectly required capacity 240; v1 used the canceled
continuation route. Old v1/v2 roots must not be edited or reused; a fresh pinned
commit creates fresh output roots. The producer's screen schema remains v2.
Model, seed, split, training and output-artifact semantics are unchanged.

The initial queued job is `jc2fc_after_matching`; the full science DAG is
submitted automatically after matching and this campaign's fresh gate succeed.
Inspect only this family with `squeue --me -o "%.18i %.58j %.2t %.10M %R"` and
`grep -E 'JOBID|jc2fc_'`. The CLI's `gate` and `results --per-class` subcommands
accept the new `campaign_spec.json`. Logs retain the shared `JC2-LFH node=...`
training prefix. Re-running a successfully submitted stage verifies and returns
its exact ledger instead of submitting duplicates. An interrupted ambiguous
sbatch acknowledgement intentionally requires operator reconciliation.
