# JetClass2 dz-fix coarse fusion-to-fusion KD chain

Status: implementation plan; real SPORC/debug acceptance is required before
science submission. This is a new isolated experiment, not a recovery of CMS.

## Scientific registration

Use the authenticated `20260918_dzfix` partial snapshot and exact TRAIN_500K
membership: 500,000 training, 1,000,000 validation, and 1,000,000 sealed test
jets. Import the winner of the existing three-candidate salience screen through
the completed dz-fix continuation. Do not rerun matching, choose a candidate
early, reuse September-10 artifacts, or import any CMS checkpoint or metric.

Views retain the persistent HLT skeleton. U removes unused offline tails;
matched U100 slots still have offline features. D transitions those matched
features to HLT. D000 is the native HLT endpoint. The input contract remains
17 features, 11 classes, capacity 240, and no source/matching metadata inputs.

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

Creation locks an explicit source continuation and exact after_screen ledger.
The queue launcher depends on that ledger's after_screen job (documented as
21741416), never on one fit or `afterany`. When the durable continuation is
already complete, authenticate it and omit the expired Slurm dependency.
After completion, authenticate screen selection and compact foundation; do a
full campaign dry run; submit authenticate/partition/preflight gates. A separate
success-dependent launcher validates their artifacts and submits science.
No parent jobs, roots, ledgers, matching outputs or CMS artifacts are modified.

New execution evidence includes full-population paired cache construction,
native installed-Weaver single and paired optimizer steps, worst-length batch
stress at batch 256, train-bank publication/readback, finite inference and
checkpoint round trip. CPU/GPU peak headroom and epoch-runtime projection must
fit the registered debug resources. Failure stops science without changing
batch size, precision, schedule or population. An explicit new specification
is required if resources/science need to change.

Cross-commit preparation reuse has its own versioned source-import contract.
It authenticates producer continuation, completed screen, selector, foundation
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
The helper defaults to the handoff's exact dz-fix continuation and inventory;
`CONT_SPEC`, `INVENTORY`, `LAUNCH_ROOT`, `CAMPAIGN_ROOT` can explicitly override
locations. Their content and lineage are validated, not trusted by existence.
It never fetches into, updates, cancels or writes an existing campaign.

The initial queued job is `jc2fc_after_matching`; the full science DAG is
submitted automatically after matching and this campaign's fresh gate succeed.
Inspect only this family with `squeue --me -o "%.18i %.58j %.2t %.10M %R"` and
`grep -E 'JOBID|jc2fc_'`. The CLI's `gate` and `results --per-class` subcommands
accept the new `campaign_spec.json`. Logs retain the shared `JC2-LFH node=...`
training prefix. Re-running a successfully submitted stage verifies and returns
its exact ledger instead of submitting duplicates. An interrupted ambiguous
sbatch acknowledgement intentionally requires operator reconciliation.
