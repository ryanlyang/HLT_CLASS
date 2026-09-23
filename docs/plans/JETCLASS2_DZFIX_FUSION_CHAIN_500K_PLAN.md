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

### GPU-memory repair after preflight 21757056

The capacity-320 preflight reached an actual CUDA OOM in the full cross-pair
embedding on its longest-jet batch, after the short single/fusion miniature
fits succeeded. Raising the 90% gate or CPU RAM request cannot repair that.
The new registered storage policy is `pair_saved_tensors_cpu_v2`: during CUDA
training with gradients enabled, autograd-saved tensors inside the context,
primary and cross pair embeddings are stored losslessly in pinned CPU RAM
using layout-preserving saved-tensor hooks, and copied back when backward needs them. Pair
outputs, attention and the rest of the model remain on GPU. Evaluation bypasses
offload. Nothing is written to disk. There is no pair-population chunking,
recomputation, reduced precision, truncation, smaller batch, or changed BN update.
The shared historical model's default path remains unchanged.

This is an execution change, not a completed A100 acceptance. The fresh gate
must compare storage on/off for three training updates using installed Weaver
in CUDA FP32 and BF16: logits, loss, parameter gradients, all BN buffers,
updated weights, AdamW state and subsequent eval logits. Identical RNG draws
and nonzero fusion projections exercise context gradients; the existing
native-mask comparison remains a separate check. Each paired longest-jet
stress step must prove actual packing and retrieval at all three pair sites.
Counters are cumulative tensor-save bytes, NOT distinct/live CPU memory or
claimed GPU savings. Actual process RSS, allocated/reserved GPU memory and
step timing are recorded. The unchanged 90% GPU/80% CPU and <=23h projected-fit
gates determine whether transfer overhead/headroom are acceptable on SPORC.

### Strict parity repair after preflight 21765886

At source `b35fbda64d2d823a9eb9c5592017074db58d6ac8`, all four batch-256
stress paths completed, including the two fusion paths with actual saves and
retrievals at all three pair sites. The logged allocated GPU peak was
20,816,062,976 bytes (19.39 GiB), versus 41,731,227,648 reserved bytes (38.87 GiB).
The run failed later on strict FP32 pair-BN parameter-gradient parity, not OOM;
it did not publish acceptance or release science.

The old failure is reproducible with installed Weaver 0.5.3 / PyTorch 2.5.1
on a local CUDA device. Even native-versus-native with identical seeds can fail
that comparison under unconstrained CUDA reductions. Parity comparisons now
enable deterministic algorithms and disable TF32 and cuDNN benchmarking in a
scoped context, restoring all flags on success or error. Only the preflight
worker sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts. These are
diagnostic controls, not a new production training recipe. The old tolerances
remain FP32 rtol=2e-5/atol=2e-6 and BF16 rtol=.01/atol=5e-4.

Storage v2 preserves saved tensors' dtype, shape and strides (including
transposed, gapped and overlapping views) by copying their physical storage
span to pinned RAM and rebasing the offset. The original `save_on_cpu` pinned
path made contiguous copies, which can change backward reduction selection.
No GPU tensor references are retained in a packed record; inference and model
state keys are unchanged. Feature gradients for both branches join the parity
checks. Matching, loss, seeds, batch 256, capacity 320 and resource ceilings
are unchanged. The K2 campaign and its separately chosen batch size are not
modified or inherited.

A bounded diagnostic now runs **before** full population caches: authenticate
existing assignments for four distinct registered train jets, build the exact
U050/U000 and D000/D000 views, run native-mask plus FP32/BF16 storage parity,
and publish each successful diagnostic atomically. It never reads final test
or recomputes matching. A failure stops before the long cache build. Successful
early evidence does not replace the subsequent full-population cache, parity,
batch-256 stress, memory/time, bank and checkpoint acceptance. Those mandatory
later checks still run under this campaign's production resource constraints.

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

LAUNCH_SPEC and CAMPAIGN_SPEC advance to v5 for layout-preserving storage and
the scoped parity protocol; ACCEPTANCE advances to v3 and also requires four
campaign-bound EARLY_PARITY/v1 diagnostics. The prior v4/v2 execution is not
silently accepted. SOURCE_IMPORT stays v3: matching/capacity/source reuse is unchanged.
The previous v3 consumer fixed inventory-derived capacity, v2 introduced
direct-screen provenance but incorrectly required 240, and v1 used the canceled
continuation route. Old roots must not be edited or reused; a fresh pinned
commit creates fresh output roots. The producer's screen schema remains v2.
Model, seed, split, training and scientific output meanings are unchanged.

For this retry, use the same completed debug screen through the queue helper;
its authenticated durable completion removes the need for a purged Slurm job
dependency. Do not rebuild matching or overwrite the failed campaign at
`2f0afe5c` or `b35fbda6`. The user previously canceled old after-gate job
21757081; do not reuse that historical ID as current cleanup guidance.
Any cleanup of a newly blocked after-gate requires its exact current ledger.
The helper does not cancel jobs or
touch any `jc2k2`/`jc2salp` jobs. Only the new, source-pinned acceptance may
release this new campaign's science DAG.

The initial queued job is `jc2fc_after_matching`; the full science DAG is
submitted automatically after matching and this campaign's fresh gate succeed.
Inspect only this family with `squeue --me -o "%.18i %.58j %.2t %.10M %R"` and
`grep -E 'JOBID|jc2fc_'`. The CLI's `gate` and `results --per-class` subcommands
accept the new `campaign_spec.json`. Logs retain the shared `JC2-LFH node=...`
training prefix. Re-running a successfully submitted stage verifies and returns
its exact ledger instead of submitting duplicates. An interrupted ambiguous
sbatch acknowledgement intentionally requires operator reconciliation.
