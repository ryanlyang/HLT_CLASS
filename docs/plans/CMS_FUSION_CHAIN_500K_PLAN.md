# CMS adjacent fusion-to-fusion KD chain, with paired terminal routes

## Authority and isolation (2026-09-20)

The user requested this exact first ladder experiment. It is a new SPORC/debug
campaign alongside, not replacing, the running CMS coarse withdrawal and direct
fusion campaigns. Do not cancel or edit their jobs, source checkouts or artifacts.
This plan governs only `ladder=fusion_chain`; old campaign identities stay fixed.
No intermediate withdrawal, learned gate schedule, output ensemble, weight
selection, extra seed panel or new CE control is included.

## Scientific graph

Notation below is `(context, primary)`. The primary branch owns the classifier;
the richer context injects the existing one-way attention residuals at blocks
2/4/6/8. Every arrow is frozen-teacher logit KD, not a weight transfer:

```text
ACQUIRE_U050: (U000, U050)                 [import completed coarse acquisition]
  -> FUSION_U100: (U050, U100)
  -> FUSION_D066: (U100, D066)
  -> FUSION_D033: (D066, D033)
  -> FUSION_D000: (D033, D000)
       |-> FINAL_D000_DIRECT: single D000
       `-> FUSION_D000_D000: (D000, D000)
               `-> FINAL_D000_BRIDGE: single D000
```

All seven new fits are cold-started. The imported acquisition was originally
cold-trained with U000 KD. New fusion nodes use the existing ordinary paired
CE/KD kernel, with alpha fixed at one and the node role `fusion_pair_kd`.
This role deliberately does not run the historical context-permutation
diagnostic, whose view-name dictionary cannot independently permute two inputs
with the same coordinate. No prediction or loss kernel is changed.

At D000/D000 both branches receive exactly the same genuine HLT tensors, but
have separate parameters. This is learned fusion, not a prediction average.
It is HLT-only but still a two-branch architecture. Both final models are
ordinary single-ParT HLT-only models with identical D000 initialization,
sampler and dropout seeds; only their frozen teacher differs. The bridge arm
costs one additional fit and is not compute matched to direct compression.
No result is assumed superior, and poor metrics never prune either branch.

## Fixed data and recipe

Reuse original CMS/Scouting (21 features, 15 classes), not JetClass2. Keep the
exact 500k train / 250k validation / 250k sealed-test registration, linear
salience matching, persistent-HLT support, train-only scales and authenticated
preparation from the existing native campaigns. U000 is the persistent anchor;
OFFLINE is the separate pure-offline reference. D066/D033 are exact thirds.

Every new fit uses C25/P75, T=2, batch 256, one A100, BF16 forward/FP32 loss,
the existing AdamW recipe, warmup through pass 3, hold 3e-4 through 45, cosine
to 1.5e-5 through 60 and floor through 100. Preserve minimum/patience rules:
the patience clock starts at 60, significant AUC delta 5e-5, patience 15,
earliest stop 75, restore best checkpoint. Paired models validate with both
views at alpha one, single models with D000 only.

Keep the fixed validation checkpoint/diagnostic/report partitions, 50/25/25.
Only checkpoint rows select epochs. Print all registered nodes on REPORT rows,
including accuracy, AUC, geometric macro R50 and per-class QCD rejection;
recovery uses M0HLT=0%, OFFLINE=100%, linear R50 space. Distinguish privileged
fusion, HLT-only two-branch and HLT-only single-ParT rows explicitly. Report
both final-minus-DIRECT_D000 and bridge-minus-direct-final deltas. The final
test remains sealed; this exploratory follow-up does not claim unseen test
performance or statistical significance from previously inspected validation.

## Reuse, versions, and queue safety

CAMPAIGN_SPEC/v7 and GRAPH/v4 register this separate graph, with `cmsfc_` names.
There are 21 science tasks: seven CPU imports (five original dense shared
tasks plus the completed coarse acquisition fit and its bank), seven fresh
fits, five fresh reducers, aggregate and completion. Four imported reference
fits plus the imported first acquisition make twelve logical fits in all.
Import M0HLT, OFFLINE, U000, reduce_U000 and DIRECT_D000 from the original
accepted dense source; import only ACQUIRE_U050 and its reducer from an
explicit completed coarse-v5 source. Both source roots remain read-only.

New PREPARATION_IMPORT/v4, SHARED_SOURCE/v3, ACCEPTANCE_IMPORT/v3 and
ACCEPTANCE_REUSE/v3 distinguish this consumer. ACQUISITION_SOURCE/v1 binds
the coarse spec, source commit, canonical live ledger, exact node/recipe,
completed receipts, report, checkpoint, teacher lineage and bank payloads.
The two acquisition imports must already be complete when creating the new
campaign. They need no dependency on aged-out Slurm IDs and never wait on
withdrawal. Imports publish copied compact artifacts and new consumer-bound
receipts with explicit original hashes; no source artifacts are relabelled.

Reuse the genuine accepted dense U000/U000 longest-batch GPU evidence from
job 21720795 only with the same strict code/data/resource/site checks. The
ordinary paired KD and single-view KD routes were exercised there, including
same-coordinate paired inputs. Every new view is within that registered
envelope. Kernel, worker, batch, 85% CPU / 90% CUDA policy and exact preparation
remain unchanged. This is evidence reuse, not a new GPU measurement. The named
installed environment is assumed unchanged; incompatible evidence fails closed.

Creation and `prepare` are CPU verification/dry-run only. `prepare` first
requires installed Weaver and runs the focused same-view CPU parity test;
it does not submit a duplicate GPU miniature or refit any reference. Live submission
requires clean pushed pinned source, an accepted imported gate, exact dry plan
and `AUTHORIZE CMS FUSION CHAIN 500K EXACT SPEC`. Use the crash-safe exact-ID
submission journal; rerunning submit must not duplicate recorded jobs. No
cancellation, scheduler resource mutation or existing-root overwrite exists
in the helper. Ordinary workers stay final-test sealed. Existing retirement
commands must reject this parallel campaign before any scheduler operation.

## Verification

Test unchanged old graph hashes; the exact new edges, seeds and two endings;
seven fresh fits; debug-only plans; complete read-only acquisition import and
bank joins; missing, corrupt, wrong-source and stale-kernel rejection; canonical
source ledgers; same-view paired-cache/tensor/gradient behavior; matched final
initialization; frozen teacher banks; all tasks in a tiny production-path run;
idempotent/partial submission; report recovery and sealed test. Exercise the
new same-view model with installed Weaver when available. Synthetic local
tests are not new SPORC acceptance. Record actual test and reuse evidence in
HANDOFF and donor paths/commit in LEGACY_SOURCE_MAP.

## Operator entry points

After a scoped commit/push and fresh detached SPORC checkout, use the existing
coarse `campaign_spec.json` as the source (not its withdrawal checkpoint and
not the standalone direct-fusion spec):

```bash
bash scripts/queue_cms_fusion_chain.sh prepare "$COARSE_SPEC" "$NEW_CHAIN_ROOT"
bash scripts/queue_cms_fusion_chain.sh submit "$NEW_CHAIN_ROOT/campaign_spec.json"
bash scripts/queue_cms_fusion_chain.sh results "$NEW_CHAIN_ROOT/campaign_spec.json"
bash scripts/queue_cms_fusion_chain.sh status "$NEW_CHAIN_ROOT/campaign_spec.json"
```

The known donor is
`/home/ryreu/atlas/HLT_Classification/checkpoints/cms_salience_learned_coarse_500k_debug_reuse_7cf69030_r1/campaign_spec.json`.
Use a new root for this study, never that source directory. `prepare` verifies
the imported runtime/preparation evidence locally and creates the full dry
plan, but queues nothing. `submit` publishes only the 21 new science jobs;
the five reference tasks and first acquisition/reducer are explicitly CPU
imports, not duplicate training. Rerunning `submit` uses the exact journal.
Keep unrelated local matching/scouting edits out of this scoped commit:
scientific kernel compatibility is deliberately strict and fails closed if
the source differs from the accepted donor.
