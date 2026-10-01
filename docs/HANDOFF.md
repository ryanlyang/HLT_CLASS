# Current Handoff

## 2026-09-22: K2 strict CUDA parity repair; early checks before full caches

Scope is **K2 only**. User supplied failure 21767292 at source
`49516092a646cfe21f7bb1377b07a142161eb3d2`: FP32 gradient parity in
`mod.pair_embed.embed.0.weight`, maximum absolute difference `1.5348196e-5`
against unchanged `atol=2e-6, rtol=2e-5`. Peak allocated CUDA memory was
694701568 bytes; no batch-128 or batch-256 probe ran. This was not an OOM.

Investigation with isolated installed Weaver 0.5.3 / PyTorch 2.5.1 on the local
RTX 2000 Ada reproduced strict FP32 failures with longer synthetic inputs,
including a native-versus-native repeat with equal RNG seeds. Preserving
strides alone did not fix those failures. Deterministic CUDA algorithms and
full FP32 parity settings, together with layout-preserving offload, pass the
same strict comparisons. This explains a reproducible failure mechanism;
the exact remote jets/A100 still require the new gate.

Changes:

- `concat_k2_model.py`: preserve saved tensor shape/strides/dtype in pinned CPU
  storage. Compare native/optimized training with deterministic algorithms,
  no TF32 and no cuDNN benchmarking, then restore the original backend flags
  and full matmul-precision policy even on exceptions. No tolerance widening.
- New `concat_k2_parity.py`: authenticate existing K2 maps and read four
  distinct registered train jets; check real native-wrapper and three-update
  FP32/BF16 storage parity on D100 and offline-free D000 **before** full
  expanded caches. Persist each completed early report independently.
- `concat_k2_runtime.py`: require that early evidence plus all original
  population parity, ordered 128/256 probes, resource and round-trip gates.
  No silent batch fallback and no automatic acceptance after a failure.
- `concat_k2_campaign.py`, `concat_k2_preparation_import.py` and the K2 worker:
  launch/campaign v5, acceptance v4, storage policy v2, EARLY_PARITY/v1; bind
  parity backend/tolerances. Set cuBLAS workspace before Python in preflight
  only. Original freshly computed v1-v5 preparation donors remain supported.
- Focused memory/parity/reuse/execution tests, active plan, contract and donor
  map updated. Matching producer files, data, science seeds/model/loss/LR and
  physical production batch 256 were not changed by this repair.

Final focused memory plus installed-native endpoint tests: **53 passed** in
37.26s (includes real local pinned CUDA transfers and installed Weaver).
FP32/BF16 three-update checks cover ordinary and x3 duplicate inputs up to 576
tokens. Tests also cover transposed/gapped/overlapping/expanded saved views,
backend restoration, corrupt evidence and unchanged 128-before-256 handling.
The real synthetic-ROOT early sampling/failure suite passed **5 tests** in
36.87s. The broader K2 campaign, memory, early-parity, source, execution and
reuse regression passed **227 tests, no skips** in 394.78s with installed
Weaver available. The final 53-test focused rerun additionally covers every
restored matmul-precision mode; these counts overlap, not independent fits.
Bash syntax, scoped whitespace checks and CLI help pass. These local results
are not SPORC acceptance.

Next remote step: commit/push the scoped repair, use a new pinned checkout and
fresh launch/campaign roots, and set `K2_PARTITION=debug` with the original
completed `jc2_dzfix_concat_k2_1f930650_r1/campaign_spec.json` as
`K2_REUSE_SPEC`. Do not use the nested-import `49516092` root as the donor.
No assignment jobs are needed on that verified-import path. Pending jobs can
still move debug <-> tier3 without changing other resources. New A100 preflight
and batch probes remain mandatory before the automatic science release.
No remote jobs were submitted, canceled or changed, and no final-test rows
were accessed. No full-batch A100 fit or throughput claim is made.

## 2026-09-22: opt-in completed K2 preparation reuse

The user authorized reuse after noticing that the `df29abcc` replacement
campaign was recomputing assignments. This block is **K2 only** and does not
cancel, submit or alter any remote job. The already queued replacement remains
unchanged until a separately authorized exact-ID cutover.

Added `concat_k2_preparation_import.py` and `--reuse-preparation-spec` /
`K2_REUSE_SPEC` to the existing K2 create/queue path. The explicit donor may be
the completed original `jc2_dzfix_concat_k2_1f930650_r1/campaign_spec.json`;
its failed GPU preflight is irrelevant to reuse of its authenticated completed
matching foundation. There is no implicit latest-root discovery. Original
freshly computed K2 v1/v2/v3/v4 donors are supported; nested imports and
one-to-one salience maps are rejected.

Validate matching producer bytes, selected formula/source, complete preparation
receipt closure, population, capacities, maps and lock. Copy compact arrays
without links into new owned artifacts, reparent reports with explicit donor
lineage, inventory every copied payload, and rebuild the new lock. The donor
is never written. Matching/data producer files are unchanged by this block.

The new graph has six gates (`authenticate`, `import_preparation`,
`foundation_lock`, `partition_validation`, `audit_storage`, `preflight`) and
the unchanged 17 science tasks. No assignment or matcher jobs are scheduled
on the import path. Fresh GPU acceptance remains mandatory, including physical
128-before-256 probes; science batch remains 256. The import is CPU-only
(metadata resource class). Debug/tier3 partition-only portability is retained.
Launch/campaign specs become v4; PREPARATION_IMPORT is v1; acceptance stays v3.

Local evidence: focused K2 campaign/source/execution/memory/reuse suites:
**193 passed, 2 skipped** in 278.42 seconds under `tagging-hlt` with a fresh
workspace pytest basetemp and bytecode disabled. Both skips require installed
Weaver, unavailable locally. Coverage includes real synthetic ROOT map/input
parity at every rung, unchanged donor bytes, no matching/ROOT reads during
import, no assignment jobs, required fresh GPU gates, original donor v1 through
v4, portable dry plans and corrupted/incomplete/wrong-family donor rejection.
Queue-helper Bash syntax, CLI help and scoped whitespace checks pass.

Changes cover the K2 campaign/source/runtime, new import module, thin CLI and
queue helper, focused tests and plan/contract. Baseline donor and provenance
are recorded in `docs/LEGACY_SOURCE_MAP.md`. No SPORC GPU pass is claimed by
this local implementation. The next remote step is a new pushed, pinned
checkout and fresh debug launch with the explicit donor, followed by its
real import and memory-preflight gates; do not edit an existing immutable spec.

## 2026-09-22: K2 pair-storage repair and ordered 128/256 GPU probes

Scope is **K2 only**, not the separately repaired jc2fc fusion chain. The
user-supplied log for old K2 preflight 21757208 at source
`1f9306504dfd040c9c22e0a89829d277d1ff2194` shows actual CUDA OOM in native pair
embedding: next allocation 8.22 GiB, free 1.77 GiB on a 39.52 GiB A100. Its
completed cache construction and small execution fits did not certify the
production-size longest batch. This was not merely the headroom threshold.

Added K2-only `concat_k2_model.py`: instance-local native pair saved tensors
are stored in pinned CPU RAM and restored for backward. Full pair/BatchNorm
population, dtype, state keys, seeds, CE/KD recipe and particles remain
unchanged. No recomputation, microbatching, clipping or other-campaign model
change. Eval/no-grad bypasses the storage path. Adapted from the committed
local storage pattern at `b35fbda64d2d823a9eb9c5592017074db58d6ac8`; precise
donor files are recorded in `LEGACY_SOURCE_MAP.md`.

`concat_k2_campaign.py` now binds pair-storage/probe policies and v3
LAUNCH_SPEC/CAMPAIGN_SPEC/ACCEPTANCE. Scientific view/matching/seed identities
remain v1. `concat_k2_runtime.py` uses this model for fits, reduction and the
fresh gate. Preflight verifies native FP32/BF16 three-update training parity,
then tests longest real batches **128 before 256**, with three optimizer steps
and validation per batch and fresh model/optimizer states. Each probe is
published immediately, including failures. A 256 OOM preserves successful
128 evidence and prevents science release. A 128 OOM prevents the 256 attempt.
There is no silent batch fallback: production stays 256 pending a separately
registered decision. Actual CUDA save/retrieval counters prevent no-op claims.
The 90% GPU, 80% CPU and projected 23-hour fit gates remain unchanged.

Pre-existing uncommitted K2 partition-portability work is preserved, including
its source/submit/CLI/shell changes and execution tests. Queue the requested
replacement explicitly with `K2_PARTITION=debug` and fresh roots at a clean
pushed commit. Pending-job partition-only moves debug/tier3 remain allowed;
old pinned jobs cannot acquire the new code via an scontrol update. Nothing
was submitted, canceled, migrated or changed on SPORC by this implementation.

Local evidence: final K2 regression passed **168 tests with two Weaver skips**
(87.80s), covering the K2 memory, campaign, source and execution test files.
The memory subset separately passed 20 tests with one Weaver skip before the
last distinct-row probe test was added; counts overlap. These tests exercise
actual pinned CUDA transfers on the local RTX
2000 Ada, FP32/BF16 optimizer/gradient/BN parity, hook cleanup, incomplete/OOM
probe publication, unchanged checkpoint keys and fail-closed evidence checks.
An isolated synthetic pair test measured 102501888 bytes native versus
66255872 bytes offloaded (~35% lower); this is **not** a full-Weaver/A100
memory or performance result. A pre-existing CPU test's timing extrapolation
was made deterministic without relaxing the real runtime gate. CLI help,
both Bash syntax checks and scoped whitespace checks passed. Actual
installed-Weaver and genuine SPORC acceptance remain the next required
validation. No speedup or full-batch A100 fit is claimed.

## 2026-09-22: dzfix fusion pair saved-tensor memory repair implemented locally

The user-reported SPORC/debug preflight 21757056 at pinned source
`2f0afe5c438cbab5d66e047c4d86534aab02032b` failed with a real CUDA OOM in
the native full-pair embedding during the batch-256 longest-jet stress.
It was not merely the 90% acceptance threshold. Its dependent launcher
21757081 cannot advance after that failure; unrelated `jc2k2` and `jc2salp`
jobs are outside this repair.

The dzfix-only fusion adapter now stores tensors saved by autograd inside
context/primary/cross pair embeddings in pinned CPU RAM and restores them
for backward, using PyTorch's native saved-tensor hooks. Pair computations,
full combined pair/BatchNorm populations, model state keys, dtype and inputs
are unchanged. There is no recomputation, microbatching or particle cropping.
The shared fusion model's new pair hook defaults to the historical native
call, so other campaign adapters do not enable this policy. Eval/no-grad
inference bypasses offload entirely.

Batch 256, capacity 320 without truncation, C25P75/T2, seeds, LR/early stopping,
500k/1M/1M population, sealed final test, debug resource requests, 90% GPU
headroom and 23-hour projected fit ceiling are unchanged. CPU/GPU memory
and timing must still be measured by the fresh genuine A100 gate; CPU
transfers can cost time, and no production speedup or memory-fit claim is made.

LAUNCH_SPEC/CAMPAIGN_SPEC are now v4 and ACCEPTANCE is v2; SOURCE_IMPORT stays
v3. Acceptance requires installed-Weaver FP32 and BF16 CUDA parity over three
AdamW updates (logits, loss, gradients, BN state, weights and optimizer state),
plus three actual batch-256 longest-population stress updates. Each pair site
must record real CUDA saves and backward retrieval. Memory samples include
allocated/reserved GPU peaks and CPU RSS. Old/no-op/CPU-only offload evidence
cannot authorize science. Transfer counters are not unique live-memory sizes.

Changed surfaces: `dzfix_fusion_{model,chain,runtime}.py`, the native-default
hook in `salience_learned_model.py`, fusion-chain tests and new
`tests/test_jetclass2_dzfix_fusion_offload.py`, plan, contract and donor map.
Repository-local donor/baseline is
`6172f5be459f0a0f7b6c3ed99ee3c6f8ac7e542c`; no external code was copied.
The focused storage tests passed 6 tests with 1 installed-Weaver skip on
the local RTX 2000 Ada GPU, including actual pinned CUDA transfers, FP32/BF16
training parity with varied physical vectors and unequal view lengths,
BN tampering, inference bypass and hook cleanup. The synthetic isolated pair
test reduced peak allocation from 103227392 to 66981376 bytes; that is not a
full-Weaver/A100 production measurement. Weaver is unavailable locally.

The affected fusion-chain, salience, learned-handoff/withdrawal and dzfix
continuation regression suite passed 118 tests with 3 installed-Weaver skips
and 2 existing PyTorch mask-type warnings (371.14s). The final strengthened
offload fixture was rerun separately: 6 passed, 1 Weaver skip (28.37s); counts
overlap and are not additive. CLI help, both queue/worker Bash syntax checks,
and scoped whitespace checks passed. An optional all-JetClass2 sweep was
stopped while running additional unrelated fixtures and is not counted as a
completed regression result.

Next step is commit/push of the reviewed repair, then the existing queue helper
from a new pinned checkout/root in debug. It authenticates and reuses the
completed matching screen and runs the four new gates before auto-submitting
the 21 science tasks. No matching work needs rerunning and no prior root or
acceptance should be patched in place. No commit, push, remote submission,
cancellation or artifact change was performed here. Recheck the remote state
before any optional exact-ID cleanup of the old blocked launcher 21757081.

## 2026-09-22: K2 tier3/debug pending-job portability implemented locally

The user's new execution requirement supersedes the earlier K2 debug-only
plan. New K2 launch/campaign/acceptance contracts are v2; initial submission
defaults to tier3 (`K2_PARTITION=debug` / CLI `--partition debug` remains
available). A hashed K2-only execution policy admits exactly SPORC tier3
and debug for manual pending-job partition-only changes. Shared execution
profiles and validators, matching, views, model/training recipe, seeds,
data splits and the source-screen dependency are unchanged.

Workers resolve the actual Slurm partition, authenticate account/QoS,
exact job, CPUs/RAM/time, environment and GPU, then record requested/actual
sites in hashed execution records bound into task/launcher receipts.
Preflight can run on either allowed partition; science may use the other
only with identical accepted GPU model/memory/compute capability/software.
All requests retain the <=24h common envelope and <=23h fit projection.
Original submission ledgers remain immutable; monitoring shows actual versus
requested partitions. Moving a launcher does not retarget descendants:
their initial requests still use the campaign's registered partition.

Changed surfaces: `concat_k2_{campaign,source,submit,runtime}.py`, new
`concat_k2_execution.py`, CLI, queue helper and Slurm worker, two updated K2
test files plus new `test_jetclass2_concat_k2_execution.py`, plan, contract
and donor map. Repository-local donor is the debug-only K2 commit
`1f9306504dfd040c9c22e0a89829d277d1ff2194`; no external source is copied.

Verification: broader K2/fusion-chain/salience/SPORC regressions passed
229 tests with 2 installed-Weaver skips (Weaver unavailable locally) and
2 existing PyTorch mask-type warnings. The expanded final execution-policy
suite passed 57 tests, and the final monitor/cross-partition acceptance
subset passed 3 tests. These runs overlap; counts are not additive.
Both shell files passed Bash syntax checks, CLI help exposes both
partitions, and scoped/new-file whitespace checks passed.

No commit, push, Slurm update, cancellation or live submission was performed.
The user-reported v1 jobs 21757208 (preflight pending Priority) and 21757209
(after_gate pending Dependency) still use old immutable debug-only source;
they are NOT made movable by this local patch. Current remote state must be
rechecked before any exact-ID cutover. Other jc2fc/jc2salp jobs are out of scope.
Next step: commit/push, create a new pinned checkout and v2 launch/root, then
run its staged preflight before science. This patch does not migrate/reuse
old K2 preparation or certify new real A100 execution. Do not edit old JSON,
ledgers or pinned checkouts to retrofit the policy.

## 2026-09-21: K2 concatenation ladder staged implementation complete

The isolated `JETCLASS2_DELPHES_CONCAT_K2_*/v1` campaign now implements the
user-selected D100 -> D075 -> D050 -> D025 -> D000 (HLT x3) -> HLT x1
compression ladder. Four registered comparisons are HLT x1 CE, HLT x3 CE,
pure offline CE and direct rich-D100-to-HLT-x3 KD: ten cold fits, five
teacher reducers and 17 science tasks including aggregate/complete.

Every new `jc2k2_*` job explicitly requests SPORC/debug, including both
launchers, per-file CPU assignment work, partitioning, acceptance and science.
The source adapter authenticates the existing debug matching screen's exact
completion ledger or durable completed artifacts. It imports only the
selected salience formula and population, not old maps or weights. New
capacity-two matching, support/crop diagnostics and full-population caches
are produced in a fresh root. New genuine expanded-input A100 acceptance
must pass before the science launcher is allowed to submit any fits.

The archived SPORC inventory has max HLT=277 and offline=306, yielding 832
padded tokens for this study; code rederives and authenticates that bound.
K=2 overflow keeps the highest frozen native salience offline particles,
never drops HLT or jets, and records per-role/class loss diagnostics.
Fillers are active HLT-owner copies. D000 and HLT x1 deployment adapters
cannot access offline data or matches. All 17 features are rebuilt from the
supplied set. The selected HLT-only fits publish deployment manifests.

Files: six new `jetclass2_delphes/concat_k2_{views,data,campaign,source,runtime,submit}.py`
modules; thin `scripts/jetclass2_concat_k2.py`,
`scripts/queue_jetclass2_concat_k2.sh`, `sbatch/run_jetclass2_concat_k2.sh`;
two `tests/test_jetclass2_concat_k2*.py` suites; new
`docs/contracts/JETCLASS2_DZFIX_CONCAT_K2.md` and updates to the active plan,
plan index and donor map. Repository-local donor baseline is
`2f0afe5c438cbab5d66e047c4d86534aab02032b`; adapted file hashes and deliberate
scientific changes are recorded in `docs/LEGACY_SOURCE_MAP.md`.

Verification: 181 focused tests passed, 2 installed-Weaver tests skipped
because Weaver is absent locally. Coverage includes bounded exhaustive
capacity-two comparisons, all rung invariants, overflow retention and frozen
salience, real synthetic ROOT reads/maps/caches, sealed test, CPU test-double
fit/reducer/compression and preflight dispatch, source/receipt corruption,
dependency closure and full canonical dry runs, idempotent submission,
ambiguous-ack failure, and report/deployment parity. Existing salience,
fusion-chain and K2 count-audit regressions passed. Two existing PyTorch
mask-type deprecation warnings remain in the fusion regression tests.
Bash syntax, new-file whitespace and changed-document Markdown links passed;
the broad unrelated temporary-directory Markdown walk was not run.

Evidence boundary: **no installed-Weaver/A100 acceptance, remote K2
preparation, science fit or Slurm submission was performed here**. This is
ready for the staged debug queue entrypoint, not a claim that batch 256 at
832 tokens already fits the GPU or 24-hour limit. The real gate fails closed
on memory/runtime/lineage failure; it cannot silently trim, change batch,
reuse old acceptance or skip poor scientific results. Existing campaigns,
raw data, completed audit artifacts and unrelated dirty work were preserved.

Exact next step: commit/push these changes, create a clean pinned SPORC
checkout, run `bash scripts/queue_jetclass2_concat_k2.sh` (dry), then the
same command with `--execute` to authorize the registered staged workflow.
The CLI also provides `monitor`, `audit`, `gate`, and `results --per-class`.
No manual full-DAG bypass is provided. Failed attempts retain their evidence
and require reviewed exact-ID, restart-zero handling in a fresh root.

## 2026-09-21: K=2 retained after the K=3 comparison

The user chose K=2 for the first fixed-slot concatenation experiment after
reviewing 0.473136% count overflow at K=2 versus 0.201912% at K=3 across
6,858,920 safe audited jets. The active
`docs/plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md` now records
acceptance of that tradeoff, three active tokens per HLT particle, unchanged
offline-only least-salience overflow cropping, and no initial K=3 campaign.
D100 versus pure offline on common validation rows is an empirical comparison,
not a gate; weaker results remain reportable and do not trigger automatic
changes or cancellation. Any revision is a separately registered follow-up.

Documentation-only update: plan, audit interpretation, and this handoff.
No donors, contract versions, runtime code, immutable artifacts, or jobs were
changed. Focused scaffold checks: 3 passed; changed-document links and
whitespace checked separately. Matcher/campaign implementation and genuine
expanded-input Weaver/SPORC resource acceptance are still outstanding.

## 2026-09-21: completed broad local dzfix K=2 count/pT audit

The user's local dzfix path is confirmed under
`C:/Users/22rya/ComputerScience/CERN/data/jetclass2_10M_20260918_dzfix/jetclass2`.
Its 331-file authenticated inventory and the existing partial-transfer plan
reproduce the active SPORC inventory hash. The read-only audit excludes the
UNION of final-test files in both recorded file-role registries (97 files),
then scans all 6,858,920 selected rows in the remaining 234 files with source
checksums before/after, exact class-count checks and latest-cycle/schema checks.

32,452 jets (0.473136%) satisfy N_offline > 2*N_HLT; 99.526864% fit without
cropping. SPORC ordinary overlap is 3,593,995 jets at 0.492210% overflow.
The tail is class dependent: QCD 0.1234%, X_mm 11.8598%, X_tauhtaum 6.5912%.
206,840 offline particles (0.073576% of the population's offline tokens)
exceed capacity; the median overflowing jet loses three tokens. This is a
larger safe-reservoir census, NOT the exact 500k/1M registered row subset.
See `docs/JETCLASS2_DZFIX_K2_CAPACITY_AUDIT.md` for methods and all class/tail
tables; `artifacts/jetclass2_dzfix_k2_capacity_audit_v1/` holds immutable
scope, joint histogram and report. Final test remains sealed.

A second pass rechecked stored counts against actual jagged lengths for every
audited jet and measured minimum raw-scalar-pT loss on all 32,452 overflow jets.
Median per-overflow loss is 0.4490%; 2,767 jets (0.040342% of the entire census)
must lose more than 10% of offline scalar pT. This is a count-constrained
lower bound, not the final salience policy or a classifier-information bound.
Its immutable report is `artifacts/jetclass2_dzfix_k2_pt_audit_v1/report.json`.

New reusable diagnostic code is `jetclass2_delphes/capacity_audit.py` and
`capacity_pt_audit.py`, with thin `scripts/audit_jetclass2_k2_*.py` CLIs and
new diagnostic-only v1 schemas.
No donor file was copied and no old contract, matching foundation, training
worker, campaign or Slurm job was changed. The K=2 ladder plan now records
this evidence and the confirmed local path; matcher/training implementation
and real expanded-input Weaver/SPORC resource acceptance still remain.

Focused local verification: 19 tests passed (new audit, partial-snapshot,
selection-policy, and scaffold checks); changed-document links and whitespace
checked separately. No Weaver/GPU training gate was run or claimed.

## 2026-09-21: fixed-slot K=2 concatenation ladder design recorded

Added `docs/plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md` and its
plan-index link. This separate dzfix 500k/1M/1M design uses one original HLT
slot plus two rich destinations per HLT particle, global salience-aware
capacity-two assignment, explicit least-salience rich-only overflow cropping,
and HLT fillers. Support stays exactly 3*N_HLT; rich contents progress to an
offline-free HLT x3 endpoint, followed by logit KD into ordinary HLT x1.
It is a single concatenated-set encoder, not the existing two-encoder fusion.

The user subsequently fixed the ladder to D100 -> D075 -> D050 -> D025 ->
D000 -> HLT x1 compression. D100 explicitly means native offline + native
HLT + fillers (with rich-only overflow cropping), superseding the draft's
hybrid persistent-U000 rich pool. Distinct CONCAT_K2 artifact names prevent
confusion with the historical single-copy D000. The quick audit measures
whether N_offline > 2*N_HLT is rare; <= is the no-cropping case. At the
design-only stage no dataset audit had been run; the subsequent local
count/pT study and confirmed Windows path are recorded above.
The document distinguishes these agreed semantics from proposed salience
selection, training/control registration, and unmeasured resources.
Next is a read-only train/validation count and salience-loss audit, not a live
launch. Existing one-to-one foundations and their no-truncation policies are
unchanged; new matching/view contracts and genuine expanded-input SPORC
acceptance will be required. This is documentation only: no runtime changes,
donor migration, contract publication, remote audit, or Slurm submission.
Documentation verification: 124 local links across the new plan, plan index,
and handoff resolve; scoped whitespace and code-fence checks pass. Focused
existing scaffold/salience checks passed 16 tests (two unrelated tests
deselected) in the tagging-hlt environment. These tests do not validate an
unimplemented capacity-two matcher or establish real-Weaver/SPORC readiness.

## 2026-09-21: new dz-fix fusion-to-fusion coarse chain on SPORC/debug

Implemented the isolated campaign in
`docs/plans/JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md` and
`docs/contracts/JETCLASS2_DZFIX_FUSION_CHAIN.md`. This ports the scientific
CMS fusion-chain graph, not its checkpoints, to the authenticated September-18
dzfix TRAIN_500K population: 500k train, 1M validation, 1M sealed final test.
The selected persistent-HLT salience foundation is imported read-only from the
active v2 debug screen; all 12 fits are fresh, including U000 and the first
Fusion(U000,U050). No CMS ACQUIRE_U050, metric or GPU acceptance is reused.

The five adjacent paired teachers progress through U050/U100/D066/D033/D000.
Both endings are registered: direct single D000, and D000/D000 fusion followed
by single D000. All KD is cold C25P75/T2 with alpha=1; no withdrawal, intermediate
compression, output ensembling or extra seed panel. Fresh M0HLT, pure OFFLINE,
persistent U000 and ordinary direct-KD references support report-subset recovery.
Validation is class/identity-stratified 50/25/25 checkpoint/diagnostic/report;
matching already used this validation reservoir, so report is not unseen test.

New code: `jetclass2_delphes/dzfix_fusion_{chain,source,data,model,runtime,submit}.py`,
`scripts/jetclass2_dzfix_fusion_chain.py`,
`scripts/queue_jetclass2_dzfix_fusion_chain.sh`,
`sbatch/run_jetclass2_dzfix_fusion_chain.sh`, and
`tests/test_jetclass2_dzfix_fusion_chain.py`. A default-no-op bias hook in the
existing `salience_learned_model.py` supports the new adapter's compact shared
padding mask; historical behavior is unchanged. Full Weaver pair-BN population
and gradients are retained. Donor paths/commits are in `LEGACY_SOURCE_MAP.md`.
Artifacts use `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*`; LAUNCH_SPEC,
SOURCE_IMPORT and CAMPAIGN_SPEC are v3, other artifacts retain v1.

The user's SPORC launch at `e3b02d6f6f0f3b6433230603a2438767b69e25fd`
exposed a stale fixed-capacity check: all three salience foundations and the
bottleneck control have capacity 320, not 240. Inventory
`10d41d10cf509e11db432c80ecd844bfdac2d5392ffee93a370e6638d9e57435`,
TRAIN_500K counts, split membership and foundation hashes all matched. That
attempt failed before publishing a launch root or submitting any jobs.
v3 registers `inventory_max_selected_round_up_16_no_truncation_v1`, verifies
the complete canonical input contract at the inventory-derived capacity, and
reports the specific mismatched field/path. It never changes existing matching
or truncates particles. Native/paired cache budgets and GPU longest-batch stress
already consume the selected foundation capacity; their limits are unchanged.
The matching handoff's copied 240 statement is corrected to 320. Producer
screen schema v2, job 21748725, debug routing, ladder, seeds and loss are unchanged.

Capacity-fix evidence: baseline 36 passed/1 Weaver skip; replacing the source
fixture with inventory-derived 320 reproduced the reported launch failure.
After the fix, fusion-chain/salience/continuation/learned regressions passed
**97 tests with 2 installed-Weaver skips** in 116.72 s. Coverage includes all
four foundation contracts, metadata rounding, tampered capacities/schema,
>240-particle native paths without truncation, capacity-driven RAM bounds,
320-column single/paired stress padding, and rejection of v1/v2 launch specs.
CLI help, shell syntax and scoped whitespace checks pass. No real new A100
preflight, commit, push or submission was performed here. Next: push these
scoped changes and repeat the queue procedure at the new pinned commit; the
real GPU/RAM/runtime gate must still pass at capacity 320 before science.

All new `jc2fc_` jobs explicitly use debug, not tier3: account reu-aisocial,
qos_tier3, one A100 for GPU tasks, 8 CPUs/320000 MiB and <=24-hour walltime.
The RAM request is conservative for the new 1M paired validation population.
The initial launcher now binds the active screen's exact eight-task ledger
and `complete` job (21748725), not the canceled continuation job 21741416.
The source is `jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json`, from
commit `0d25a4a53aafb1348c8279d86bac7dbac82c8841`. Foundation locations are
read from that spec/selection, not guessed from the old continuation root.
Completed durable evidence avoids stale Slurm IDs. No old continuation receipt
or production dry run is required. The consumer materializes its own full dry
run and four new gates, then submits a second
afterok launcher for the 21-task science DAG. Existing campaigns are neither
modified nor cancelled. Exact submission intents/receipts prevent blind retries
after an ambiguous sbatch acknowledgement. The queue helper is dry by default
and accepts `SCREEN_SPEC` / `--screen-spec`, not `CONT_SPEC`. Old v1 launch or
campaign roots are rejected; v2 roots likewise cannot bypass the v3 capacity
policy. Use fresh roots at the new pushed source. The
producer's separately registered tier3 production is not moved or inherited.

The fresh gate measures full paired cache construction, native single/fusion/
same-view/compression miniature kernels, real longest-jet batch-256 memory,
checkpoint and T2 bank round trips, installed-Weaver FP32/BF16 mask parity,
90% CUDA/80% CPU limits, and a conservative maximum-fit runtime that fits debug.
No test data is read by ordinary input workers. No local result substitutes for
this real A100 evidence.

Current direct-screen migration evidence: **67 passed, 2 skipped** across
fusion-chain, salience screen, historical continuation and learned-handoff tests.
Skips require installed Weaver, absent locally. New tests cover the exact v2
screen ledger/dependency closure, pending/completed boundaries, old schema and
wrong-parent rejection, foundations outside the screen root, corrupt receipts,
producer tier3/consumer debug separation, and the CLI's printed dependency.
The initial implementation's earlier full JetClass2 run was **240 passed,
3 skipped**; that entire suite was not rerun for this targeted migration.
CLI help, both shell syntax checks and tracked diff whitespace checks pass. No Slurm
jobs were submitted from this workspace. Next: commit/push these scoped changes,
use a clean pinned SPORC checkout, then invoke the queue helper. Scientific jobs
remain fail-closed until matching completion and the new real debug gates pass.
The prior chat commands naming CONT_SPEC or asserting parent 21741416 are
obsolete; use the updated helper and a fresh v3 launch at the new commit.

## 2026-09-20: isolated CMS fusion-to-fusion coarse KD chain

Implemented the user-requested chain under
`docs/plans/CMS_FUSION_CHAIN_500K_PLAN.md`: reuse completed ACQUIRE_U050
(U000 context/U050 primary), cold KD into U050/U100, U100/D066, D066/D033,
then D033/D000. At that last paired teacher, register both endings: direct
single-D000 KD, and D000/D000 learned fusion followed by single-D000 KD.
All new KD is ordinary C25P75/T2; alpha stays one, no withdrawal or output
ensemble. Both final single-ParT models share initialization/sampler seeds.
The bridge costs one extra fit and is not a compute-matched arm. Same-view
fusion has independent branch parameters and identical genuine HLT inputs.

New CAMPAIGN_SPEC/v7, GRAPH/v4, `ladder=fusion_chain` and `cmsfc_` debug jobs
keep this study separate. The 21-task science DAG consists of seven CPU
imports, seven fresh fits, five fresh reducers and aggregate/completion.
Preparation and accepted dense references are reused read-only. Only the
completed first acquisition and its bank come from coarse-v5; no running
withdrawal or downstream coarse result is a dependency. Import validation
binds exact source specs/commits, canonical ledgers, receipts, checkpoint/
teacher identities and payload hashes. Old graphs and scientific kernels
are unchanged. Source roots are never overwritten or cancelled.

PREPARATION_IMPORT/v4, SHARED_SOURCE/v3, ACCEPTANCE_IMPORT/v3 and
ACCEPTANCE_REUSE/v3 distinguish the new consumer. ACQUISITION_SOURCE/v1 and
FUSION_CHAIN_AGGREGATE/v1 bind its additional import and report semantics.
Genuine accepted dense preflight 21720795 remains the resource evidence;
strict native code, data, environment assumption, resources and 85% CPU /
90% CUDA checks remain. This is explicit evidence reuse, not a new GPU run.
The new `fusion_pair_kd` node role uses the unchanged ordinary paired kernel
without the historical coordinate-keyed context-permutation diagnostic.

Added `create-fusion-chain`, results/status support and
`scripts/queue_cms_fusion_chain.sh`. `prepare` requires installed Weaver and
checks the same-view CPU route before importing/verifying evidence and
materializing the dry plan; it queues nothing and performs no full training.
`submit` checks the gate and submits the exact journalled science DAG with
its own authorization phrase. Partial submission resumes recorded jobs;
there is no cancellation or scheduler-update path. Results list all twelve
logical models on validation REPORT rows with offline recovery, per-class
QCD R50 and both final comparisons; privileged/H2/H1 labels distinguish
input and architecture costs. Final test remains sealed.

Existing native/coarse/reuse/temporary-memory/direct Python 3.10 regression:
119 passed, five installed-Weaver skips in 224.90 s. Final new-chain suite:
16 passed, one installed-Weaver skip in 216.35 s. This includes the complete
synthetic production chain with both endings, matching single-view initial
states, same-view independent-branch gradients, source-byte preservation,
invalid source command/miniature-fit rejection, final-test sealing and exact
partial-submission recovery. Genuine Git donor/runtime comparisons pass;
helper shell syntax, CLI parsing and scoped whitespace checks pass. These
local tests do not constitute new SPORC acceptance. No commit/push, scheduler
submission, cancellation or remote artifact mutation was performed.

Python 3.13 cross-version chain/reuse/direct regression additionally passed
45 tests with one installed-Weaver skip in 269.11 s, including the reviewed
Python-version-dependent AST identities. The SPORC `prepare` helper must run
the installed-Weaver CPU check in the existing accepted environment; no local
fake-Weaver test is represented as that check or as a new GPU measurement.

Next: scoped commit/push, fresh detached SPORC checkout, then helper
`prepare` and `submit` against the completed coarse-v5 source. The working
tree contains unrelated matching/scouting and documentation edits; do not
stage all changes. Kernel reuse deliberately rejects a scientific code
change bundled into this orchestration-only commit.

## 2026-09-19: standalone CMS direct Strategy B comparison

User requested a direct fusion/withdrawal comparison alongside the running
coarse ladder, not a replacement. New scientific authority is
`docs/plans/CMS_SALIENCE_DIRECT_FUSION_500K_PLAN.md`. CAMPAIGN_SPEC/v6,
GRAPH/v3 and `ladder=direct_fusion` register one U000 -> D000 transition:
ACQUIRE_D000 (D000 primary, U000 context, U000 KD), frozen acquisition bank,
WITHDRAW_D000, then exactly extracted HLT-only CARRIER_D000. Ordinary
DIRECT_D000 remains the imported single-view KD comparator. Both phases keep
the existing scientific kernels, losses, seed alias, batch and LR schedule;
two fits versus one is explicitly not a compute-matched comparison.

The eleven-task science DAG imports the five shared reference/control tasks
from the original accepted dense debug source and adds only two fresh fits,
one reducer, extraction, aggregate and completion. Dense/coarse graph hashes
are unchanged. All jobs are pinned to debug with the separate `cmsdf_` prefix;
imports are labelled `import_`. No coarse jobs are dependencies. The direct
spec is rejected by `retire-dense` before any scheduler operation.

PREPARATION_IMPORT/v3, SHARED_SOURCE/v2, ACCEPTANCE_IMPORT/v2 and
ACCEPTANCE_REUSE/v2 bind the direct consumer to read-only dense evidence,
including the original accepted longest-U000/U000 batch-256 envelope from
job 21720795. The strict code/data/resources/partition and 85% CPU / 90% CUDA
checks remain. No duplicate GPU gate is required when compatibility verifies;
no new GPU measurement is claimed. The existing installed conda environment
must be unchanged. No production/model/training/data/worker kernel was edited.

Added `create-direct-fusion` to the CLI and a dedicated helper
`scripts/queue_cms_direct_fusion.sh`: `prepare` creates/imports/verifies and
dry-materializes without submission; `submit` checks the imported gate and
submits the exact new science DAG; `results` prints recovery and direct-KD
deltas. The helper contains no cancellation or scheduler-update path.

Focused new Python 3.10 tests: 15 passed, including the complete tiny
production-path chain using fake Weaver, source-file preservation, exact
extraction, source-completion dependency resolution and idempotent submission.
Full native/coarse/reuse/temporary-memory/direct regression on Python 3.10.19:
119 passed, 5 installed-Weaver-dependent skips in 294.11 s. Python 3.13.12
coarse/reuse/direct regression: 52 passed in 162.12 s. The pre-change
coarse/reuse baseline was 37 passed. Both interpreters verify the unchanged
real-Git kernel/probe fingerprints. Helper shell syntax, CLI parsing and
scoped whitespace checks pass. Local tiny evidence is not remote acceptance.
No source commit/push, remote submission, cancellation or data mutation was
performed. Next: scoped commit/push, fresh detached checkout on SPORC, then
`prepare` and `submit` against the original dense debug source. Shared HANDOFF
and donor-map files also contain pre-existing unrelated unstaged updates;
do not stage those entire files blindly with this implementation.

## 2026-09-19: CMS coarse reuse Python-version fingerprint correction

User's SPORC creation at `4f862c11045943f1237d1cef166d32a85c341ed3`
failed before cancellation/submission with `Preparation code changed`.
Git comparison proves the preparation files and accepted runtime unchanged.
The exact coordinate/preflight AST allowlists had been recorded with local
Python 3.13, whose default dump omits empty lists; SPORC Python 3.10 includes
them. Reproduced both the preparation and subsequent preflight mismatch with
local Python 3.10.19: two focused failures before the fix.

Added only exact reviewed 3.10 old/new hash pairs alongside the original 3.13
pairs in `preparation_import.py` and `preflight_reuse.py`; existing serialized
fingerprints and immutable import descriptors remain unchanged. Tests exercise
both allowlist pairs, reject mixed/unknown/changed-file proofs, and compare
real Git producer f2e8a374, accepted donor 7bb17138, coarse 48ab8609 and failing
consumer 4f862c11. Native model/training/data, graph, thresholds and schema
versions are unchanged. Python 3.10.19 native/coarse/reuse/temporary-memory
regression: 104 passed, 5 installed-Weaver-dependent skips in 218.81 s.
Python 3.13.12 coarse/reuse regression: 37 passed in 102.78 s. Actual Git
preparation and accepted-runtime comparisons pass under both interpreters.
Shell syntax and scoped whitespace checks pass. No new SPORC/GPU measurement
was performed; no remote job or artifact was modified, and no commit/push was
performed by the agent. Next: scoped push and retry the source-pinned
`reuse-and-switch` command against a fresh new-commit coarse root.

## 2026-09-19: user-authorized reuse of accepted dense preflight for coarse

The user requested skipping the duplicate coarse GPU preflight and confirmed
none had been submitted. This amends the active coarse plan: opt-in campaign
v5 can reuse genuine compatible dense-v3 acceptance instead of running a new
GPU gate. It does not change existing v4 campaigns or weaken the memory policy.
The historical instruction below requiring a fresh coarse gate remains the
v4/default path, not a restriction on this explicitly authorized v5 path.

Added native `preflight_reuse.py`, ACCEPTANCE_IMPORT/v1 and ACCEPTANCE_REUSE/v1,
with original receipt/report/source/job identity and runtime/probe-code checks.
The previously measured U000/U000 longest-batch envelope applies to the same
model/data/resources; only the exact reviewed preflight schema-predicate change
is normalized. The recorded measurements remain those of the dense donor,
explicitly `fresh_gpu_measurement=False`. No new EXECUTION_ACCEPTANCE is forged.
Software in the named SPORC conda environment is assumed unchanged.

`switch_cms_salience_coarse.sh reuse-and-switch DENSE_SPEC NEW_ROOT` performs
CPU evidence imports, gate verification, exact dense-only retirement and
31-task coarse submission. It leaves M0HLT/OFFLINE/U000/reduce_U000/DIRECT_D000
jobs and all original artifacts intact. Default fresh-preflight behavior is
unchanged. Internal baseline is coarse commit
`48ab8609ee87ba72ab9868dfc951a36a1c9d851e`; accepted runtime donor remains
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` / SPORC job 21720795.

Focused baseline: 14 passed. New reuse tests: 12 passed in 51.18 s, including
real Git-code equivalence, CPU-only import, honest source-labelled evidence,
missing/corrupt/mismatched proof rejection, unchanged legacy gates and explicit
science authorization. Combined native/coarse/reuse/temporary-memory regression:
100 passed, 5 installed-Weaver-dependent skips in 211.30 s. Shell syntax,
CLI option discovery and scoped whitespace checks pass. No new GPU run was
performed; genuine acceptance remains the named source job. No remote job,
commit or push has been changed by this implementation turn.

## 2026-09-19: native CMS coarse replacement with shared-job preservation

User authorized replacing the dense ladder with
`U000 -> U050 -> U100 -> D066 -> D033 -> D000`. Active authority is
`docs/plans/CMS_SALIENCE_LEARNED_COARSE_500K_PLAN.md`. Native campaign v4 / graph
v2 register five transitions (14 logical fits, 31 science tasks). Dense v1-v3
graphs remain byte-identical. Batch, losses, matching, data budgets, validation
roles, seeds for common nodes and the 85% CPU / 90% GPU gate are unchanged.

The replacement can import the completed original preparation and the five
common tasks from accepted dense source `7bb171382b7206013bc5d9308a4c22b2929bc7f4`.
Those are M0HLT, OFFLINE, U000, its reducer, and DIRECT_D000. Pending/running
source jobs are preserved with exact dependency edges; authenticated completed
outputs do not depend on aged-out Slurm IDs. There are ten fresh transition
fits. New PREPARATION_IMPORT/v2 permits only the reviewed coordinate-AST
addition, and SHARED_SOURCE/v1 binds source spec, live ledger, receipts and
scientific-code identities. No dense carrier or GPU acceptance is relabelled.

Added `shared_import.py`, `coarse_submission.py`, coarse regression tests and
`scripts/switch_cms_salience_coarse.sh`. Extended the native registry, campaign,
dispatch, preparation adapter and CLI; training/model/cache kernels unchanged.
`prepare` creates the isolated coarse root, verifies/imports preparation and
submits its own preflight, without cancelling anything. `preview` is read-only.
After that gate passes, `finish` validates shared-parent health, retires only
the 41 dense-specific ledger tasks and submits the 31-task coarse DAG with
crash-safe exact-command journaling. Original artifacts and unrelated jobs
are untouched. Scientific output copies go only into the new root.

User-supplied accepted dense preflight 21720795 measured 35.52 GiB CUDA peak
on 39.52 GiB capacity (89.88%) and 30.85 GiB CPU RSS, with exact extraction and
sealed final test. This supports feasibility but does not authorize the new
source. The new genuine installed-Weaver/SPORC gate remains required.
Focused final native/coarse/temporary-memory regression: **88 passed, 5 skipped
in 158.19 s**. Skips require installed Weaver; the operator helper reruns these
tests in SPORC's environment before creating the replacement. Coverage includes
all five miniature acquisition/withdrawal/extraction transitions, reference and
bank imports, sealed test, immutable source bytes, exact cancellation scope,
source-completion races and interrupted submission replay. Bash syntax and
scoped diff checks pass. Real Git confirms original/accepted preparation code
identity and unchanged common scientific-worker/seed/schedule ASTs. No jobs have
been changed remotely by this implementation session; no commit or push has
been performed. Next: scoped push, pinned `prepare`, fresh SPORC gate, then
explicit `finish`. Unrelated dirty files remain preserved and must not be
included wholesale in this migration's commit.

## 2026-09-18: native CMS-LFH temporary cross-attention memory reduction

User supplied SPORC debug preflight `21720511`, source `77c9d2b1`:
peak CUDA `39278852096` / capacity `42430300160` bytes (92.57%), after
the repeated withdrawal probes and extraction. The run rejected the strict
90% GPU headroom limit, not a reported CUDA OOM. It needs over 1.0166 GiB
less peak allocation to pass that limit; the failed gate remains immutable.

The native CMS adapter now merges context padding into one compact rectangular
cross-attention bias and shares it across all four residual injections. This
removes repeated mask additions and lets the unused full-square output storage
be released once the rectangular view is replaced. Full Weaver pair embedding
and its BatchNorm population stay unchanged. Both withdrawal routes, gradients,
batch 256, model/state keys, loss weights, RNG call order and schedule are
preserved; there is no activation recomputation, microbatching, or threshold
increase. Non-CMS fusion adapters keep the legacy path through a new base hook.
Actual peak and runtime improvement are not yet measured on A100.

Changed native `model.py`, shared
`models/hcwdl_offline_hlt_fusion_transformer.py`, added
`tests/test_cms_fusion_temporary_memory.py`, and updated the active plan,
contract explanation and donor map. Internal donor is
`77c9d2b1fe17a9fb321f1f85ebf9cd803c0f5ec1`; no external code migration or
schema/version change. Preparation/training/scientific registry code was not
edited; authenticated original preparation remains reusable read-only.

Baseline native suite: **61 passed, 1 skipped**. New allocation/parity tests:
**6 passed, 4 skipped**. Combined native/new/shared fusion regression:
**122 passed, 5 skipped, 1 deselected in 66.54 s**. Tests prove one shared
compact mask versus four legacy masks, square-backing release, gradient flow,
loss/parameter-gradient/AdamW/buffer/RNG parity over repeated updates, unequal
padded views and exact zero extraction. Floating gradient reduction order may
differ within tolerances; whole-training bitwise equality is not claimed.
Five installed-Weaver tests skip locally. One older shared test requires
Weaver without a skip guard: it failed in the pre-change baseline for missing
Weaver and was deselected in the combined run. No local GPU evidence exists.

Next: scoped commit/push, installed-Weaver parity checks (including the new
test file), and a fresh pinned SPORC debug preflight using original completed
`cms_salience_learned_dense_500k_f2e8a374_r1` preparation. Keep 85% CPU / 90%
GPU limits and the 30 full-batch probe updates. Do not submit science unless
the new real gate passes. No remote job, artifact, commit or push was changed
by this implementation turn.

## 2026-09-18: native CMS-LFH 90% CUDA gate with repeated full-batch probes

The user authorized trying a 90% GPU acceptance limit while keeping CPU at
85%, batch 256, and the production model/loss/schedule unchanged. SPORC debug
job `21719837` on source `9af094b0` completed the four miniature routes and
exact extraction, then rejected CUDA peak `37448891392` of `42430300160`
bytes (88.26%). It did not report a CUDA OOM. This supersedes the earlier
unknown-peak status below; the failed campaign remains failed and immutable.

New native `CAMPAIGN_SPEC/v3` freezes this policy. New
`EXECUTION_ACCEPTANCE/v2` requires five consecutive optimizer updates per
alpha=1/0.5/0 in each paired route: 30 recorded probe updates, explicitly
using the 256 longest U000 training jets. The optimizer persists throughout
each route's probe, with no between-step cache clearing or peak reset.
Science revalidates every step's batch size and high-water measurements,
policy/version identity and strict 85% CPU / 90% CUDA limits. Old campaign
v1/v2 still requires its 85%/85% acceptance v1. No batch reduction, gradient
accumulation, activation checkpointing or new scientific fit is introduced.

Changed native contracts/campaign/production, focused tests, the active plan
and reusable contract; donor lineage is recorded in `LEGACY_SOURCE_MAP.md`.
Baseline: **42 passed, 1 skipped in 51.71 s**. Updated focused suite:
**61 passed, 1 skipped in 63.31 s**. Tests cover full 256-row consecutive
updates, longest-jet selection, unchanged optimizer between steps, strict
memory boundaries, missing/tampered probe evidence, legacy gates and
read-only v1/v2 preparation import. An additional
policy-copy/legacy-gate/import regression run passed all
**7 selected tests in 9.92 s** after isolating policy objects from the registry.
Local GPU/Weaver evidence is mocked; installed-Weaver parity remains skipped
because Weaver is not installed.
Science graph, seed, coordinate and schedule ASTs are unchanged from
`9af094b08cdc112a8a5048374fee4f135b704087`.

No remote job or artifact was changed, and no commit/push was performed.
Next: push only these scoped changes, then create a fresh source-pinned
SPORC debug root and rerun preflight, importing original completed
`cms_salience_learned_dense_500k_f2e8a374_r1` preparation read-only. Do not
reuse the failed gate or submit science until the new genuine gate passes.
The prior peak would fit under 90%; repeated real-A100 measurements, not
local tests, determine whether the new probe actually stays below it.

## 2026-09-18: native CMS-LFH preflight memory cleanup and diagnostics

User-supplied SPORC debug job `21719606`, source `85fd0214`, completed all
four miniature fit routes and extraction checks, then failed the final 85%
headroom gate. Slurm reported `66938876K` CPU MaxRSS against a 192000 MiB
request; the old exception omitted its actual CPU/CUDA measurements, so the
failed dimension and GPU peak cannot be established from that log alone.
The earlier paired-cache extraction repair did work on this genuine run;
this is not a successful execution acceptance or permission to run science.

Refactored native `production.py` so each miniature's model, optimizer and
loss references expire before the next route. A new weak-reference assertion
failed against the old loop at `fusion_withdrawal`, proving acquisition
parameters were still retained. Added flushed CPU/GPU memory readings after
cache construction, fits, alpha-regime probes, extractions, cleanup and at
the final check. A refusal now identifies CPU RAM, CUDA, or both, with peak
bytes and capacities. CUDA high-water statistics reset once only, and the
unchanged strict 85% gate uses that retained peak rather than current usage.
No batch-size, model, objective, matching, schedule or scientific schema
change; no external donor migration. See `LEGACY_SOURCE_MAP.md` for lineage.

Local baseline: **39 passed, 1 skipped in 31.87 s**. After repair:
**42 passed, 1 skipped in 45.25 s**. Tiny native ROOT preflight tests use
mocked Weaver/GPU evidence, verify released parameters between routes,
preserve byte-exact extraction, and inject CPU-only/GPU-only/combined
headroom failures. All failures publish no execution acceptance and remain
blocked by the science gate. Installed Weaver is unavailable locally.

No jobs, remote artifacts, or existing campaign roots were modified. The
actual A100 peak after cleanup is still unknown; this is not a claim that
the next gate must pass. Next: scoped commit/push, then a fresh pinned debug
preflight reusing the original `cms_salience_learned_dense_500k_f2e8a374_r1`
preparation. Keep the existing memory request and safety margin until the
new telemetry establishes whether further memory optimization is needed.

## 2026-09-18: native CMS-LFH paired-cache extraction repair

SPORC debug preflight `21719164` at source `08c36479` completed full-population
cache construction and the first three miniature fit routes, then failed with
`KeyError: 'features'` while evaluating an extracted ordinary primary model on
the retained paired validation cache. This was a batch-interface bug, not a
matching failure or evidence of insufficient memory.

Native CMS `training.predict` now chooses primary-only batches for ordinary
single-view models as well as alpha-zero fusion. Privileged fusion still
requests both views. The extraction equality requirement remains byte-exact;
matching, models, losses, schedules, populations and scientific contracts are
unchanged. No schema version bump or external donor migration is involved;
the internal baseline is recorded in `LEGACY_SOURCE_MAP.md`.

Focused baseline: **30 passed, 1 skipped**. Before the fix, six new ordinary
inference cases and a tiny-data full-preflight regression failed; the latter
reproduced the production `KeyError` at the same extraction call. After the
fix: **39 passed, 1 skipped in 39.79 s**. Tests cover both probability
temperatures, context non-access, privileged paired inference, and both
acquisition/withdrawal extraction checks. The full-preflight test uses real
native orchestration and tiny ROOT inputs with mocked Weaver/GPU evidence;
it is not remote acceptance. The installed-Weaver test was extended but is
skipped locally because Weaver is unavailable.

No remote jobs or artifacts were modified. Next: push the scoped fix, create
a fresh pinned debug campaign/root importing preparation from the original
`cms_salience_learned_dense_500k_f2e8a374_r1` producer (not the failed imported
debug campaign), and run a fresh genuine SPORC preflight. Existing immutable
preparation can be authenticated read-only; RAM caches are rebuilt. Do not
requeue the old source-pinned job or submit science before the new gate passes.

## 2026-09-18: native CMS-LFH selectable SPORC partition

The native CMS/Scouting dense learned-handoff CLI now creates execution
`CAMPAIGN_SPEC/v2` with explicit `--partition tier3|debug` (tier3 remains the
default). Submission, preflight measurement, GPU allocation authentication and
science-gate validation agree on the selected partition. All other scheduler,
environment, resource and scientific checks remain. Debug is an explicit
operator route subject to RC policy, not an inferred production permission.
Legacy v1 specs remain readable and tier3-only; no existing queued job changes.

Added `--reuse-preparation-spec` and `PREPARATION_IMPORT/v1` for a fresh root
to authenticate completed native selection, assignment, calibration, coupling
and validation-partition artifacts read-only. The import stage checks exact
preparation-code Git identities, scientific/population equivalence, canonical
paths, receipts and payload hashes. It does not copy dense caches, re-run
matching, import a trained model, or reuse the source GPU gate. A new genuine
A100 preflight is mandatory before the unchanged 46 science tasks. Current
worker counts come from the consumer's allocation, not the producer's.

Files: native CMS `campaign.py`, `contracts.py`, `production.py`, new
`preparation_import.py`, thin CLI, focused tests, active plan and reusable
contract. Provenance is recorded in `LEGACY_SOURCE_MAP.md`. Focused local test
result: **30 passed, 1 skipped** (installed Weaver unavailable); shared SPORC
allocation/debug-profile regression suites: **27 passed**. CLI create help
exposes both new options. Synthetic ROOT tests verify imported views and
validation partitions are byte-identical and source files remain untouched.
This is local evidence only: no SPORC job was submitted, canceled or modified,
and no genuine debug GPU acceptance has run for this change. Next: commit only
these scoped changes, push, use a pinned fresh worktree/root, import the old
preparation, run that root's debug gate, then explicitly submit science.

## 2026-09-18: dz-fix partial-snapshot migration staged locally

The active [dz-fix migration plan](plans/JETCLASS2_DELPHES_DZFIX_500K_MIGRATION_PLAN.md)
targets the currently available files under Luka's September 18 producer path.
The new dataset combines the prior all-reco PUPPI-neighbourhood change with the
HLT `dz` primary-vertex-reference fix; it is still a PUPPI dataset. Production
closure is not required because the downloaded bytes will be frozen explicitly.

Added a scalar-metadata-only partial snapshot planner and CLI. It uses a
parent-inventory-bound, source-proportional deterministic file order and the
real 60/20/20 whole-file splitter. It requires all four registered training
capacities through 2M plus 1M validation and 1M sealed test with 5% headroom.
This corrects the tempting but invalid assumption that a raw 2.5M-row snapshot
could support the 500k/1M/1M campaign. Destination verification binds exact
paths, inventory/split hashes, ROOT bytes, cycles, and schema before reuse.

The transfer layer now also has a deterministic normalized USTAR builder and a
safe extractor. It hashes selected sources, archive bytes and each extracted
file; exact member coverage/order and metadata are enforced, paths cannot
escape, existing destinations are refused, and success receives an immutable
receipt. Extraction does not replace the subsequent ROOT re-inventory gate.

The final queue boundary is implemented but inert. Its transition planner
authenticates the old campaign/live ledger and the new completed-screen campaign,
canonical dry ledger and command plan. It requires a new inventory/root with the
same scientific graph and role counts, queries all exact old IDs, rejects unknown
states, and only prints active IDs. It cannot call `scancel` or submit. A second
mode records that every old-ledger job is terminal after explicit cancellation.

Focused baseline before edits: 26 existing JetClass2 inventory/split tests
passed. After the planner, transfer and exact-ledger transition implementation,
49 inventory/split/salience regression tests passed in 152.44 s.
Transfer-focused tests cover deterministic archive bytes,
changed source/archive rejection and extra-member refusal. Together these cover
deterministic capacity selection, insufficient-source refusal, plan tampering,
and destination mismatch. No dataset was copied,
no remote command or Slurm mutation was made, and no old job was cancelled.
Next: scoped commit/push, download the current EOS byte snapshot, audit and plan
locally, transfer only the selected archive, then re-inventory/verify on SPORC.

## 2026-09-17: CMS2JC2 response routes every stage to debug

Per the user's request, new preparation, acceptance, science, confirmation,
transfer/reporting, and recovery plans use SPORC `debug`. Account `reu-aisocial`
and QoS `qos_tier3` remain unchanged, as do CPU-only execution, resource requests,
scientific populations, and authorization/acceptance gates. Updated response
`campaign.py`, `preparation.py`, `submission.py`, `measurement.py`, `tasks.py`,
their focused tests, the active implementation plan, and the response contract.
Submission availability checks, worker allocation checks, and exact-job receipt
reconciliation now agree with the requested partition. Old immutable tier3
specifications are rejected rather than silently rerouted; recreate specs/dry
plans using a clean pushed snapshot of this updated source.

Validation: baseline **53 passed in 16.19 s**; after this routing change,
**58 passed in 30.28 s** across `test_cms2jc2_response.py`,
`test_cms2jc2_response_campaign.py`, `test_cms2jc2_response_lifecycle.py`, and
`test_cms2jc2_response_operations.py`. Coverage includes all 15 acceptance,
71 science and nine confirmation task requests, preparation, same-source
recovery, mocked worker partition validation, and stale-site rejection.
These mocks are not remote acceptance evidence. No live jobs were submitted,
canceled, or modified; no SPORC availability/resource certification was made.
No donor migration or scientific schema change; existing donor lineage remains
unchanged. Next: push the updated source, create/review fresh debug plans, and
perform the required real CPU acceptance before authorizing full science.

## 2026-09-17: CMS2JC2 response staged implementation complete locally

This checkpoint supersedes the partial-implementation boundaries in the older
entries below. The user-authorized provisional physical conventions remain
explicit assumptions; no producer confirmation or detector-closure claim is
invented. The [active plan](plans/CMS_CALIBRATED_JETCLASS2_HLT_RESPONSE_THREE_FAMILY_IMPLEMENTATION_PLAN.md)
and [response contract](contracts/CMS2JC2_RESPONSE_PREPARATION.md) now document
the implemented staged execution and its remaining remote gates.

Implemented an isolated 15-task CPU acceptance stage, 71-task primary science
stage (27 primary + six sensitivity fits), and nine-task separately authorized
confirmation/offline-only transfer stage. Added `campaign`, `submission`,
`orchestration`, `tasks`, `measurement`, `synthetic_acceptance`, `diagnostics`,
and `plots` under `src/hlt_classification/cms2jc2_response/`, extending the shared
engines, CLI, absolute-path worker and focused tests. New v1 artifact kinds are
listed in the reusable contract. No existing ladder contract or raw dataset
was changed. Internal donor files remain those in `provenance.DONORS`, baseline
commit `2b4c2531c39118edebc8c7890d312f2285ca8eab`; see `LEGACY_SOURCE_MAP.md`.

Queue safeguards: exact reviewed canonical plan hash and stage phrase; durable
pre-submit intents and exact job receipts; exclusive worker claims; source and
installed numerical-library byte checks; recursive authenticated dependencies;
last-published output inventories; read-only monitoring and explicit ambiguous-ID
reconciliation. Same-source recovery refuses active/pending/unknown jobs, reuses
authenticated completed outputs, and restarts approved terminal tasks in fresh
attempt directories. It never cancels or changes other jobs. Full science needs
the actual 20k/100k acceptance/resource lock; excess resources block submission
without reducing scientific populations or candidate counts. Serial evaluation
currently requests one CPU; fit preprocessing uses spawned processes, followed
by allocation-bounded numerical threads. No claimed unmeasured speedup.

Diagnostics now include all-observation scalar/joint-tail counters, bounded
quantiles with sampling-error disclosure, histogram occupancy, rare categories,
class-only reporting, fit bridge/count audits, support/clamp flags and a shallow
file-disjoint two-sample test. The current three-file selection role cannot
satisfy its minimum four-file split, so that diagnostic reports unavailable.
200 stratified physical examples for each permitted role and 50 worst-selection
examples have SVG panels and operation/validity traces. Reverse exact-merge-tie
ordering is a separately labelled diagnostic on the fixed selection examples.
Neither labels nor these diagnostics feed fitting or primary selection. The
selected response is frozen before independently claimed confirmation/transfer.
Operational completion is distinct from qualified/unqualified/inconclusive
science. JetClass2 native HLT and final-test readers are absent from this route.

Validation on `tagging-hlt`: **123 passed in 311.98 s** (83 response tests plus
40 existing scouting-foundation/JetClass2/split-registry tests). Tests include
actual tiny ROOT -> all 27+6 fits -> selection -> claimed five-replica
confirmation -> offline-only JetClass2 ROOT transfer, plus lifecycle failures,
source/byte changes, ambiguous submission, exact-ID recovery, combined CPU
limits and resource refusal. Tiny fixture minima are monkeypatched only inside
tests, never production. AST/whitespace checks passed for all 48 response Python
source/CLI/test files; `git diff --check` passed, with existing LF/CRLF warnings.
CLI `--help` and provisional `readiness` were exercised without scheduler access.

Latest real local fit-role previews on current source completed for A_L/B_L/C_L,
each with 16 location + 16 disjoint residual CMS jets and byte-replay checks.
They use `artifacts/cms2jc2_response_provisional_preview_v4_{A_L,B_L,C_L}/`:

- A_L: report `f1cf29979f91714583ba01e0f565851324186f4afb0194358e4f8afae433ac43`, 41.64 s.
- B_L: report `b6d75fb2638d2db261514c10acb614bfbcd196b7b7f907ae9906744a910f4c95`, 46.88 s.
- C_L: report `8580cc9cbf1a4c50dac05ac56ba470ae68b856f88e5653bfff197c4b17aaa87b`, 44.00 s.

These are explicitly development-only, not held-out closure or SPORC acceptance.
All reproduce the warning of 10/16 location and 15/16 residual jets resolved;
unresolved jets remain counted. The three new preview directories total roughly
7.2 MiB. No raw ROOT copy, full proxy dataset or rolling resume was written.

**Exact next task:** commit/push only this implementation, use a clean dedicated
SPORC worktree, run/authenticate the separate metadata preparation, then review
and authorize the real 15-task CPU acceptance plan. Full 71-task science remains
gated by those measurements; nine-task confirmation is authorized later. A
contradictory Luka reply needs new convention/source-bound artifacts, not edits
to existing fitted responses. No remote connection, Slurm submission, job
mutation, dependency installation, commit or push occurred during implementation.
Unrelated dirty work and all earlier development artifacts are preserved.

## 2026-09-17: explicitly provisional CMS2JC2 response implementation resumed

The user authorized likely assumptions while awaiting Luka's reply. This
supersedes the earlier compatibility-only pause below, but does not authorize
remote submission or waive physical, source, role or resource boundaries.
The active plan now records that amendment. New contract
`CMS2JC2_RESPONSE_PROVISIONAL_COMPATIBILITY/v1` is distinct from verified review;
its exact registered policy is validated, not silently edited. Local assumption
artifact `artifacts/cms2jc2_response_native_provisional_v1.json` has hash
`f493318e5719601ad51a14aaa65ae24ae2c5681df21148016089b014db65f80a`.

Assumed conversions: GeV momenta; CMS cm -> mm; JC2 stored mm; JC2 D0 sign
flipped into the CMS dxy convention; explicit uncertainty/validity and PID
rules; regular CMS PF excludes lost tracks. Preserve native p4 and vertex
references. Known raw-CMS/PUPPI-JC2 and reference-point differences are disclosed,
not declared equivalent. Fitted responses/transfer reports remain provisional
and cannot claim physically qualified JetClass2 detector transfer.

Added integrated `topology`, `records`, `parallel`, `response`, `worker`,
`metrics`, `conditioning`, `evaluation`, `selection`, `support`, `transfer`,
`storage`, `graph`, `assumptions`, and `development` modules under
`src/hlt_classification/cms2jc2_response/`; extended the thin CLI and focused
tests. The generator includes ordered non-overlap merges, singleton loss/split,
additional components, output identity/validity states and correlated continuous
response. Calibration records stay in RAM. Parallel sampling has deterministic
per-file quotas and explicit N/k weights. The common fitting engine distinguishes
20k/100k probes from full registered fits and refuses partial populations.

Evaluation includes six blocks, explicit missing coverage, fit-only metric
scales, separately applied particle/category/kinematic conditions, common
source-file bootstrap, replica averaging and independent qualification. Sparse
observable/cell backoff and numerical summary error bounds are disclosed.
Class labels are available only through an explicit held-out diagnostic route;
they never become fitting/generation inputs. Locked JetClass2 transfer rejects
native HLT and reports support clamps/unseen states. The 70-task graph registers
27 primary + 6 sensitivity fits, with 2x16 CPU fit lanes and 4x8 CPU evaluation
lanes (64 combined); it is not yet a live executable science coordinator.
Atomic storage guards count partial attempts toward the 12 GiB cap and enforce
the 2 GiB reports/examples cap plus free-space headroom.

Validation: **94 passed** in 110.10 s: 54 focused response tests plus the 40
existing scouting-foundation/JetClass2/split-registry tests. `git diff --check`
passed (existing LF/CRLF warnings only). The six response test files cover
conditional models, whole generation, validity/replay, sampling, summary error
bounds, paired statistics, one-SE selection, isolated transfer, combined CPU
lanes, storage caps and partial-population rejection. No package installation,
environment change, remote connection, commit, push or Slurm mutation occurred.

Bounded **real local fit-role** checks completed for A_L, B_L and C_L, each on
16 location + 16 disjoint residual-role CMS jets; inference on four residual
examples replayed exactly. Source bytes were checked before/after each latest
preview. These are not selection/confirmation closure and not remote acceptance.
Artifacts: `artifacts/cms2jc2_response_provisional_preview_v3_{A_L,B_L,C_L}/`.
Development-report hashes:

- A_L: `1bbe4ae6ce0d4ac40c4e9437f95640e2c8298697e78606ea04ad539a7b3cadb2`
- B_L: `ffe1490cbc776a03a2dcde9a6f9f49ca31eec6e67632435e114de4ddfa0182e9`
- C_L: `5b5e79304adb05216d4b56f3d65201027b62eb208f4f263e1f08701f05b87911`

The first real preview exposed overflowing simple-family momentum extrapolation;
the tracking-linear term is now tracking-only, with bounded calibrated scale/
response envelopes and counted clamps. Exact association search gained an
admissible bound and cost-dominance pruning. Current development limits still
resolve only 10/16 location and 15/16 residual jets in this small sample:
four search-limit and three object-limit cases. Those jets are counted, not
claimed as successful matches. This warns that production association limits/
performance need measurement; it is not evidence of detector closure. Earlier
v1/v2 previews are retained as development history and must not certify current
source. No raw ROOT copy or full proxy dataset was written.

**Still not full-science queue ready.** Exact next work: finish the staged live
coordinator/receipts/recovery, complete non-selecting tails/discriminator/plots,
independent confirmation execution, and production CPU miniature/resource gates.
The CLI reports that boundary explicitly. The source-bound preparation job is
still the only live submission route. There is no need to wait for the producer
reply to continue this implementation; a reply contradicting an assumption
requires a new bridge/source-bound run. Existing campaign code and unrelated
dirty work were preserved. Donors remain the internal baseline
`2b4c2531c39118edebc8c7890d312f2285ca8eab`, documented in `LEGACY_SOURCE_MAP.md`.

## 2026-09-17: CMS2JC2 response implementation checkpoint; physical-review gate

Implemented preparation and independently tested calibration primitives under
`src/hlt_classification/cms2jc2_response/`, thin CLI
`scripts/cms2jc2_response.py`, CPU preparation worker
`sbatch/run_cms2jc2_response_cpu.sh`, and focused
`tests/test_cms2jc2_response.py`. The package's optional `response` extra adds
CPU scikit-learn/threadpoolctl dependencies without installing/changing any
environment. Reusable boundary:
[CMS2JC2 preparation contract](contracts/CMS2JC2_RESPONSE_PREPARATION.md).

Implemented: source-bound immutable artifacts and nine-candidate registry;
physical-review refusal and raw adapters; latest-cycle source authentication;
whole-file outer/inner splits and exact nested budgets; offline-only JC2 reads;
keyed randomness, neighbourhood features, partial/group association with exact
bounded search; portable table/spline/tree conditional models; held-out
temperature and correlated residual primitives. These are **not yet an
integrated scientific response workflow**. Missing are complete topology/set
generation, common bounded calibration records, six-block evaluation/selection,
confirmation/transfer/visuals and the bounded full scientific DAG. The CLI
explicitly reports `science_queue_ready: false`; no science-submit route exists.

The historical split was recovered read-only from local backup archive
`C:/Users/22rya/ComputerScience/CERN/copy_of_HLT_Classification_RC/HLT_Classification_RC.tar.gz`,
member `HLT_Classification/checkpoints/pmard_pilot_c3e40850_prefix_recovery_r5/data/splits/split_manifest.json`.
Its content hash is
`5a8cbb66522a0b6e0131d009a464b4e7a257260899afd3df67a387276c977838`.
Real local metadata/scalar audits checked all 53 latest-cycle schemas and then
checksum-verified the 32 historical training files before/after replaying their
exact selection. No real particle arrays or historical validation/final-test
populations were used for response fitting.

Whole-file allocation is feasible: response fit **2,222,819**, selection
**277,546**, confirmation **277,490** jets. Inner location/residual counts are
1,778,252/444,567. Both production-source categories occur in every role.
Selection and confirmation each contain only three files: their millions of
objects must not be advertised as many independent uncertainty clusters.
Local artifacts in `artifacts/cms2jc2_response_development/` include inventories,
roles, nested masks, unresolved compatibility review, source/environment records
and development capacity evidence
`4aef782acb621af23933c42d2564f84026301f7426f053126ffb9cd8f0bc8117`.
This is dirty-checkout, non-executable evidence, not a production acceptance lock.
The durable artifacts are approximately 3 MiB; no raw dataset was duplicated.

Validation: **71 passed** across `test_cms2jc2_response.py` (31 tests),
`test_scouting_foundation.py`, `test_jetclass2_delphes.py`, and
`test_jetclass2_delphes_split_registry.py`. New coverage includes physicality,
unknown PID/missing errors, RNG replay, exhaustive small association/split
references, nonconstant known-response fixtures for all three families,
residual variance accounting, actual synthetic ROOT reads with no HLT/labels,
nested-mask corruption and guarded dry/ambiguous submission behavior. CLI help
and readiness checks pass. Initial scaffold check remains 3 passed/1 failed due
to the four previously recorded broken links in unrelated scratch snapshots.
Those files were not edited. No remote, installed-Weaver or SPORC miniature
acceptance is claimed; this CPU preparation step does not exercise Weaver.

**Stop condition:** the response plan requires genuine common-field review.
The existing JC2 handoff still calls stored-mm/tracking-error/producer semantics
provisional. It cannot prove compatibility with CMS dxy/dz/significance or PF
four-vector weighting. A question was sent requesting producer source/notes.
The historical-split question is now resolved locally and needs no user action.
Next obtain those field definitions, or an explicit amendment authorizing a
named provisional/reduced-field study; then finish the integrated workflow and
real CPU resource/acceptance gates. Do not silently import the older classifier
campaign's provisional authorization. No commit, push, remote connection,
Slurm submission/cancellation or existing campaign mutation occurred.

## 2026-09-17: CMS-calibrated JetClass2 HLT response comparison plan

Added the implementation-authoritative design
`docs/plans/CMS_CALIBRATED_JETCLASS2_HLT_RESPONSE_THREE_FAMILY_IMPLEMENTATION_PLAN.md`
and indexed it in `docs/plans/README.md`. This is documentation only: no response
implementation, generated dataset, commit, remote connection or Slurm submission.
The plan compares basic calibration, neighbourhood-aware smooth calibration and
small boosted-tree responses, each at three complexities and nested 250k/1M/FULL
budgets (27 fits), plus six association-sensitivity fits. Full roles require at
least 2M calibration, 250k selection and 250k locked confirmation CMS jets, with
whole-file disjointness inside the original training reservoir. Capacity is an
explicit preflight requirement, not a claim that those new partitions exist.

Fixed label-independent jet/object random keys provide reproducible statistical
response. Fitting is label-blind and does not optimize KD performance. The plan
requires a verified common physical interface, partial/group associations rather
than forced full-cardinality truth, joint residual calibration, independent
qualification, bounded CPU/storage use and JetClass2 offline-only access.
Existing JetClass2 memberships are retained for the initial transfer audit, with
their historical HLT-match selection disclosed. Original final tests stay sealed.

Local CMS compact data were confirmed at the documented Windows path (53 ROOT
files); RC paths are recorded but SPORC compute-node access/CPU partition policy
remain unverified. Next: implement the read-only provenance/compatibility/capacity
audit and versioned bridge/split contracts, then real CPU-worker acceptance before
any full submission. No code was migrated and no executable contract was changed.

Documentation checks: all 118 relative links across the new plan, plan index and
handoff resolve; no trailing whitespace; targeted `git diff --check` passes.
The scaffold suite is unchanged before/after this edit: 3 passed, 1 failed due
to four pre-existing broken handoff links inside unrelated `scratch/` source
snapshots. Those snapshots were not modified. No remote/Weaver acceptance was
run or claimed for this documentation-only step.

## 2026-09-16: bounded JetClass2 raw jet-pair diagnostic

Added `scripts/audit_jetclass2_jet_pairing.py` and the reusable
`jetclass2_delphes/pairing_audit.py` to investigate the proposed raw HLT/offline
jet mispairing hypothesis. The diagnostic contract and usage are documented in
`docs/JETCLASS2_JET_PAIRING_AUDIT.md` under
`JETCLASS2_DELPHES_JET_PAIRING_AUDIT/v1`. It reads a bounded sample from frozen
train/validation membership, verifies source bytes, and compares actual global
jet-axis distances and kinematic correlations with no-self, same-class/source,
same-class/source/pT-bin, and same-class/file shuffled offline partners.
It reports optional producer matching distances and source/entry identities.
No particle matcher, training objective, source campaign, job, or final-test
role is modified/accessed. Particle chunks are transient; output is small JSON.

Pre-change focused reader/selection checks passed 4 tests. Seven new diagnostic
tests passed, including deliberate within-class pairing corruption, phi wrapping,
singleton exclusion, frozen-membership sampling, source-role isolation, and
tamper rejection. This CPU diagnostic neither uses nor certifies Weaver/GPU
training. A real local run then read 8,192 registered training jets across 32
checksum-verified files (16 per production source). Report:
`artifacts/jc2_jet_pairing_audit_train_v1.json`, hash
`94d4e504ad76e1b7834a2981c8d84eec273819a30d63ec74e8fef85188aed9fe`.
Median constituent-sum axis delta-R was 0.02914 versus 1.90000 for same-class/
source shuffled partners. Stored producer distances agreed with independent
constituent-sum calculations within 2.41e-7. This argues against wholesale
random pairing, but is not truth-level certification or a full-population audit.

The sample revealed a class-specific concern: X_ee (60 jets) and X_mm (102)
had median delta-R 0.52058 and 0.37836 and pT rank correlations 0.14304 and
0.06329. QCD/X_bb/X_cc medians were 0.02334/0.03176/0.03382 with pT
correlations 0.82359/0.76867/0.71995. Tau modes also had larger angular tails.
Next scientific diagnostic is per-class D033-to-D000 metric degradation and,
if warranted, producer reconstruction/jet-association review. These observations
alone do not establish the cause of the distillation failure. No SPORC jobs
were submitted or altered; no training or final-test access occurred.

## 2026-09-16: JetClass2 D033-only C25/P75 D000 endpoint ablation

Implemented the isolated one-fit experiment requested to diagnose the salience
MT20 coarse-ladder D033-to-D000 collapse. Authority:
`docs/plans/JETCLASS2_DELPHES_D000_D033_ONLY_C25P75_500K_PLAN.md`; reusable
contract: `docs/contracts/JETCLASS2_DELPHES_D000_D033_ONLY_C25P75.md`.

The new `jetclass2_d000_d033_only.py` package/CLI and
`sbatch/run_jetclass2_d000_d033_only.sh` authenticate and reuse the completed
MT20 `JC2SMT20_COARSE_D033_from_D066` T=2 train bank, then train exactly one
fresh D000 model with only 25% CE + 75% D033 KD at T=2. Its D000 view,
initialization/sampler seeds, architecture, schedule, batch, data membership,
validation population, and matching foundation remain paired to the existing
MT20 endpoint. D066/U100/U050/U000 teachers are forbidden. The independent
one-job root neither depends on the remaining source DAG nor mutates it.

Creation requires checksummed completed M0HLT, U000, D033, MT20 D000, and D033
reducer artifacts. The job runs a real single-teacher C25/P75 backward check,
discards the probe model, reseeds, and performs the full fit. Views and teacher
probabilities are RAM-only; selected weights and compact reports persist.
Routing is fixed at creation to SPORC debug or tier3. Local focused evidence:
the eight new tests plus the eleven unchanged JetClass2 MT20 tests pass (19
total); adding five shared JetClass2 training tests gives 24 passing. No SPORC
job was submitted and no final-test role was accessed.

## 2026-09-16: isolated JetClass2 native offline+HLT concatenation oracle

Implemented the user-requested one-fit, one-Slurm-job comparison under
`JETCLASS2_DELPHES_NATIVE_CONCAT_*/v1`. Authority:
`docs/plans/JETCLASS2_DELPHES_NATIVE_CONCAT_500K_PLAN.md`. Thin CLI:
`scripts/jetclass2_delphes_native_concat.py` (`create`, dry/live `submit`,
worker-only `run`, `results`); worker:
`sbatch/run_jetclass2_delphes_native_concat.sh`.

One canonical ParT attends jointly to native offline followed by native HLT
particles, with a learned two-code reconstruction embedding. Numerical inputs
retain each side's own native normalization and p4. No matching is performed,
no residual HLT is double-counted from persistent U000, and no tokens are
trimmed. Capacity is 240 per reconstruction / 480 combined. It is explicitly
an offline-enabled oracle, not an HLT-deployable checkpoint. The CE fit uses
the exact existing 500k membership, held-out validation roles, control seed,
batch 256 and 100-pass/min60/ES15 schedule. Reference U000 is labelled
**persistent-HLT**, not pure offline. The printer includes baseline/control,
U000, concatenation, recovery and per-class rejection/censoring; it can add
the existing static fusion result once that report is durably complete.

Creation binds the existing learned-handoff campaign, completed M0HLT/U000/
CE_SINGLE_D000 artifacts and validation partition. It does not wait for static
fusion or other ladder jobs. Default requested envelope is one SPORC tier3
A100, 8 CPUs/workers, 128 GiB and 48h. Its single worker performs source/receipt
authentication, RAM-cache construction, installed-Weaver parity, full-batch
longest-actual-view memory/optimizer probe and a discarded real-data miniature
before resetting weights/RNG/optimizer and starting the full CE fit. Resource
and walltime guards do not silently alter the batch or shorten training.
Only selected weights and compact reports persist. Immutable claims and
guarded submission receipts prevent duplicate writers. Existing source roots,
matching outputs and queues are never modified.

Local evidence: pre-change donor tests 33 passed / one Weaver skip; combined
data, learned-handoff, withdrawal and new-concat tests 64 passed / two Weaver
skips. Separate tests against the existing isolated Weaver-core 0.5.3 CPU
installation passed (3), including tagged-vs-ordinary zero-tag forward/input/
parameter-gradient parity and nonzero source-embedding gradients. No shared
environment was modified. This is local one-job tooling readiness, **not a
claim of measured SPORC concatenation acceptance**. That study-specific check
is performed in the job and must pass before science. Nothing was committed,
pushed or submitted here; unrelated dirty files were preserved.

Next: commit/push only this study's eight new files plus its donor-map entry
(stage this HANDOFF hunk separately from unrelated edits), then create a fresh
source-pinned SPORC worktree/root. Reference the existing
`jc2_salience_learned_handoff_withdrawfix_b0154465_r1/campaign_spec.json`.
Create emits the one-command plan and dry ledger. Explicit live phrase:
`AUTHORIZE JETCLASS2 NATIVE CONCAT SINGLE JOB`. No current job needs cancellation.

## 2026-09-15: CMS dense installed-Weaver synthetic-input repair

The SPORC pre-submission tests at `f33b8bff611786129ff941aa09aeca2d9856af37`
passed 20 tests, then failed the installed-Weaver wrapper check with
`Invalid CMS inference`. Execution stopped before campaign creation or job
submission. The test fixture generated all four p4 components independently:
its D080 sample contains 57 negative-energy particles and 86 with E <= |pz|
out of 120. These are invalid inputs to Weaver's rapidity/pair geometry; the
local fake pair module returns zeros and therefore did not detect them.

`tests/test_cms_salience_learned.py` now constructs positive-energy, unit-mass
synthetic p4 with E = sqrt(px^2 + py^2 + pz^2 + 1), following the existing
native Scouting parity helper. An unconditional regression checks the fixture
without Weaver, and the installed test exercises alpha 1 and .5 as well as
exact alpha-zero/extracted-model equality. Production code, scientific
contracts, actual CMS inputs, matching and schedules are unchanged. The
nonfinite guard and exact parity requirement are not relaxed.

Local focused-plus-neighbor verification: 72 passed, one installed-Weaver
test skipped in the ordinary local environment. A separate isolated temporary
Weaver-core 0.5.3 CPU installation then passed both fixture and installed-model
checks (2 passed). The original committed fixture reproduced the same
`Invalid CMS inference`; switching only to physical p4 yielded finite 30x15
probabilities with the identical model. No existing Conda environment was
modified. This CPU evidence is not SPORC/A100 acceptance, which remains
required. Push the repair and use a fresh source-pinned worktree/root
to rerun the original preparation instructions; no cancellation or cleanup
is required for the pre-submission test failure.

## 2026-09-15: isolated native-CMS salience Strategy-B dense ladder

Implemented the user-approved single dense study under the new
`cms_salience_learned/` package and `CMS_SALIENCE_LEARNED_DENSE_*/v1`
family. Authority is `docs/plans/CMS_SALIENCE_LEARNED_DENSE_500K_PLAN.md`.
It uses original native CMS/Scouting 21-feature/15-class inputs, 500k train,
250k validation and a sealed 250k final-test commitment, PT_LINEAR salience
matching, and persistent-HLT support. No JetClass2 assignments, weights,
reference metrics or resource acceptance are imported.

The one spine is U000/U033/U066/U100/D080/D060/D040/D020/D000. There are
20 fits (three fresh references, one global direct-KD control, eight cold
acquisitions and eight warm withdrawals), eight exact extractions, sixteen
probability reducers, aggregate and completion: 46 science tasks. Each next
arrow receives only the previous ordinary carrier's probability bank. Native
HLT baseline/direct/final-rung initialization seeds are matched. Reporting
uses a held-out validation sub-role; final-test branch reads are forbidden.

CLI: `scripts/cms_salience_learned.py`; worker:
`sbatch/run_cms_salience_learned.sh`. Creation emits immutable prepare/gate/
science plans and all dry-run ledgers. Live science requires the new campaign's
genuine SPORC A100 production-worker acceptance. Defaults: 16 CPUs, 16 spawned
preprocessing workers, 192000 MiB host RAM, one A100; configurable CPU/worker/
memory request is frozen at creation and checked against its own measurement.
No existing campaign/spec/queue was changed, and nothing was submitted.

Local combined focused evidence: **71 passed, one installed-Weaver test
skipped** (Weaver absent); the new campaign accounts for 20 passed tests.
Tests include real synthetic ROOT preparation through selected
teacher bank -> acquisition -> withdrawal -> exact extraction -> next bank,
all four loss routes, process-parallel matching parity, source/identity seals,
disjoint validation roles and the staged submission guard. Prior donor-area
baseline was 51 passed. This is local tooling readiness, **not** genuine A100
acceptance. Next: commit/push exact source, create a fresh SPORC worktree/root,
submit preparation, run the gate, then authorize science. Original native CMS
raw data and split-manifest accessibility on SPORC still need that remote check.

Existing uncommitted native decoder/category-count additions were preserved;
review them with the source owner when assembling the pushed source snapshot.
No files in current CMS/JetClass2 campaign roots were modified.

## 2026-09-15: original Tigris Strategy-B runtime and report repair

User-supplied logs from `hcwlfh1` jobs `117683`, `117675`, `117679`, and
`117673`, pinned to `fd1ed1d01d54bf2ad4d42ffa6311432263a14770`, all show
successful full-population RAM-cache preparation followed by
`ValueError: TRI60 optimization recipe differs`, before the first training
update. The original learned-handoff adapter supplied the legacy
`Tri60TrainingRuntime.warmup_fraction` as `.03`, whereas the shared trainer's
runtime contract requires `.05`. Its separately supplied pass-based
`LR_SCHEDULE` already defines the intended three-pass warmup and overrides
that fractional field for actual optimizer updates.

The Tigris adapter now retains the validated `.05` legacy field and validates
the runtime, explicit LR schedule, and early-stopping policy before loading
fit artifacts/caches and at preflight entry. Actual training remains warmup
1--3, hold through 45, cosine decay through 60 to `1.5e-5`, then a constant
floor through maximum pass 100, with minimum 60 and patience 15. The former
preflight exercised model forward/backward without traversing the shared
trainer's runtime check, which is why its success did not catch this failure.

A synthetic end-to-end run through `run_fit -> train_tri60_node`, using the
unmodified scientific runtime and a tiny CPU model/cache, then exposed a
second failure at terminal report publication: `LearnedNode` lacked
`representation_seed_alias`. A read-only property now returns the same `None`
already recorded in every registered node payload. This prevents a late
report failure after a complete fit; it does not introduce representation KD
or change any randomization seed. Graph payload/hash and recipe payload were
compared with committed source and are unchanged. No contract version change
or donor migration is required. The shared TRI60 trainer, matching/data code,
and all SPORC/JetClass2 implementation files were left untouched.

Pre-change focused coverage passed 15 tests with the installed-Weaver-only
seed test excluded. New regressions reproduced the runtime failure for all
25 fit authorities and the real CE training entry; after both repairs, the
focused suite passed 44 with that one exclusion. Coverage checks exact LR
boundaries, early-stop policy, fail-before-cache behavior for both fit and
preflight, complete training/checkpoint/report publication, selected-model
reload, and absence of rolling resume. The combined original learned-handoff,
output-handoff, offline/HLT fusion, TRI100 and TRI60 suites passed 123 tests
with the same one installed-Weaver exclusion (10.44s). These are local
synthetic regressions, not a new real-Weaver/GH200 acceptance result.

No commit/push, job cancellation/submission, remote artifact mutation, or
final-test access occurred. Next: push the repair, inspect the exact original
science ledger and current job states, then use the existing source-pinned
restart-zero recovery after its terminal-subject prerequisite is met. Preserve
authenticated completed source/control outputs; do not edit the immutable
`fd1ed1d0` campaign or rerun failed jobs from the unchanged old worktree.

## 2026-09-15: JetClass2 salience MT20 preflight inference-tensor repair

SPORC gate job `21692827`, executing source `d52a8714`, authenticated and
materialized the complete 500k U000 RAM cache, then failed at the first
train-mode student forward.  The preflight had created its shared CUDA input
tensors inside `torch.inference_mode()` for two teacher forwards and then
reused those inference-only tensors for the student.  Installed Weaver's
input BatchNorm therefore could not save its input for backward and PyTorch
raised `RuntimeError: Inference tensors cannot be saved for backward`.  This
was a preflight tensor-lifetime bug, not a matcher, data, resource, loss, or
scientific failure.  No science stage was submitted.

The preflight now creates the shared input tensors before entering inference
mode.  Teacher inference remains gradient-free; the student receives the same
values as ordinary tensors and completes its exact C20/P80 two-teacher loss and
backward probe.  Matching, memberships, model, optimizer, loss coefficients,
temperature, campaign graph, and durable artifact semantics are unchanged, so
no scientific contract version changes.  A BatchNorm regression test exercises
the exact teacher-inference-to-student-backward boundary.  The focused MT20
suite passed 11 tests, and the combined JetClass2 training, salience and MT20
suite passed 29 tests in 79.32 seconds.  Whitespace checks passed.

Because the failed immutable campaign is pinned to the old source commit, do
not run it from repaired source or alter its spec.  Commit and push this repair,
then create a fresh source-pinned MT20 campaign/root and rerun the three-task
gate.  The completed selected salience foundation and screen remain reusable
read-only parents; matching is not repeated.

## 2026-09-14: JetClass2 learned-handoff withdrawal output compatibility

SPORC preflight job `21670166`, executing source
`15094633f9aa3a0e3f9e418704ac3c0a46dac11d`, finished the shown U050/U000
validation caches but failed in the withdrawal-loss probe: the new
`FusionOutput` exposes `primary_states` and `primary_mask`, whereas the shared
withdrawal objective consumes `hlt_states` and `hlt_mask`. The failure was a
Python interface mismatch, not evidence of bad matching, resource exhaustion,
or scientific performance. The dependent launcher `21670296` remained blocked;
this gate did not release learned-handoff science.

Added read-only compatibility aliases to the JetClass2 output only. They return
the original lower/primary tensors with no copies or detach, including for
U-side transitions. The shared legacy loss, scientific coefficients, model
parameters, checkpoint schemas, matching, populations, schedules, and campaign
DAG remain unchanged; no contract version change or donor migration is needed.
Both the preflight loss call and production withdrawal/morph training consume
the repaired output. The source map records the reused API and producer commit.

The original focused suite passed 15 tests and skipped its installed-Weaver
test. New tensor/dispatch regressions reproduced the exact AttributeError
(9 failures before the fix, with 3 exact-zero cases already passing). After
the fix, the focused suite passed 27 with one installed-Weaver skip. Coverage
includes the fixed six loss coefficients, primary padding masks, directed
consistency gradients, invalid-surface rejection, and context-free dispatch
for both withdrawal roles. The installed-Weaver test now additionally exercises
acquisition followed by withdrawal backward and the exact-zero endpoint.
The final combined learned-handoff, salience, legacy fusion and adjacent
handoff selection passed 66 tests (62.88s), with one installed-Weaver skip
and one older Weaver-only seed test explicitly deselected. An earlier broad
run reached 66 passes but failed that older test solely because Weaver is
absent. CLI help and scoped whitespace checks passed. These results do not
substitute for genuine installed-Weaver/A100 acceptance.

No commit, push, remote job/artifact mutation, or final-test access occurred.
Real installed-Weaver/A100 acceptance remains required. The next execution must
use freshly committed/pushed source, a clean pinned worktree and fresh campaign
and launcher roots; the completed salience screen/foundations are authenticated
read-only parents and do not need retraining. Rerun the four-task gate before
the science dry/live stage. Do not edit the failed immutable campaign's source
pin or release its dependency-blocked launcher.

## 2026-09-14: JetClass2 salience MT20 three-spine implementation

Implemented the isolated SPORC `TRAIN_500K` successor to the earlier
multi-grandparent experiment. It starts from fresh M0HLT and persistent-HLT
salience U000 references, then registers DIRECT, COARSE and DENSE
path-density ablations. Each downstream fit receives C20/P80 T=2 logit KD from
every earlier selected model in its own spine: the immediate teacher receives
0.50, historical teachers share 0.30 with nearest-first geometric ratio 1/2,
and a single teacher receives 0.80. Cross-spine teachers, ensembles and warm
starts are absent. The campaign has 16 fresh fits, 12 compact train-bank
reducers, aggregate and completion (30 science tasks), plus three staged gates.

The implementation directly authenticates and reuses the completed
`SALIENCE_PT_LINEAR` foundation and its immutable selection evidence, without
loading losing candidates, recomputing assignments or repeating the screen.
It retains the exact 500k/1m/1m memberships. Multi-teacher components are
identity-joined, accumulated in float64, normalized and converted to float32
only in RAM; mixture arrays, particle views and rolling state are never durable.
Only selected checkpoints, reports, task receipts and compact single-model T=2
train banks persist. The new real-A100 preflight exercises the selected U000
view, production model, two-teacher mixer and exact C20/P80 backward pass.
Science cannot be dry-run or submitted before that source-bound gate passes.
The new operational routing policy sends every gate, fit, reducer and metadata
task to SPORC `debug` by default, with repeatable exact-task `--tier3-task`
overrides recorded in the dry/live command plan. Both routes preserve the same
one-A100, 8-CPU, 72-GiB environment and QoS; all task walltimes are below the
registered 24-hour debug ceiling. Already-submitted tasks are not mutated
between partitions and instead require exact cancellation/recovery.

Added the active plan and reusable contract, native campaign/mixer/production
modules, a thin CLI and SPORC worker, and focused graph/loss/mixing/Slurm tests.
Pre-change JetClass2 salience/training/production baseline: 23 passed in 484.46s.
After the three-spine and winner-only reuse revision, the combined MT20,
salience, training and production suite passed 31 in 515.58s. After the
debug-default/task-level-tier3 routing revision, the focused MT20 suite passed
10 in 22.29s and the combined MT20/debug-profile/SPORC execution suite passed
37 in 359.53s. Python compilation, CLI help,
shell syntax and scoped whitespace checks passed. No commit, push, campaign
creation, SPORC submission, installed-Weaver/A100 execution, or final-test
access was performed. Existing salience, learned-handoff, Tigris and user-owned
worktree changes were preserved. Next, push exact source, create a fresh
campaign from the completed winner foundation and screen root, dry/live the three-task gate,
then dry/live science only after the gate passes.

## 2026-09-14: exact deferred launch for salience learned handoff

Added a source-pinned, CPU-only two-stage launcher for the 500k SPORC
learned-fusion campaign. It binds one exact live salience-screen ledger and
screen `complete` job, then uses `afterok` to create and submit only the four
campaign gates. A second launcher binds all four exact gate job IDs, validates
the science gate, materializes the canonical 87-task science dry run, and only
then submits science. Both workers authenticate their own exact Slurm receipt
and SPORC allocation; they do not poll, hold GPUs, mutate either salience
screen, or access final test. A failed screen or gate leaves downstream work
dependency-unsatisfied. Focused salience plus learned-handoff tests passed 26,
with the one local installed-Weaver test skipped because Weaver is absent.
The first live gate exposed a non-scientific authentication-report constructor
collision (`artifact(..., kind=...)`). The field is now unambiguously named
`diagnostic_kind`, and a regression test executes that exact gate branch. The
old source-pinned gate descendants must be retired by exact ID; recovery uses
a fresh campaign and launcher pinned to the fix commit.

## 2026-09-13: JetClass2 salience learned-fusion handoff implementation

Implemented the isolated `TRAIN_500K` Strategy-B campaign for SPORC on top of
the selected persistent-HLT full-cardinality salience foundation. The frozen
graph has fresh `M0HLT` and `U000` references; DIRECT, COARSE, and DENSE spines;
and DIRECT/ACQUIRE/WITHDRAW fits at every one of the 14 arrows, including the
U-side arrows. The campaign owns 54 fresh fits and a 91-task preregistered DAG.
No old campaign, assignment, probability bank, model, or source worktree is a
runtime dependency beyond the authenticated salience screen/foundation.

The user clarified the ablation scope during implementation. The ten-fit
control panel runs exactly once, attached to DENSE `U100->D080` and
`D020->D000` where rung-specific comparisons are needed. Its three global
controls now test `U000->D000`, not `U100->D000`: fixed U000 context, an exact
denominator-25 U000-to-U100-to-D000 context morph, and morph-checkpoint
withdrawal. The morph reaches D000 at pass 51 and keeps it through pass 100.
No stale U100-named global-control node remains.

Added versioned graph/recipe/data/model/cache/campaign/production contracts,
an asymmetric 17-input/11-output Weaver fusion model, exact physical primary
extraction, deterministic three-way validation partitioning, compact T=2
probability publications, detailed transition/control/extraction comparisons,
staged gate/science exact-DAG submission, monitoring, and terminal-only
restart-zero recovery. Fixed coordinate particle views, dynamic morph views,
hidden states, optimizer state, and in-progress best weights are RAM/device
only. Withdrawal releases its richer cache at pass 61; alpha-zero dispatch
does not touch context. Only selected checkpoints, reports, locks, task
attestations, and compact probability banks are durable. Final-test access is
absent.

Focused new/salience tests: 32 passed and one installed-Weaver architecture
test skipped because Weaver is not installed in the local Windows environment
(9.44s). The remaining historical Strategy-B tests passed 15 with its one
installed-Weaver test deliberately deselected (8.86s). CLI help, Python
compilation, shell syntax, and scoped whitespace checks passed. No commit,
push, campaign creation, SPORC job, or final-test access occurred. Before live
science submission, exact pushed source must create a fresh campaign, complete
the four-task SPORC gate (including the real A100 acquisition/withdrawal
miniature and measured paired-cache resources), pass a separate science dry
run, and receive explicit authorization.

## 2026-09-13: auxiliary debug-only profile continuation

User supplied pending auxiliary profile job 21628181 from prepared study
`jc2_offline_aux_500k_bee8bc48_r1` and requested debug execution to avoid the
tier3 wait. Its four-hour, one-A100, eight-CPU, 72-GiB request fits the supplied
debug partition limits. The idle debug node advertised no GPU, however, so
this is not evidence of an immediately available A100. The old worker pins
tier3; changing only its Slurm partition would fail allocation validation.

Implemented `continue-debug` in the auxiliary CLI, backed by new
`offline_aux/preparation_import.py` and narrow changes to campaign, execution,
submission and workflow. A fresh source-pinned study imports exactly nine
authenticated CPU task receipts (sample, seven target shards, normalize) from
the original study read-only. Seventeen committed preparation-kernel files
must be byte-identical across producer and consumer source commits. Original
receipt identities remain intact; banks are neither copied nor republished.
Only the new PREPARE profile task uses debug. All discovery, confirmation and
report GPU jobs remain tier3; scientific definitions are unchanged. Imported
payloads count toward the existing study storage limit. New artifact families
are PREPARATION_IMPORT/v1 and EXECUTION_ACCEPTANCE_DEBUG/v1, explicitly binding
debug measurement to the same A100 environment/resources on tier3. Ordinary
profiles and recovery keep their existing contracts. No old study is mutated.

The active auxiliary plan now includes the user-authorized operational
exception; its reusable contract documents the import and resource evidence.
LEGACY_SOURCE_MAP records internal helper reuse at producer commit
`bee8bc48a1e634d0858f689b6b733ee3b2232265`; no external donor was migrated.
Added `test_jetclass2_delphes_offline_aux_debug.py`. Pre-change auxiliary suite:
40 passed (80.48s). Final debug, pipeline, math and storage suite: 59 passed
(88.08s), covering read-only receipt/payload reuse, corrupted inputs, exact
single-profile submission, acceptance bindings, allocation and storage guards,
and unchanged tier3 discovery. CLI help and scoped whitespace checks passed.

No commit, push, remote cancellation, submission, installed-Weaver or A100 run
was performed here. Next: publish only this scoped change, create a clean RC
worktree and fresh debug continuation, audit its one-job PREPARE plan, then
submit that real GPU profile. Preserve the original bee8bc48 study and its
CPU outputs: the continuation depends on them. Any retirement of job 21628181
must first confirm its current pending state and exact original ledger binding.
The next scientific stage remains explicitly authorized only after the new
profile passes. Unrelated shared worktree changes were preserved.

## 2026-09-13: JetClass2 500k salience persistent-HLT three-spine implementation

Implemented an isolated SPORC pipeline for the user-requested JetClass2
Delphes `TRAIN_500K` study. It reuses the frozen 500k/1M/1M split registry and
the measured one-A100 resource envelope, but deliberately refuses to reuse the
old bottleneck assignments as salience evidence. Three fresh compact matcher
foundations are preregistered (`SALIENCE_PT_LINEAR`,
`SALIENCE_PT_QUADRATIC`, `SALIENCE_PT_QUADRATIC_CORE25`). A matched-seed U100
screen compares those three candidates with a bottleneck contextual control on
a deterministic stratified 75/25 validation firewall; only a salience candidate
is eligible to win. The production graph contains DIRECT, COARSE, and DENSE
only: 16 fresh fits, 12 probability reducers, aggregate, and completion (30
tasks). ULTRADENSE is absent.

The new JetClass2 adapter preserves the complete HLT skeleton. U000 places
offline content on matched HLT slots, keeps unmatched HLT particles, and adds
only unmatched offline particles as the removable tail. U progression removes
that tail; U100 retains HLT cardinality with offline matched-slot content; D
progresses to an exact offline-free native-HLT D000. Full-cardinality salience
assignments are compact identity/offset/int32 maps. Dense matching matrices and
all particle views stay RAM-only; there is no rolling optimizer resume. A
candidate-independent endpoint-linear-pT-weighted delta-R diagnostic is used
only as the late selection tie-breaker. Assignment locks require raw-row sample
recomputation as well as byte, lineage, cardinality, endpoint, and bounded exact
solver checks. Final-test particles remain inaccessible.

New native modules are `jetclass2_delphes/salience_{views,foundation,cache,
campaign,readiness,screen,production}.py`, with three thin CLIs and three
SPORC workers. The active plan and contract are
`JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT_500K_PLAN.md` and
`JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT.md`. The reusable salience
matcher/contract files are new implementation, not a legacy donor migration.
No old campaign, raw snapshot, split registry, job, or checkpoint was mutated.

Pre-change JetClass2 focused baseline: 63 passed (573.48s). The final combined
reusable-matcher and JetClass2 salience suite passed 27 tests (65.00s). The
wider JetClass2/SPORC regression selection passed 75 tests (617.10s). An
earlier combined run found one toy-fixture recomputation-count edge case; it
was corrected to use `min(sample_rows,file_rows)`. All three CLI import/help
checks and scoped whitespace checks passed; the local Windows bash executable
was unavailable for `bash -n`, while the three workers remain minimal
strict-mode wrappers of the established SPORC helper. No commit, push, Slurm
submission, installed-Weaver run, or A100 execution was performed locally.
Queue order is intentionally staged:
three non-scientific foundation DAGs, then the genuine A100/four-fit screen,
then a separately authorized production dry/live submission after the screen
lock exists.

The screen completed on 2026-09-14 with `SALIENCE_PT_LINEAR` selected. The
reusable matching guide and JetClass2 contract now record the exact SPORC
foundation/screen roots, mandatory retained artifacts, authentication example,
same-commit supported reuse path, and the source-contract boundary for future
graphs. Same-population recipe, seed, or graph changes may reuse the compact
assignments; population, decoder, schema, truncation, matcher, endpoint, or
support-policy changes may not.

## 2026-09-13: auxiliary PREPARE concurrent storage-audit fix

User-supplied PREPARE job 21627750 (`targets_VAL_SELECT_01`, source
`62b6d1bc560a97205465a55cb6ec481413675363`) printed completion of its 100k-row
target calculation, then failed in `storage_audit`. The audit listed a sibling
TRAIN_01 publication temporary and subsequently stat-ed it after its normal
removal: `.0001_hlt_pt.npz.s9d1hofv.tmp`. This traceback establishes an audit
race, not invalid targets, an OOM, or exhausted storage. The other six target
jobs 21627744--21627749 completed; normalize/profile 21627751/21627752 were
still pending in the supplied accounting. No scientific fits had started.

Changed auxiliary `contracts.py` to collect type/size with one non-following
stat. Vanished atomic-publication temporaries resolve to their completed
destination when present, counted once if also enumerated. Aborted temporaries
and the exact attempt-local transient submission claim can disappear normally.
Live temporaries still count; other missing listed entries, permissions,
symlinks, 64-MiB file and 4-GiB study violations fail. This remains a live storage
observation, not a transactional quota or substitute for receipt/checksum
validation. Scientific definitions, payload formats, resources and source-pin
requirements are unchanged; no schema version bump or donor migration.

Added `tests/test_jetclass2_delphes_offline_aux_storage.py` and clarified the
storage behavior in the auxiliary reusable contract. Pre-change existing suite:
21 passed (89.55s). New race regressions reproduced the exact stat traceback
before the fix. Post-change auxiliary suite with the first 16 storage cases:
37 passed (83.35s). Final expanded storage suite: 19 passed (1.21s), including
real publisher link/unlink windows, live temporary accounting, recovered-final
deduplication, aborted writes, claim cleanup, missing durable files, symlinks,
permissions and size limits. Scoped whitespace checks passed.

No commit, push, deletion, cancellation, submission, or new installed-Weaver/A100
acceptance was performed here. Preserve the current study and its completed
receipts; the failed shard's existing files alone do not authorize reuse. The
implemented recovery is same-source only. Next: publish this scoped fix and
use a fresh reviewed source-pinned study/GATE/PREPARE, not an in-place edit of
the old immutable spec/worktree. Retire the old blocked jobs only with exact
study-ledger checks and user direction. PREPARE's genuine A100 profile is still
required before any scientific submission. Shared unrelated edits remain intact.

## 2026-09-12: auxiliary GATE JSON-path boundary fix

User-supplied job 21624032 failed in the first auxiliary GATE at source
`c1525a8317c28b2eb9d4158758fc6934adf8b277`: the JSON study's string `data_root`
reached the native Path-only verifier, causing `str / str` before the first
ROOT content check. This was not a reported resource or data-integrity failure.
The failed root is `jc2_offline_aux_500k_c1525a83_r1`; preserve it for evidence.

Fixed only the auxiliary `build_roles` and `read_rows` entry boundaries to
convert the root to `Path`. The latter also covers target/cache process workers.
Native path containment, checksums, row capabilities, splits, targets and all
scientific settings are unchanged; no contract version bump or new donor code
is required. Shared native Delphes code and other campaigns are untouched.

Pre-change auxiliary tests: 18 passed (33.60s). Three new JSON-round-trip
regressions reproduced the exact TypeError before the fix. Afterward, all 21
auxiliary tests passed (59.76s), including identical role manifests, particle
rows, target payload hashes and RAM-cache arrays for Path versus JSON-string
roots with one/two workers, plus sealed-role refusal. Five existing native
path-safety/relocation/corruption tests passed (8.31s); scoped diff checks passed.
Changed files: auxiliary roles.py, its pipeline tests, and this handoff entry.

No commit, push, remote job change or new real-Weaver/A100 acceptance was done.
Next: push the scoped code/test fix, create a fresh auxiliary study/worktree
pinned to that commit, dry-run and submit its GATE using the original read-only
readiness metadata. Do not repin/edit the failed immutable spec or use the
same-source recovery command with changed code. PREPARE/scientific gates remain.

## 2026-09-12: explicit cross-experiment split consistency and scaling order

The user clarified that setups should progress from 500k training jets to 1M,
then 1.5M, then 2M, keeping the same exact 1M validation and 1M sealed test jets
throughout. Expanded section 6.1 of the
[SPORC/new-dataset guide](JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md) to make this
a cross-experiment rule, not just a within-ladder convention. Every experiment
at a given size reuses the identical training membership; larger sizes extend
the nested sets. Different methods, campaigns and training seeds must not
redraw splits. The guide specifies membership-hash/row-identity checks and
clearly labelled, shared reporting masks for already registered internal
validation subdivisions. The quick-start, final checklist and copy-paste
briefing now repeat this requirement.

Only the guide and this status note changed in this follow-up. No split
artifacts, code, jobs or dataset bytes were changed; no donor code or contract
version change was needed. Later sizes remain separately gated/authorized,
not automatically submitted. Scoped validation passed 116 local links, code
fences, whitespace and section numbering across the two documents. Scaffold
tests remained at 3 passed and the same pre-existing scratch-link failure
(four unrelated broken links; post-edit run 1.34 seconds). Git whitespace
checks passed. No new real-GPU validation was needed or performed for this
documentation clarification; the existing RC evidence remains as recorded.

## 2026-09-12: detailed new-dataset/SPORC guide for other chats

Added [JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md](JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md)
and linked it from README. It explains the SPORC/x86-64 environment, shared raw
snapshot, eleven-class/reduced-input schema, zero-error assumptions, exact
nested split registry, U/D matching semantics, code entry points, pinned
submission/recovery, progress/metrics, storage and evidence boundaries.

The user supplied successful debug-profile job 21619139 evidence (1:04:50,
miniature/full-population passed, A100-PCIE-40GB, 8 CPUs/workers, 72 GiB, no
rolling resume or final-test access), a passed 59-task TRAIN_500K dry run,
and the subsequent live science queue. Its evidenced root is
`jc2_sporc_four_spine_500k_82032e35_r1`, using source
`82032e35177f83436741d7fa1b9d38fbc4b3efc7`; planned train/reduce walltimes were
808/43 minutes. This supersedes older pending-evidence status, not the
scientific plan or gates. These are user-provided RC observations, not new
remote verification by this documentation task.

The latest user request was to cancel ULTRADENSE only and retain DIRECT,
COARSE and DENSE. A command was supplied but no execution confirmation was
provided. The guide explicitly preserves that uncertainty, warns that the
global aggregate/complete will block if a registered branch is abandoned,
and that generic whole-graph recovery could resurrect intentionally cancelled
tasks. It does not certify the separate offline-auxiliary study's GPU gate.

Documentation-only change: no donor code migrated, contracts added/versioned,
jobs submitted/changed, data read for science, or source committed/pushed.
Unrelated local work is preserved. Before editing, scaffold tests passed 3/4;
the global Markdown-link test failed on four existing links in an unrelated
scratch source snapshot. Post-edit scaffold outcome is unchanged: 3 passed,
the same one test failed on those same four scratch links (1.26 seconds).
Scoped checks passed 207 repository-relative links across the four touched
documents, balanced code fences, all six Bash examples, the Python example,
the PowerShell example, and new-guide whitespace. Production create/submit
CLI help was checked against the documented options; scoped git whitespace
checks passed. No scientific/GPU test was rerun for this documentation change.
Next operational task: inspect the current canonical ledger and branch-
cancellation state only if requested; do not resubmit the campaign.

## 2026-09-12: offline auxiliary study implemented through staged queue tooling

Implemented the [four-arm 500k plan](plans/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION_500K_PLAN.md)
in isolated `jetclass2_delphes/offline_aux/` modules, a thin CLI and SPORC worker.
The [executable contract](contracts/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION.md)
records the science, new v1 artifact families, commands, resource gates and
recovery. The existing migration and unrelated dirty work are unchanged.
No source was committed/pushed and no remote jobs were submitted or changed.

Implemented: exact native-row capabilities; 28-number offline target shards;
TRAIN-only normalization/pT bins; final-class-vector auxiliary heads; paired RNG,
fixed loss grid, 60–100-pass selected-only training; all-ten configuration and
all-twelve reporting locks; native single-model metrics and paired file-cluster
bootstrap; source-pinned dry/live submission with durable intents; same-source,
exact-stage restart-zero recovery. No matching foundation or external campaign
completion is required. Compact targets and selected weights persist; particle
views, pair matrices and in-progress weights stay in RAM. Generated outputs are
limited to 64 MiB per file / 4 GiB per study with explicit headroom checks.

The concrete gates are GATE (metadata-only split plus bounded TRAIN target
timing), PREPARE (seven target shards, normalization, genuine A100 parity/resource
pass), DISCOVERY (10 fits + lock), CONFIRMATION (12 fits then locked 800k report
and 1,000 paired file draws). Each stage requires separate creation, a full dry
run and explicit authorization; no stage auto-submits science.

Local evidence: 26 pre-change dataset/split tests passed; 18 new focused
auxiliary tests passed, including native synthetic ROOT/one-versus-two-worker
byte equality, HLT-only reads, wrong-role refusal, selected-state restore,
lambda-zero shared-update parity, weighted metrics versus explicit repetitions,
missing/censored bootstrap draws, fresh matched seeds and ambiguous submission/
active-job recovery refusal. A read-only sample of 1,024 real TRAIN rows produced
finite valid targets with zero pair-invalid rows, mass clamps or pT floors;
no report/test particles or generated files were involved in that check.
The broader Delphes regression invocation passed 74 tests in 474.09 seconds
(it included the then-current 11 auxiliary tests; the expanded 18-test auxiliary
suite also passed separately). Python parsing/whitespace checks passed for the
17 new implementation/CLI files; Bash syntax and new contract/plan local-link
and code-fence checks passed. No claim is made that the known unrelated scratch
Markdown-link failure from the prior documentation turn was repaired.

Real installed-Weaver/A100 validation has **not** run here: this Windows Python
environment has no installed Weaver. The implemented PREPARE job is the required
authoritative check; it cannot be replaced by toy tests or the other chat's
matching readiness report. Source commit/push + clean RC worktree and the GATE
submission are the exact next external steps. Scientific fit submission remains
blocked by design until the genuine PREPARE report validates. See LEGACY_SOURCE_MAP
for the unchanged in-repository donors at `82032e35177f83436741d7fa1b9d38fbc4b3efc7`.

## 2026-09-12: standalone offline auxiliary-supervision plan (documentation only)

The user requested a fully specified plan for the separate four-arm Delphes
study in
[JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION_500K_PLAN.md](plans/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION_500K_PLAN.md).
It registers CE, offline composition, offline structure and combined auxiliary
supervision on the existing TRAIN_500K profile: ten discovery fits over three
loss weights, then twelve fits under three fresh paired confirmation seeds.
The plan freezes 28 compact p4/PID-derived targets, head/loss equations, RNG and
schedule, 200k selection/800k held-back internal validation masks, reporting
locks, single-model metrics, and the limitation that event independence within
those validation slices is unproven. Test remains sealed. No matching,
pretrained offline teacher, logit KD, ensemble, dense targets or rolling resume
is part of this study. It defines isolated SPORC/A100 acceptance and measured
resource gates; the other chat's migration and existing jobs are unaffected.

Only this new plan, its plan-index entry and this handoff note were changed by
the documentation task. No donor code was copied and no reusable executable
contract was added/versioned; the proposed artifact families remain design
requirements. Source, targets, scientific fits and real-Weaver/A100 acceptance
for this new study are not implemented by the MD. Next task, if requested:
implement the isolated roles/targets/contracts and focused tests, then the
model/loss and staged execution surfaces before genuine GPU readiness.

Pre-change scaffold check, with local src on PYTHONPATH: three tests passed;
the repository-wide Markdown-link test failed on four existing broken links
inside an unrelated scratch source snapshot. That snapshot was not modified.
Post-write scaffold result is unchanged: three passed, the same link test
failed (1.31 seconds). A scoped audit passed all 104 local links across the
three touched documents, balanced code fences and new-plan whitespace checks;
plan arithmetic and all 432 derived initialization/sampler/per-pass RNG seeds
were checked for consistency/collisions. Scoped `git diff --check` passed.
No commit, push, SSH, data processing or Slurm operation was performed.

## 2026-09-12: profile-only debug continuation for completed Delphes preparation

The user supplied completion/queue evidence for the TRAIN_500K readiness
preparation at `jc2_sporc_ready_500k_275b984d_r1`; profile job 21612446 remained
pending on tier3. The supplied debug partition has A100 resources, permits the
same account/QoS and allows a 24-hour walltime. This is scheduler evidence,
not real model acceptance. The user requested a fix that profiles on debug
without rerunning sample, matching arrays or the completed foundation lock.

Added the isolated `profile_attempt.py`, create/submit/run CLI and thin debug
worker. A fresh PROFILE_ATTEMPT_SPEC/v1 and exact one-job dry ledger reference
the canonical old readiness/hash and completed foundation/lock hashes. Reuse
rechecks array checksums, coverage, semantic producer bytes and the full lock
read-only. The assignment-producing source files are unchanged from 275b984d;
no raw data, matching outputs or old artifacts are copied, edited or deleted.
Default request: one A100, 8 CPUs/workers, 72 GiB, four hours, no requeue.
The separate live authorization phrase is
`AUTHORIZE JETCLASS2 DELPHES DEBUG PROFILE ONLY`. Submission retains exact-ID
dry/intent/receipt protection and never cancels jobs or auto-launches science.

The added `sporc_a100_debug` execution site does not change either original
site object. RUNTIME_PROFILE/v3 explicitly binds debug as measurement site
and tier3 as production site, with only the named same-A100/environment/
CPU/RAM transfer permitted. All original real-Weaver, full-population,
capacity/batch, no-resume and sealed-test checks still run. Scientific commands
remain on tier3 and enforce the actual measured GPU/environment/resources.
Old v2 profiles remain same-site only. PROFILE_ATTEMPT_RESULT/v1 links the
completed profile to its attempt, foundation lock and actual Slurm job.
No new donor code was copied; existing repository helper attribution remains
unchanged. The active plan and reusable contract document this narrow exception.

Local pre-change SPORC tests passed 14/14 in 54.42 seconds. Initial debug
tests exposed a test-only scheduler mock intercepting Git provenance calls;
the fixture now captures real local producer hashes before replacing the
scheduler. The first corrected debug suite passed 11/11 in 137.62 seconds.
The clean-HEAD snapshot with only the scoped Delphes code/test changes passed
all six Delphes suites: 63/63 in 446.39 seconds, including the additional
missing-dry-ledger/post-creation-corruption regression. The new worker, shared
helper and documented RC queue block pass Bash syntax checks; all 39 Delphes
Python surfaces parse via AST. Both original site objects/hashes are unchanged.
No installed-Weaver/A100 acceptance is claimed locally.

Next: commit/push only the scoped Delphes changes, create a new clean RC
worktree, create the profile-only dry run against the existing foundation,
and separately submit its single debug job. If replacing the old pending job,
cancel only the original profile ID verified from its live ledger; leave all
completed predecessors and unrelated project jobs untouched. The copy/paste
procedure is in `docs/JETCLASS2_DELPHES_SPORC_READINESS.md`. No remote job or
artifact was changed during implementation.

## 2026-09-12: SPORC A100 migration readiness implemented locally

The user moved the new Delphes benchmark to SPORC `tier3`, account
`reu-aisocial`, QoS `qos_tier3`, one A100, and the isolated x86-64 prefix
`/home/ryreu/miniconda3/envs/atlas_kd_sporc`. The supplied installation log
passed dependency/import/CPU checks with Torch 2.5.1+cu118, NumPy 2.2.6 and
Weaver 0.5.3. It is NOT real GPU acceptance. The active migration plan explicitly
substitutes a genuine A100 gate for historical Tigris/GH200 requirements in
this new namespace only. Existing jobs and old worker defaults are unchanged.

Added `execution.py` and `readiness.py`, thin
`prepare_jetclass2_delphes_sporc.py`, an isolated Conda helper and scoped
PowerShell staging helper. Production profile/creator/submitter/workers now
bind the site, real allocation, exact measured GPU, Python/Weaver/numerical
environment, CPU/RAM and workers. New contracts: EXECUTION_SITE/v1,
READINESS_SPEC/v1, RESOURCE_MEASUREMENTS/v1; updated INSTALLED_ENVIRONMENT/v2,
RUNTIME_PROFILE/v2 and CAMPAIGN_SPEC/v3. Older runtime/spec identities are not
silently reinterpreted. No loss, feature, batch, schedule, view or split change.

Cache budgets now divide 75% of host RAM in proportion to each selected role's
conservative resident/worker/IPC bound, keeping 25% reserved. The old 53% train /
22% validation allocation was inappropriate for 500k train / 1M validation.
Native profile metadata at capacity 240 and eight workers yields upper bounds
12.48 GiB train and 34.78 GiB validation, requiring at least 63.01 GiB total
allocation under this rule. The readiness job starts at 80 GiB, 8 CPUs/workers,
one A100 and four hours; these remain UNMEASURED starting requests. It performs
real Weaver parity, bounded CE/offline/KD checks, a full selected-population
training/validation pass, capacity-240/batch-256 backward, reducer inference and
U000/U050/D050 full cache timings. Failed time estimates leave diagnostics, not
a usable profile. Production walltimes are computed with explicit headroom and
an initial 48-hour ceiling, never silently clamped. Reducer minimum is 30 minutes.

The readiness-only queue has sample -> assignment array -> lock -> profile,
not 31 scientific fits. Assignments request one CPU/4 GiB/two hours per element,
at most 16 concurrently; sample/lock use one CPU/4 GiB/30 minutes. Canonical dry
run, a separate readiness authorization phrase, raw checksum/storage checks,
exact-ID receipts, ambiguous-acknowledgement refusal and `--no-requeue` are
included. There is no auto-launch, cancellation, hold or reprioritization.
The subsequent 59-job scientific queue still requires the real profile and
its separate dry run/authorization. Durable artifacts remain compact assignments,
probabilities, selected weights and reports; all views/optimizer state stay in
RAM and rolling resumes remain disabled.

The frozen split registry and all four profiles re-authenticated locally with
unchanged hash `72e8b76555e3707e90aa7ea46d521c93a2b801e46ee94c35ecee978e324d7d01`;
their five-file total remains 15,124,579 bytes. All data memberships are
unchanged. The producer's unchanged-label reply is recorded as additional
supporting evidence in the plan without editing old immutable provisional
policies. Charged zero-error interpretation remains provisional and unchanged.

Local evidence: pre-change Delphes suites 36/36 in 769.73 s; focused SPORC plus
production suites 19/19 in 540.88 s (scheduler/CUDA mocks explicitly synthetic).
A clean-HEAD-plus-Delphes validation snapshot passed all 50 Delphes tests and
three scaffold checks in a 959.04-second run. The remaining scaffold link test
found two pre-existing HEAD references (README and HANDOFF) to the untracked
`docs/HCWDL_MHPE_TRI60_STRATEGY_EXPLAINER.md`; this unrelated unpublished file
is deliberately not swept into the deployment. The final 80-GiB default also
passed its focused submission/RAM regressions (2/2 in 60.82 s).
Three Bash files pass syntax checks. The staging helper excludes pre-existing
dirty Scouting/salience files, the salience HANDOFF section, and mixed
README/plan-index changes while preserving all working copies. No remote
acceptance is claimed.
Donor helper reuse remains at `fd1ed1d01d54bf2ad4d42ffa6311432263a14770`, as
recorded in LEGACY_SOURCE_MAP; no old scientific donor was edited.

Next external gate: transfer only the compact profile bundle, create a clean
pushed-source SPORC worktree, inspect and submit the readiness-only plan, then
return actual assignment/Weaver/GPU/runtime evidence. Raw data are already on
shared RC storage; no raw reupload is needed. No SSH, SCP, Slurm submission or
remote job mutation was performed during local implementation. Commands are
also preserved in `docs/JETCLASS2_DELPHES_SPORC_READINESS.md`.

## 2026-09-11: fixed-evaluation, nested training-size registry implemented

The user replaced the initial full-data migration run with TRAIN_500K and
requested reusable TRAIN_500K / TRAIN_1M / TRAIN_1P5M / TRAIN_2M profiles.
All four have identical 1,000,000 validation and 1,000,000 sealed test jets;
training populations are exactly 500,000 / 1,000,000 / 1,500,000 / 2,000,000
and nested. The active migration plan section 6 supersedes earlier full-data
defaults. Whole-file reservoir roles, raw source selection, features, matcher,
optimizer and no-resume semantics are unchanged. Each size trains its own
fresh HLT baseline/offline teacher; no full-data teacher is imported.

Added `src/hlt_classification/jetclass2_delphes/split_registry.py`,
`scripts/create_jetclass2_delphes_split_registry.py` (build/inspect/verify),
and `tests/test_jetclass2_delphes_split_registry.py`. Updated contracts,
splits/reader, foundation, acceptance sampling, campaign/production, both
foundation/result CLIs, production tests and migration documentation. Masks
select exact latest-cycle entry indices, stored as compact base64 packed bits
inside immutable JSON. SHA256 classwise ordering and incremental integer
Hamilton quotas give reproducible class-stratified nested populations.
No particle arrays or test predictions are needed to compile membership.

New contracts: SPLIT_DESIGN/ROLE_MEMBERSHIP/SPLIT_REGISTRY/SPLIT_PROFILE v1;
subset FOUNDATION_SPEC/CAMPAIGN_PLAN/CAMPAIGN_SPEC v2. The old SPLITS/v1 and
FOUNDATION_SPEC/v1 retain their original meaning. New production creation
refuses legacy full-population foundations instead of silently falling back.
Every matching task, RAM reader/cache, bank and resource estimate uses the
selected profile, including ordinary-role files with zero selected rows.
The graph is still 31 fresh fits and 26 teacher publications per profile;
no ensemble, optimization or deployment-input change was introduced.

Actual local bundle:
`artifacts/jetclass2_delphes_scaling_splits_20260911_v1/registry.json` plus
four self-contained files under its `profiles/` directory. Total persisted
size: 15,124,579 bytes (about 14.42 MiB), not dense data or copied ROOT files.
Registry content hash:
`72e8b76555e3707e90aa7ea46d521c93a2b801e46ee94c35ecee978e324d7d01`.
Shared validation membership:
`8f04f54797f7cd181812815eb409b90cb1cf7e503c322061e4718d23f06060b5`.
Shared final-test membership:
`995f747b09a47c2e7cbdc996ba552542562ef0e074b3cb7a9de2b87f7438106e`.
Compilation checked checksum/cycle/schema and label/matched scalar counts
against the immutable 333-file snapshot. Exported profiles validate against
the registry and exact nesting/evaluation identity checks pass. The original
100,000-row-chunk compilation was independently replayed with 32,768-row
chunks over every source file: **METADATA REPLAY: PASS**, with identical
design and all six memberships. All 32 Delphes Python/CLI/test files pass AST
parsing; tracked and new-file whitespace checks pass.
The original
inventory and reservoir manifests retain their uploaded byte SHA256 values
`0425f355ba020eb49b4179a0110cd6ff644030a1c7640cf34cbc98aa10979adf` and
`4b184543d34620e28e6d884fa054d7c53f560b69131e65e8cd435ef696e1d41c`.

Evidence: baseline Delphes suites 28/28 before edits; updated four-file suites
35/35 in 224.46 s; an additional reader capability/ineligible-row regression
plus the four scaffold/link tests passed 5/5 in 16.26 s (40 distinct tests in
total). Coverage includes exact/nested quotas, shared evaluation identities,
relocation/chunk invariance, metadata-only test handling, invalid membership,
capacity failure, canonical mask padding, corruption/role escape, ordinary
reader counts, two-process RAM caching, assignment cross-profile rejection,
versioned graph/production gates and synthetic full-DAG publication/recovery.
Remote/source gates are mocked ONLY in disposable orchestration tests; these
are not real Weaver/GH200 evidence. Donor hash/publication helper reuse and
the preceding uncommitted local migration are recorded in LEGACY_SOURCE_MAP.

The raw dataset is already uploaded and checksum-verified on RC according to
the user's supplied output. Only the new profile metadata needs transferring.
Explicitly deferred: commit/push, clean pinned RC worktree, chosen-profile
matching foundation, genuine Weaver/GH200 acceptance/resources and any live
submission. Nothing was committed, pushed, uploaded, queued, cancelled or
changed remotely in this step. No real matching foundation or fit was run;
final-test particle/model access remains sealed. Unrelated dirty work remains.

## 2026-09-11: Delphes migration and gated production workflow implemented

Implemented the isolated `jetclass2_delphes` package and six thin commands:
inventory/remote verification, foundation-spec creation, sample/assignment/lock
execution, genuine-Weaver acceptance, non-executable four-spine preview, and
source-pinned production preparation/profile/create/run/submit/monitor/recover/
results. Two absolute-path Slurm workers cover preparation and campaign jobs.
The [v1 contract](contracts/JETCLASS2_DELPHES.md) and updated
[active plan](plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
record the concrete 11-class selection, provisional labels/zero-error handling,
whole-file splits, 17-feature inputs and exact full-cardinality U/D semantics.
Other modules supply ragged RAM-only process preparation, class-aware metrics,
immutable identity-joined probability shards, a new 17/11 model wrapper and
CE/KD training kernel. Production binds 31 fresh fits, 26 teacher publications,
aggregate and completion into 59 exact jobs. It authenticates installed model
source/numerical versions, publishes only selected weights, checks real runtime
evidence, requires an exact dry run and explicit live authorization, and refuses
overlapping/ambiguous recovery submissions. `pyproject.toml` declares a separate
Delphes optional dependency extra. No existing campaign source, data, jobs or weights were
modified. The pre-existing dirty worktree was preserved; nothing was committed,
pushed, uploaded or submitted by this task.

Authoritative local inventory:
`artifacts/jetclass2_delphes_20260910_provenance_v1/inventory.json`, hash
`91b8a67191eb8f5f3d833bd3347211eafd12f9939263c7dabf89c8159b02603a`.
All 333 files were hashed and their latest ROOT cycles audited: 25,430,379,343
bytes, 12,216,863 stored rows, 8,996,860 selected rows under default source policy.
Frozen split counts: train 5,376,107; validation 1,810,176; final test 1,810,577.
Every class occurs in every role. File/content disjointness is checked;
generator-event independence across files remains explicitly provisional.

Foundation intent:
`artifacts/jetclass2_delphes_foundation_provenance_v1/foundation_spec.json`, hash
`e77536097223d9fc1e4796064a5a6e9c04c286f1bf76bfed1639379cb6f53eee`.
It registers 259 train/validation file tasks, capacity 240 without truncation,
and no final-test particle access. A bounded native audit checked 2,064 selected
jets from those files (one file has no selected rows), 50,682 HLT and 87,956
offline particles, and 50,513 exact smaller-side pairs. Both endpoint equalities
and intermediate finite inputs passed; eight exhaustive references use native
prefixes capped at four per side, not an unbounded permutation audit. The
sample report hash is
`fc9e30ff554db66c36dc078ea62540a0a0fded7f5fe8cc624ecb4295c2f6ee54`.
Nonzero displacements with zero error are retained (four HLT and eight offline
individual value/error cases in this sample). Initial provenance-free local
preview artifacts were retained but are not eligible for current validation.

Local focused suite: **53 tests passed in 122.42 seconds**, across the three new
test files, scaffold, original input tests and exact bottleneck solver regressions.
The production suite was subsequently expanded to traverse all 59 tasks and
rerun after final source-snapshot verification wiring: **5 passed in 98.82 seconds**.
All 29 new Python files parse; both Slurm wrappers pass Bash syntax checking.
Tracked whitespace checks pass (only normal Windows CRLF warnings). Coverage includes
synthetic ROOT cycles, relocation/corruption, split leakage, dummy rows, jagged
mismatches, zero errors, endpoint isolation, rectangular exhaustive parity,
one-/two-process cache equality, compact publication, KD equation, probability
joins, metric censoring, selected-weight restoration without resume files,
source/allocation gates, exact dry/live dependency plans, full synthetic DAG
publication, completed-task reuse, live/unknown-job recovery refusal, and a lost
sbatch-acknowledgement failure that cannot trigger blind duplicate submission.
Scheduler calls and external execution gates are mocked in tests. Kernel and
full-DAG tests use an explicitly labelled tiny model, not Weaver. The local
scientific Python has PyTorch/CUDA but no installed Weaver; no genuine Weaver
or Tigris acceptance has been claimed. Donors and dirty transitive dependency
hashes are recorded in `LEGACY_SOURCE_MAP.md`.

Next deployment stage: commit/push the isolated changes, transfer the frozen raw
snapshot and manifests to a NEW RC location, verify all remote files, execute
source-pinned compact preparation, and run the provided genuine Weaver/GH200
acceptance plus full-population resource probe. Then create/audit the exact
production dry run and obtain live submission authorization. The preview CLI
remains non-executable; the separate production wrapper enforces these gates.
Same-source recovery is implemented; code-changing recoveries deliberately need
a new explicit lineage transition, not a source hot-patch. Real final-test
evaluation is not provided by this validation-only campaign.

The authoritative local inventory/split and foundation/sample/preview files
total about 1.52 MB. No full-size particle cache, representation target or
rolling resume was created. Full production is not zero-storage: projected
probability payload is approximately 13.2 GiB, plus compact matching arrays,
selected checkpoints and reports. Failed compact attempts are retained for
inspection; no automatic broad cleanup exists. Label IDs, zero-error semantics
and cross-file generator independence remain explicitly provisional as authorized.

## 2026-09-11: provisional Delphes label and zero-error policies accepted

The user authorized proceeding while Luka's confirmation is pending. Section
3.1 of the [migration plan](plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
now records unchanged upstream JetClass2 numeric labels and zero impact-parameter
errors as unavailable uncertainties, including for charged particles. Finite
raw displacement values are preserved; no division by zero, epsilon-derived
significance, or particle deletion is authorized. Unknown labels, negative
errors, and nonfinite required data still fail validation.

Whole-file split groups carry an explicit cross-file event-independence
assumption, reduced input fields are accepted, and exact producer cards/revision
can be gathered alongside implementation. Any later correction requires new
affected artifacts and reassessment rather than editing existing results.
Provisional and producer-confirmed evidence remain distinct. No adapter code,
contract implementation, training, remote submission, or donor-code migration
occurred in this documentation update. Scaffold/link checks pass 4/4 both
before and after the change; the tracked documentation whitespace check passes.

## 2026-09-11: JetClass2 Delphes migration plan and initial local audit

The [migration plan](plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md)
documents the downloaded September 10 paired Delphes production: 333 ROOT files,
25,430,379,343 bytes, 12,216,863 latest-tree rows, and 11,443,876 matched HLT rows.
All files opened with one branch/type schema; checked matched-row scalar fields
were finite. A 20-file/4,836-matched-row constituent sample had consistent
lengths and finite exclusive particle identities. This is preliminary local QA,
not remote checksum verification or an exhaustive constituent validation.

The plan records multiple saved ROOT tree cycles, zero impact-parameter errors,
absent event identifiers, and unavailable legacy detector-quality inputs. The
old minimum-16 setting is correctly identified as a padding floor, not a
selection cut. Producer code/cards, integer-label mapping, units/sentinels,
selection/weight semantics, and generation-group provenance remain to be pinned.
Proposed choices include 11 classes, a common JetClass-style feature family,
grouped splits, and fresh HLT/offline references. None is represented as an
already frozen executable campaign. Existing FullSim artifacts and unrelated
salience/fusion work are not modified by this migration design.

This step changes documentation only; no donor code or contract implementation
was copied, no training or Tigris action occurred, and no final-test model output
was produced. The pre- and post-edit scaffold/link suites pass 4/4 with the repository
`src` on `PYTHONPATH`. Stage A's reproducible inventory/audit and synthetic ROOT
fixtures are the next implementation task; production readiness remains pending.

## 2026-09-11: full-cardinality salience-matching implementation

The reusable scientific and integration guide is
`docs/HCWDL_FULLCARD_SALIENCE_MATCHING_CODE_GUIDE.md`. It records the exact
objective and tie hierarchy, persistent-HLT endpoint semantics, code and
artifact interfaces, safe-reuse boundary, staged candidate-selection flow,
storage guarantees, and a future-campaign adoption checklist. The guide does
not alter the implementation-authoritative plan or any scientific contract.

The active scientific plan at
`docs/plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_SALIENCE_MATCHING_IMPLEMENTATION_PLAN.md`
freezes a new isolated forced-alignment control. It retains exact smaller-side
coverage and persistent-HLT support while replacing lexicographic worst-edge
optimization with an integer-exact bounded angular-closeness utility weighted
by particle pT salience and, in one preregistered candidate, a mild at-most-25%
jet-core bonus. Three fixed candidates feed a four-fit CE-only U100 endpoint
screen with a deterministic held-out validation selection partition. The
selected salience matcher then feeds a fresh 30-fit, 26-reducer four-spine
campaign under new contracts and roots. Bottleneck is a contextual screen
control and all three salience candidates remain reportable regardless of
performance.

The plan is now implemented under the separate
`HCWDL_FULLCARD_SALIENCE_* /v1` and
`HCWDL_TRI100_FOUR_SPINE_SALIENCE_PERSISTENT_HLT_* /v1` contract families.
The integer-exact production matcher has an independent exhaustive reference,
complete smaller-side coverage in either cardinality orientation, three frozen
salience candidates, deterministic tie-breaking, compact assignment artifacts,
all-row pT/geometry/category/charge diagnostics, and sampled recomputation.
Existing particle decoding exposes raw category-flag multiplicity only as
optional matching diagnostics; ordinary classifier inputs and old matcher
semantics are unchanged.

Three isolated candidate-foundation DAGs rebuild every assignment-dependent
descendant while reusing the pure-offline U000 artifacts read-only. A separate
four-fit, CE-only U100 screen compares bottleneck context with all three
salience candidates under matched seeds and a deterministic validation
firewall. Its immutable selection lock binds the one-shot screen report,
winner foundation, matcher specification, and frozen multiplicity. Only that
authenticated winner can create the separate 61-task production DAG: one
fresh persistent-HLT U000 anchor, 29 immediate-parent C25/P75 temperature-2 KD
fits, and 26 single-component reducers over the unchanged four-spine geometry.

The new production path is single-GH200 and effective batch 256, uses the
registered 100-pass floor-tail schedule with minimum 60/patience 15/best
restore, and has no ensembles, M1, DDP, cross-spine teachers, or dependency on
running campaigns. Dense matching matrices and particle views stay in RAM;
only compact assignment indices, ordinary checkpoints/reports, and 15-class
probability banks are durable. Optimizer and rolling-resume state are absent,
recovery is exact-ledger restart-from-zero, and final test is unavailable.

Local verification is 22/22 for the new contract/matcher/screen/production
surface and 68/68 when combined with the old bottleneck and matching-repair
neighbors. Repository-wide verification is 884/884 passing after explicitly
deselecting one unrelated installed-Weaver test; the local `tagging-hlt`
environment does not contain Weaver for
`test_fusion_primary_uses_matched_seed_but_context_has_separate_seed`.
Python syntax compilation passes across every new and directly modified
module. No SSH, Slurm submission, remote mutation, cancellation, hold, or
reprioritization occurred. Installed-Weaver parity, genuine Tigris
matcher/resource acceptance, the four-fit selection itself, exact pushed
source, and explicit live-submission authorization remain remote gates rather
than claimed local evidence.

## 2026-09-02: Strategy-B adjacent learned-fusion handoff implementation

Strategy B in
`docs/plans/HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md` is implemented
as the isolated `HCWDL_ADJACENT_LEARNED_FUSION_HANDOFF_*/v1` family. Its graph
owns exactly 25 fresh fits: five matched direct-KD controls, five cold
two-view acquisitions, five exact-withdrawal students, first-rung and terminal
low/low fusion, warm-continuation, and parameter-matched controls, a cold
terminal CE model, static and dynamic U100-to-D000 fusion controls, and one
companion morph-withdrawal fit. Every learned rung ends in a physically
extracted ordinary single-view checkpoint before the next cold rung begins.
The dynamic morph can select only from pass 51 onward, after its context has
reached D000, so an earlier privileged-view checkpoint cannot be restored.

The implementation provides immutable source/control adapters, explicit
population and seed locks, graph and recipe artifacts, deterministic three-way
validation partitioning, RAM-only
paired-view and one-coordinate-at-a-time morph caches, compact probability
banks, mandatory both/zero/permuted/alpha-zero diagnostics, exact task
attestations, staged exact-ID submission, ledger monitoring, and restart-zero
source-pinned recovery. Particle views, hidden states, and optimizer state are
not durable; rolling resume and final-test access are disabled. The source is
the same completed non-persistent full-cardinality U100 artifact authenticated
by Strategy A, and no running or existing campaign is mutated.

The science submitter fails closed unless the output bytes and immutable task
attestations of authentication, validation partitioning, capacity/storage
audit, and installed-Weaver GH200 preflight all revalidate. A lone preflight
file cannot authorize the dependency-free science stage.

The lower primary and direct control reuse Strategy A's exact rung seed aliases
(terminal `S1` at D000), while context/cross-attention parameters use a
separate graph-bound seed domain. The 100-pass schedule and early-stopping
minimum AUC delta are likewise identical to Strategy A.
Only the 11 distributions with downstream training consumers retain all four
probability roles; 15 nonteachers retain `V_report` only, giving a conservative
16-GiB total durable-output bound. The untouched-report aggregate includes
2,000-replicate paired macro-AUC intervals for all five adjacent carrier
transitions, all five learned-carrier-minus-direct controls, and all eight
preregistered causal contrasts, plus per-rung context/withdrawal decomposition.

Sixteen focused Strategy-B tests cover the graph, exact imported-U100 seed
lineage, independent architecture seed
domain, exact morph schedule, paired-view identity, low/low equality,
the shared assignment under a distinct Strategy-B partition contract,
consumer-bound and report-only probability retention, complete recovery
reporting, the fixed five-point fusion-strength diagnostic, staged publication,
restart-zero recovery. Stored validation-partition semantics, assignments,
counts, and exact dtypes are all revalidated before reuse. All new Python
surfaces pass `compileall`, and both Slurm workers pass `bash -n`. The local
`tagging-hlt` environment passes 36/37 focused regressions across Strategy B,
the reusable fusion implementation, and the established TRI100 trainer. The
sole remaining case is the intentional installed-Weaver model-construction
test; local Weaver is absent, so it must pass in `atlas_kd_tigris` as part of
the production gate. The complete local repository suite likewise passes
862 tests and reaches only that same remote-only Weaver case; no other
regression fails. Before science can be submitted, pin and push the exact
source, execute the gate through its real installed-Weaver GH200 preflight,
inspect measured RAM/GPU headroom, and materialize the science dry run. No
Tigris task has been submitted and final test was not accessed.

## 2026-09-02: unified offline/HLT fusion and withdrawal campaign

The remaining experiments in
`docs/plans/HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md`
are implemented as one isolated eleven-fit campaign. The eight 60-pass
oracle/control rows are `CONCAT_UNTAGGED`, `CONCAT_TAGGED`, the matched
symmetric `OO`/`HH`/`OH` fusion controls, `HLT_WARM_CONTINUE`, and the matched
anchored `HH`/`OH` controls. `ANCHORED_FUSION_OH` is the fixed teacher
regardless of its validation score. Its compact identity-keyed probability
bank feeds a matched warm HLT-only direct-KD row and two 100-pass anchored
withdrawal rows (cosine and step). Both withdrawal rows validate and select
only at exact alpha zero and publish ordinary three-input HLT checkpoints
after extraction parity and offline-perturbation audits.

The campaign contains 19 exact tasks: three gates followed by sixteen science
tasks. All particle views, cross-attention states, and withdrawal
representations are regenerated in RAM/device memory and are never durable;
only compact 15-class teacher probabilities and ordinary reports/checkpoints
persist. Rolling resume and optimizer-state publication are disabled, and
recovery restarts incomplete tasks from update zero in a new source-pinned
worktree. Scientific metrics do not gate registered rows or completion, final
test is inaccessible, and the campaign has no scheduler or artifact
dependency on any running campaign.

Focused local coverage currently passes 67/67 across the new campaign,
tagged-concatenation neighbor, and established TRI60 trainer. This includes a
complete synthetic 100-pass withdrawal execution through the alpha-zero tail,
architecture/gradient controls, teacher-bank identity joins, exact endpoint
extraction, staged DAG publication, and restart-zero recovery. The only local
warnings are the pre-existing PRAD tensor-conversion warning and pytest-cache
permission warning. The complete repository suite passes 846/846. No legacy
donor file was copied. Remaining production requirements are exact pushed-source
pinning, the three-task Tigris gate (all-row capacity audit plus installed-
Weaver GH200 production-batch preflight), inspection of measured RAM/GPU
headroom, and a full nonmutating science dry run before live submission. No
Tigris job has been submitted for this campaign and final test was not
accessed.

The all-row capacity audit records separate offline, raw-HLT, and combined
maximum identities for both train and validation. The GH200 gate assembles a
single production-size batch containing every distinct extremum, so the
internal `2*O`, `2*H`, and `O+H` attention widths are all exercised before
science submission. The final focused capacity/fusion regression passes
18/18 after this hardening.

## 2026-09-02: adjacent output-fusion handoff implementation

Strategy A of
`docs/plans/HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md` is now
implemented as the isolated
`HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF_*/v2` family. The immutable 85-task DAG
owns 26 fresh fits, 9 performance-constrained mixture selections, 15 fixed
lexical prefix ensembles, compact source/model reducers, an untouched-report
aggregate, exact-ID submission/monitoring, and restart-zero failed-closure
recovery. It imports one explicit completed full-cardinality U100 report and
checkpoint without depending on or mutating its source campaign. The source is
now explicitly the completed non-persistent full-cardinality
`SP4_COARSE_U100_from_U050` from Tigris job `98318`, using
`replace_source_with_target_v1`. The initially pushed `/v1` adapter incorrectly
bound the unfinished persistent-HLT `SP4P` family; no `/v1` campaign root or
science job was created, and `/v2` makes the corrected meaning
non-interchangeable.

Validation is deterministically class-stratified into disjoint checkpoint,
blend, and report roles. The selector evaluates all 82 combinations of two
registered fusion families and 41 alpha values, uses 2,000 paired
class-stratified Gaussian-multiplier AUC-influence bootstrap replicates,
enforces macro-AUC and Xbb/Xcc/Xqq R50
noninferiority, and retains alpha zero as a graph-preserving fallback. Train
banks persist one canonical T=1 model-probability table per role and derive
the registered T=2 or T=1 teacher probabilities only in RAM at identity join,
preventing accidental double softening while nearly halving storage. All particle
views remain RAM-only; no hidden states, optimizer states, rolling resumes, or
final-test artifacts are written.

Before the source-identity correction, the focused implementation suite passed
8/8 locally; the campaign plus established TRI100, endpoint-mix, and TRI60
blend regressions passed 35/35, and the complete repository suite passed
833/833 with only the pre-existing PRAD tensor-conversion warning and the
local pytest-cache permission warning. The first corrected source gate passed
9/9 on Tigris and then safely stopped before campaign creation because the
completed job-98318 artifact is the upstream full-cardinality `/v3` family,
not its earlier `/v2` predecessor. The compatibility adapter now reconstructs
and authenticates the exact `/v3` graph, recipe, report, checkpoint, and both
unclassified-particle endpoint policies. Its focused suite passes 9/9 locally;
exact real-artifact authentication must be rerun in the new pinned Tigris
worktree before campaign creation. A genuine Tigris execution acceptance remains required
before live science submission; the preflight executes the exact authenticated
U100 stream, installed production ParT, and C25/P75 T=2 backward path. No
Tigris job was submitted and final test was not accessed in this implementation
turn. Historical same-repository semantics migrated from commit `175cbcd6` are
recorded in `docs/LEGACY_SOURCE_MAP.md`; no external donor runtime import exists.

Two read-only control reducers reevaluate the imported `M0CE60` and
pure-offline `U000` selected checkpoints on the same untouched `V_report`
identities as every new model. Recovery percentages therefore never mix the
source reports' full-validation population with the campaign's report split.

## 2026-09-02: adjacent-view fusion-handoff ladder design

The implementation-authoritative planning document
`docs/plans/HCWDL_ADJACENT_VIEW_FUSION_HANDOFF_LADDERS_PLAN.md` now freezes two
separate ways to bridge the canonical `U100 -> D080 -> D060 -> D040 -> D020
-> D000` path. The output-fusion strategy trains an adjacent poorer-view
bridge, selects the greatest validation-supported poorer-view weight from
arithmetic-probability and calibrated-centered-logit mixtures, and compresses
that temporary teacher into a fresh single-view carrier. The learned-fusion
strategy makes the poorer view the primary ParT branch, injects one-way
removable richer-view context, withdraws it to exact zero, and extracts a
single-view carrier before the next rung.

The plan freezes a disjoint checkpoint/blend/report validation partition,
paired non-inferiority selection, cold inter-rung initialization, unique seeds
between rungs, paired direct/compression seeds within a rung, RAM-only
particle/hidden state, compact probability persistence, and a final ordinary
D000 endpoint. Strategy A is now the first implementation target: its 26 fits
include the ten-fit main output-handoff chain, four additional terminal
direct/compression pairs, five seed-matched CE D000 controls, and three
identically initialized final students distilled from five-member CE, direct-
KD, and handoff ensembles. Prefix ensembles `E1..E5` use a fixed seed order,
not favorable subset selection. Full multi-seed ladder replication remains a
later confirmation. `D100` is explicitly corrected to the established
coordinate name `U100`.

This entry records the design freeze. The newer implementation entry above
supersedes its former implementation-status sentence; Strategy B remains
planning only. The existing tagged concatenation pilot focused suite still
passes 6/6 locally (one pytest cache-permission warning); final test was not
accessed and no donor file was copied.

## 2026-09-02: tagged offline+HLT concatenation feasibility pilot

Study A now has an isolated one-fit implementation for the requested first
measurement. `CONCAT_TAGGED` consumes every native-order offline particle
followed by every canonical HLT particle, deliberately retains matched
duplicates, and injects a separate learned offline/HLT content-source
embedding after the 21-channel numerical embedding. Padding is masked and has
no source identity. This is a privileged oracle and cannot be deployed.

The campaign is CE-only, matched to the established `U000` initialization,
60 passes, global batch 256, one GH200, and reports against `M0CE60`, pure-
offline `U000`, and `SP4P_U000` with the established recovery convention.
Genuine Tigris gate job `99565` correctly rejected the original `/v1`
400-slot assumption. The exhaustive follow-up measured train/validation
combined maxima of 493/459, 23/14 rows above 400, and a train HLT maximum of
214. The corrected `/v2` identity uses the smallest aligned safe capacity,
496, and explicitly retains raw HLT tokens beyond the ordinary deployable
200-token cap. The failed `/v1` campaign remains immutable historical
evidence. Train and validation views are RAM-only, and durable output is
bounded to compact evidence and model artifacts. The six-task `hcwcat2_` DAG
has capacity and genuine-Weaver
forward/backward gates before its sole fit, exact source/worktree binding,
atomic artifacts, task inventories, exact-ID monitoring, and restart-from-
zero recovery. Submission is split into a three-task gate ledger and a
three-task science ledger; the science submitter requires the validated real
preflight artifact and attestation. It has no dependency on or mutation
authority over any running campaign.

After the `/v2` correction, the focused tagged-concatenation,
full-cardinality, persistent-support, and view-cache suite passes 29/29; the
broader tagged/TRI60/representation/input compatibility suite passes 123/123.
The repository-wide suite passes 834/834 with only the unrelated existing
PyTorch warning and a local pytest-cache permission warning. Genuine Tigris
capacity characterization is complete; the corrected `/v2` capacity gate and
installed-Weaver GPU preflight remain to be executed. No donor file was copied
and final test was not accessed.

## 2026-09-02: offline+HLT oracle and privilege-withdrawal plan

The implementation-authoritative plan in
`docs/plans/HCWDL_OFFLINE_HLT_CONCATENATION_FUSION_WITHDRAWAL_IMPLEMENTATION_PLAN.md`
now separates three questions. Study A compares tagged and untagged
single-stream `offline + HLT` concatenation. Study B now freezes two
unrestricted oracle topologies: a symmetric concatenate-after-local-processing
ceiling, and the primary HLT-anchored model in which an eight-block offline
context encoder supplies one-way cross-attention residuals after HLT blocks 2,
4, 6, and 8. The HLT branch alone owns the class head. The complete first
screen contains eight fresh fits, including same-domain, warm-continuation,
and parameter controls; it contains no dropout, withdrawal, KD, or invariance
objective.

Study C is separately authorized only after the oracle screen. It supersedes
the earlier `F(H,H)` endpoint with an exact removable-residual endpoint:
`alpha=1` is unrestricted `O -> H` fusion, while `alpha=0` skips and extracts
away the offline encoder, cross-pair features, and cross-attention modules so
one ordinary HLT ParT remains. The primary curriculum trains frozen-teacher,
cosine-gated, and exact-zero paths together, uses direct HLT KD and step
withdrawal as mandatory controls, and selects checkpoints only on the
extracted HLT path.

The plan freezes full-cardinality source lineage, all-row no-truncation and
resource audits, explicit initialization maps, zero-residual parity,
source-versus-branch-role embeddings, matched-seed and paired-seed reporting,
sealed final test, RAM-only particle/activation state, compact teacher
probabilities, and restart-from-zero recovery. At the time this planning entry
was written, no implementation existed; the tagged one-fit Study A pilot is
now implemented as recorded in the newer entry above. The paired Study A
screen and all Study B/C fusion and withdrawal work remain planning only.
Focused persistent-support and MT20 tests passed 14/14 before this planning
revision. No donor file was copied.

## TRI100 full-cardinality offline endpoint repair (2026-08-31)

The v2 non-persistent full-cardinality campaign passed preflight, then its
first COARSE, DENSE, and ULTRADENSE U jobs (`98181`, `98190`, `98205`) failed
during train-view preprocessing at row 2991. The DIRECT D000 branch remained
running because it never selects the offline feature endpoint. The common
input is a legitimate offline lost-track record in the native charged
`cpfcandlt` collection with finite-binary but zero-hot particle-type flags.

The validity-only endpoint now binds
`native_collection_applicability_atomic_raw_endpoint_v1`: authenticated native
collection membership determines field applicability, and the raw offline
identity vector is copied unchanged. No charged-hadron or other class is
invented. Exclusive identities must agree with collection membership;
nonfinite, nonbinary, incompatible, and non-discrete required values still
fail closed. Established confidence-backed paths remain strict.

The exact non-persistent repair is maintained on an isolated source-pinned
`965bdad6` worktree with v3 contracts; its focused suite passes 74/74. Current
persistent-HLT-support science binds the same raw-endpoint rule under its own
v2 contracts. Its repair/campaign suite passes 66/66, the adjacent homotopy,
unified-balanced, full-cardinality, and four-spine regression suite passes
160/160, and the repository-wide suite passes 800/800. Matcher and foundation
artifacts remain reusable because no assignment changed. Failed science roots
are not reusable under the new contracts, and the still-running DIRECT job is
not to be cancelled as part of recovery. Final test was not accessed.

## TRI100 persistent-HLT-support four-spine control (2026-08-31)

An independent full-cardinality control now implements the requested monotone
U-support semantics. Every HLT skeleton slot is present from `SP4P_U000`:
matched slots carry their offline endpoints, unavoidable unmatched HLT slots
carry native HLT, and source-only offline tails disappear at the existing
balanced structural switches. U therefore only removes support or leaves it
unchanged; matched substitutions occur once rather than being duplicated as
tails, so U000 has `max(n_offline, n_HLT)` particles. U100 has the HLT count
with matched offline features, while the D path and exact-HLT D000 endpoint are unchanged. The established builder remains
the default, and the new policy is accepted only with neutral pairing-validity
provenance.

The source-pinned campaign has a distinct `hcwsp4p_` job namespace and
`HCWDL_TRI100_FOUR_SPINE_FULLCARD_PERSISTENT_HLT_*/v2` artifacts. It trains a
fresh, seed-matched, 60-pass CE-only hybrid anchor and its own probability bank,
then the unchanged four immediate-parent C25/P75 T=2 spines. The complete DAG
contains 30 fits, 26 reducers, an all-row compact support audit, genuine-GH200
preflight, aggregate, and completion. Pure-offline U000 is retained only as the
shared 100% recovery oracle; it is not a teacher. Existing campaigns have no
Slurm dependency or mutable path in this DAG.

Focused source tests for default/persistent endpoint behavior, metadata order,
monotone cardinality, exact D000, confidence-provenance rejection, raw
zero-hot/multi-hot offline endpoint preservation, nonfinite rejection, hidden-
truncation rejection, graph shape, all-row count arithmetic, campaign DAG,
probability isolation, recovery, and foundation authentication are present.
The focused repair/campaign suite passes 66/66, the adjacent homotopy,
unified-balanced, full-cardinality, and four-spine regression suite passes
160/160, and the repository-wide suite passes 800/800 in the `tagging-hlt`
environment. Python compilation and diff checks pass. A fresh source-pinned v2
genuine Tigris preflight/miniature remains required before full live
submission. Final test was not accessed.

## TRI100 full-cardinality preflight repair (2026-08-31)

The completed `571c0966` full-cardinality matcher foundation remains valid:
its exact assignment objective, compact assignments, diagnostics, coupling,
balanced sidecars, and U000-equivalence lock all completed on Tigris. The
first science campaign authenticated as job `97924`, but genuine-GH200
preflight job `97925` failed before publishing an execution lock at validation
row 1305 with `invalid matched HLT particle identity`. All 56 downstream
science jobs remained dependency-blocked and no fit or target bank started.

The cause was a real boundary mismatch. The complete-bipartite matcher
correctly admits visible particles with nonexclusive raw HLT identity flags;
category is only secondary tie information. The established repair path had
only ever seen category-gated matches and therefore required every matched
HLT identity to be one-hot before deciding charged-field applicability.

The validity-only path now uses the versioned policy
`preserve_until_identity_switch_then_atomic_endpoint_v1`. It never invents a
charged/neutral label: a finite-binary zero-hot or multi-hot matched token
keeps its HLT identity, charge, quality, and track-applicability fields
together until the existing deterministic identity switch fires, then moves
that group to the valid offline endpoint together. Nonfinite and nonbinary
identity values still fail closed. Established confidence-backed paths remain
strict, and valid-token outputs and exact rational switch hashes are unchanged. The
TRI100 full-cardinality science contracts are v2; the failed v1 campaign spec
cannot authenticate as v2, while the completed matcher/foundation v1 lineage
is deliberately reusable because no pairing artifact changed.

Focused matching/repair/full-cardinality tests pass 42/42. The broader
homotopy, unified-balanced, TRI100, and PMARD-recovery regression set passes
121/121 in 152.13 seconds. The repository-wide suite passes 791/791; the final
nonfinite fail-closed boundary was then rechecked in the focused 42-test set.
CLI import/help and diff checks pass. A fresh v2 science root, genuine Tigris
preflight, and exact downstream submission remain; the old pending v1 job IDs
must be cancelled exactly, never by job-name pattern. Final test was not
accessed.

## TRI60 M1 greedy BF16/FP32 reducer repair (2026-08-28)

Tigris inference jobs produced all five authenticated candidate-probability
shards, but reducer job `93675` correctly stopped before selection because its
singleton guard compared stored BF16-autocast training-validation metrics with
the common FP32 shard-inference metrics using one inappropriate `5e-6`
tolerance. For `SOURCE_M1_LOGIT`, the measured FP32-minus-BF16 differences were
`-3.3419e-5` accuracy, `-8.2874e-5` cross entropy, `+5.163e-6` macro AUC, and
`+0.002116070` mean log R50 (`+8.362` linear R50). Shard identities, labels,
checkpoint hashes, and probability hashes all remained valid; peak reducer RSS
was only about 22.99 GB.

The result contract is now v2. The reducer retains exact lineage checks, uses
metric-specific absolute BF16/FP32 reproduction envelopes, records every
singleton's signed drift and pass/fail decision, and still fails closed outside
that envelope. Once reproduction passes, all greedy objectives—including the
source-M1 recovery origin—use the common recomputed FP32 probability regime.
The five existing shards remain reusable. A source-pinned recovery now submits
only `greedy_reduce -> campaign_complete`, records the repair commit in the v2
result lineage, and never schedules inference. Focused greedy tests pass 12/12;
the neighboring greedy/M1-screen/TRI60 set passes 66/66, Python compilation and
diff checks pass, and no remote mutation occurred. Push and the exact two-job
Tigris recovery remain.

## TRI60 D000 optimization-budget screen (2026-08-27)

An additive source-pinned screen now tests whether the original full-data
`LOGIT_D000_from_D033E` edge can capture its apparent long-horizon benefit in
60 or 90 passes. The imported original result is compared with 17 paired fresh
fits: five delayed-decay/floor conditions, four additional peak-LR conditions,
four early stronger-KD prefix conditions, and four 90-pass compromises. Every
fit retains exact HLT D000 inputs, batch 256, temperature-2 D033E targets, the
original seed alias, and a C25P75 endpoint. The 90-pass reports retain pass-60
landmarks so schedule-shape and extra-horizon effects remain distinguishable.

The two-gate plus 17-sibling-fit DAG is nice 10000 and has no scheduler
dependency on or write path into the running TRI60/DX campaigns. Source
probabilities are joined read-only, preprocessing remains RAM-resident, and
only selected/final checkpoints, reports, ledgers, attestations, and logs are
durable. Rolling resume, partial reuse, final-test access, result-controlled
submission, and a standalone smoke campaign are disabled. Exact semantics and
commands are in
[`HCWDL_TRI60_D000_OPTIMIZATION_BUDGET_SCREEN_RUNBOOK.md`](HCWDL_TRI60_D000_OPTIMIZATION_BUDGET_SCREEN_RUNBOOK.md).

The reusable training core gained additive, validated piecewise-constant loss
and warmup/hold/cosine LR schedules while preserving the default TRI60 schedule
when no override is supplied. Focused TRI60/budget/long180 tests pass 64/64;
the complete repository passes 751 tests in 352.31 seconds with 580 existing
Matplotlib/Pyparsing deprecation warnings. Three campaign CLI help surfaces,
the result-printer help surface, Python compilation, diff checks, the worker's
static shell contract test, and local Git-Bash `bash -n` pass. The runbook also
repeats `bash -n` in the pinned Tigris worktree. No donor code, SSH action,
push, Slurm submission, cancellation, or
remote mutation occurred. Exact commit/push and Tigris execution remain.

## TRI60 M1 greedy ensemble diagnostic (2026-08-27)

An additive validation-only campaign now evaluates the completed 20-condition
M1 compression screen with deterministic forward ensemble selection. The
candidate set is exactly imported `SOURCE_M1_LOGIT` plus the 19 screen fits;
`LOGIT_D000E` remains a reference and cannot enter an ensemble. Three paths
grow from one through five distinct members by macro AUC, linear macro R50,
and equal AUC/R50 recovery from source M1 to the D000E teacher. Every selected
ensemble uses a uniform probability mean with FP64 accumulation and one FP32
cast. The complete per-class metrics are retained at every path size.

Inference is split across five independent GH200 jobs with four checkpoints
per job. Each builds the exact-HLT 957,541-row validation view once in RAM,
persists only identity digests, labels, and FP32 probabilities, and deletes
the particle view at exit. Individual shards are capped at 320 MiB and the
complete prediction set at 2 GiB. One 72-CPU reducer uses at most 32 forked
workers over shared read-only arrays and deduplicates candidate sets shared
between objectives. It also recomputes every singleton and fails closed if it
does not reproduce the immutable training metrics.

The eight-job `hcwm1ens_` DAG is authenticate, five sibling inference shards,
greedy reduction, and completion. It is nice 10000, source pinned, and has no
scheduler dependency on or write path into TRI60, DX, CE5, or the completed
M1 screen. It has no fit, training target, persistent logits/particle views,
automatic finalist selection, or final-test capability. Exact specification,
contract, queue, audit, and result commands are in
[`HCWDL_TRI60_M1_GREEDY_ENSEMBLE_RUNBOOK.md`](HCWDL_TRI60_M1_GREEDY_ENSEMBLE_RUNBOOK.md).

Implementation evidence is 6 focused new tests and 60 passing neighboring
TRI60/M1-screen tests. The complete repository suite passes 735 tests in
354.06 seconds with 580 existing Matplotlib/Pyparsing deprecation warnings.
All three new CLI help surfaces and Python compilation pass. The worker has a
focused static contract test; the local Windows Bash executable could not
start, so `bash -n` remains to be run in the clean Tigris worktree. No donor
file was copied, no legacy map entry is required, and no SSH, push, Slurm
submission, cancellation, or remote mutation occurred. Tigris execution is
the remaining step.

## TRI60 D000 matched-seed diversity ablation (2026-08-27)

An additive full-data study now isolates whether stochastic diversity explains
the gap between the five-seed CE ensemble and the original same-seed LOGIT
D000 ensemble. It retrains the five original frozen-teacher D000 edges using
the exact `CE5_S01` through `CE5_S05` initialization, training, and sampler
seed domains in fixed teacher order, while preserving exact HLT inputs,
C25P75/T2, 60 passes, batch size 256, optimizer/schedule, and checkpoint
selection. The five fits launch in parallel and form the fixed uniform
`SD5_LOGIT_D000E` validation ensemble.

The campaign is source-pinned and isolated under the `hcwsd5_` namespace with
positive nice value 10000. It has no scheduler dependency on or write path
into TRI60, DX, or CE5. The reducer deliberately publishes no train or
validation probability bank, logits, or particle views; it persists only
compact validation reports and compares against original `LOGIT_D000E`,
`CE5E`, and `U000`. The scientific plan, `/v1` contracts, and exact dry-run
and launch audit are in
[`HCWDL_TRI60_D000_SD5_ABLATION_RUNBOOK.md`](HCWDL_TRI60_D000_SD5_ABLATION_RUNBOOK.md).

Focused evidence is `6 passed` for graph/seed identity, parallel isolation,
campaign publication, uniform validation-only reduction, storage audit, and
thin workers; the neighboring TRI60/CE5 batch is `68 passed`. The complete
729-test repository inventory passes in bounded shards (`315 + 194 + 220`);
the shards avoid the local wrapper's ten-minute limit without excluding any
collected test. CLI compilation/help and `git diff --check` pass. Tigris
execution and installed-Weaver validation remain pending.

## TRI60 source/DX LOGIT ladder zoom curves (2026-08-27)

An additive validation-only diagnostic now produces separate Hbb and Hcc and
combined two-panel signal-efficiency versus QCD-rejection figures restricted
to the requested `[0.30, 0.50]` signal-efficiency interval. The presentation
aliases are `U000 -> Offline`, `LOGIT_U100E -> D100`,
`DX_LOGIT_D083E -> D080`, `LOGIT_U100_from_U050E -> D060`,
`LOGIT_D033E -> D040`, `DX_LOGIT_D083_from_LOGIT_U100E -> D020`, and
`LOGIT_D000E -> D000`. Every legend places Offline first and the D labels in
the exact `D100, D080, D060, D040, D020, D000` order. The immutable report
preserves the real artifact IDs beneath those intentionally simplified labels.

Five curves consume authenticated durable validation probability banks. The
two specialist checkpoints receive validation-only inference on their native
authenticated rung views using source-process parallelism; predictions remain
process-local. The job has its own source-pinned worker and output root and
creates no fit, checkpoint, prediction bank, campaign mutation, scheduler
dependency, selection role, deployable model, or final-test capability. Queue
and inspection commands are in
[`HCWDL_MHPE_TRI60_LOGIT_LADDER_ZOOM_ROC_RUNBOOK.md`](HCWDL_MHPE_TRI60_LOGIT_LADDER_ZOOM_ROC_RUNBOOK.md).

Focused local evidence is `66 passed` across the core TRI60, dense-extension,
existing original ROC, and new zoom-ROC suites. The new synthetic end-to-end
test writes and validates all six PNG/PDF figures, compact curve NPZ, requested
alias/order registry, working points, and content-hashed
`HCWDL_MHPE_THREE_TRACK_60E_LOGIT_LADDER_ZOOM_ROC_REPORT/v1` report. The
combined figure was also inspected visually. Tigris execution and
installed-Weaver validation remain pending.

## Original TRI60 LOGIT D000E Hbb/Hcc rejection curves (2026-08-27)

An additive validation-only diagnostic now computes exact tied-threshold Hbb
and Hcc versus QCD rejection scans for the original, non-DX `LOGIT_D000E`,
with source-matched `M0CE60` and projected-offline-input `U000` references.
The score is `p_signal/(p_signal+p_QCD)`; the y-axis is logarithmic QCD
background rejection, and zero observed QCD passes use the finite empirical
ceiling `N_QCD`. The figure uses a deterministic bounded projection of each
full upper envelope; the report records exact unthinned 30%, 50%, and 80%
working points and unclipped linear-rejection recovery from M0CE60 to U000.

`LOGIT_D000E` and `U000` use their authenticated durable validation banks.
The selected M0CE60 checkpoint receives one validation-only GPU inference
pass, whose predictions remain process-local. Durable outputs are limited to
PNG/PDF figures, compact curve arrays, and the content-hashed
`HCWDL_MHPE_THREE_TRACK_60E_ORIGINAL_LOGIT_D000E_ROC_REPORT/v1` lineage
report. The job has a separate source-pinned worker/root and no scheduler
dependency, source-campaign write, fit, target, prediction bank, deployable
model, selection role, or final-test capability. Queue and result commands are
in
[`HCWDL_MHPE_TRI60_ORIGINAL_LOGIT_D000E_ROC_RUNBOOK.md`](HCWDL_MHPE_TRI60_ORIGINAL_LOGIT_D000E_ROC_RUNBOOK.md).

Focused local evidence is `72 passed` across the core TRI60 campaign, new
diagnostic, reusable ROC curve, original LOGIT/RSET blend, and flat-eight
suites; the scaffold/link check also passes `7 tests`. The synthetic
end-to-end test writes and validates both figure formats, curve NPZ, working
points, recovery, and immutable report. The complete local repository suite is
`720 passed` in 349.12 seconds, with 400 Matplotlib/Pyparsing deprecation
warnings. Tigris execution and installed-Weaver validation remain pending.

## TRI60 flat-eight LOGIT/RSET D000 diagnostic (2026-08-26)

An additive validation-only diagnostic now evaluates the eight underlying
LOGIT/RSET D000 specialists with equal nominal weight. It authenticates the
durable five-member `LOGIT_D000E` and three-member `RSET_D000E` banks and
computes `5/8 LOGIT + 3/8 RSET`, while embedding the prior equal-family 50/50
calculation as a direct comparator. The report records the exact frozen member
registry and the negligible FP32 family-bank rounding boundary rather than
claiming bitwise equivalence to fresh raw-specialist inference.

The diagnostic has the new
`HCWDL_MHPE_THREE_TRACK_60E_D000_LOGIT_RSET_FLAT8_REPORT/v1` contract and an
independent source-pinned CPU worker, CLI, tests, low-priority Slurm job, and
output root. It creates no training fit, checkpoint, target, prediction bank,
deployable model, scheduler dependency, source-campaign mutation, or
final-test access. Queue and result commands are in
[`HCWDL_MHPE_TRI60_D000_LOGIT_RSET_FLAT8_RUNBOOK.md`](HCWDL_MHPE_TRI60_D000_LOGIT_RSET_FLAT8_RUNBOOK.md).
Focused evidence is `65 passed` across the core TRI60, existing four-D000,
existing 50/50, and new flat-eight suites. The complete local repository suite
is `704 passed` in 322.81 seconds, with 340 pre-existing
Matplotlib/Pyparsing deprecation warnings. CLI help, worker shell syntax, and
`git diff --check` pass. The pushed source commit and Tigris execution remain
to be recorded.

## TRI60 LOGIT/RSET D000 50/50 validation blend (2026-08-26)

An additive post-hoc diagnostic now reads the completed `LOGIT_D000E` and
`RSET_D000E` validation probability banks and evaluates their exact 50/50
probability average. It authenticates each probability lock, validation
manifest, stage report, campaign/graph/recipe lineage, and canonical identity
coverage. The blend uses lexical FP64 accumulation and one FP32 cast. The
report binds the completed `M0CE60` control as zero recovery and `U000` as one
recovery; macro and per-class R50 recovery are calculated in linear rejection
space.

The diagnostic has its own
`HCWDL_MHPE_THREE_TRACK_60E_D000_LOGIT_RSET_BLEND_REPORT/v1` contract,
source-pinned CPU-only worker, CLI, tests, worktree, low-priority Slurm job
name, and output root. It creates no fit, checkpoint, target, deployable model,
persistent prediction array, scheduler dependency, source-campaign write, or
final-test capability. The exact independent queue and result commands are in
[`HCWDL_MHPE_TRI60_D000_LOGIT_RSET_BLEND_RUNBOOK.md`](HCWDL_MHPE_TRI60_D000_LOGIT_RSET_BLEND_RUNBOOK.md).
Focused evidence is `60 passed` across the core TRI60, existing four-D000,
and new LOGIT/RSET blend suites. The complete local repository suite is
`699 passed` in 321.43 seconds, with 340 pre-existing Matplotlib/Pyparsing
deprecation warnings. CLI help and `git diff --check` pass. The pushed source
commit and Tigris execution remain to be recorded.

## TRI60 isolated dense extension (2026-08-24)

An additive full-population 60-pass campaign now densifies the running three
tracks without changing them. LOGIT adds exact D083, D050, and D017 stages to
the existing U000/U050/U100/D066/D033/D000 support; RSET and RREL add D075 and
D025 around the existing U100/D050/D000 support. Exact unchanged source
specialists are imported, while every changed edge, expanded uniform ensemble,
and terminal M1/M1E/M2 artifact has a distinct `DX_` identity. The frozen graph
contains 48 fresh fits, 15 reducers, and 23 read-only source-fit imports.

Campaign creation authenticates the already-complete U-stage reports and
probability banks, allowing new early stages to run immediately. One exact
`afterok` source gate waits for the running campaign-complete job before
authenticating lower reusable specialists. The command plan has zero source
commands and cannot hold, cancel, or write source jobs/artifacts. The extension
uses a separate worktree, `hcwtri60x_` job namespace, root, ledger, reports,
locks, monitoring, and restart-from-zero recovery.

Representation targets remain process-RAM-only and rolling resume remains
disabled. GPU fits and reducers request 72 CPUs to use the established spawned
full-population preprocessing path. Focused dense/original TRI60 tests pass at
65 tests. The final complete local repository suite passes at 693 tests in
316.93 seconds, with 340 pre-existing Matplotlib/Pyparsing deprecation warnings. CLI
help passes for all seven new create/run/submit/monitor/recovery entry points;
all fifteen new Python files compile. An exact pushed commit and Tigris
submission remain to be recorded.
The scientific specification and exact queue block are in
[`HCWDL_MHPE_TRI60_DENSE_EXTENSION_PLAN.md`](plans/HCWDL_MHPE_TRI60_DENSE_EXTENSION_PLAN.md)
and
[`HCWDL_MHPE_TRI60_DENSE_EXTENSION_RUNBOOK.md`](HCWDL_MHPE_TRI60_DENSE_EXTENSION_RUNBOOK.md).

## TRI60 fixed four-D000 cross-track ensemble diagnostic (2026-08-24)

A separate validation-only diagnostic now evaluates the frozen completed set
`LOGIT_D000_from_U000`, `LOGIT_D000_from_U050E`,
`RSET_D000_from_U000`, and `RREL_D000_from_U000` on one authenticated shared
exact-HLT validation stream. The primary result is a temperature-one FP32
probability ensemble with exact uniform 1/4 weights and lexical FP64
accumulation. Four fixed leave-one-out averages and component-diversity rows
are diagnostic only. The report binds campaign, graph, recipe, split,
selection, report, checkpoint, validation-identity, label, and transient-logit
hashes under
`HCWDL_MHPE_THREE_TRACK_60E_D000_CROSS_TRACK_ENSEMBLE_REPORT/v1`.

The worker requires only the four completed reports/checkpoints—not campaign
completion—and creates no fit, target bank, view cache, checkpoint, finalist,
deployable model, final-test access, or active-DAG dependency. Focused local
evidence: `53 passed` across the complete TRI60 test file plus the first four
diagnostic tests, followed by `5 passed` for the diagnostic suite including a
mocked end-to-end evaluator. The complete local repository suite then passed
with `682 passed` in 325.80 seconds (warnings were pre-existing Matplotlib/
Pyparsing deprecations). No Tigris diagnostic has yet been submitted.
The exact independent submission and result commands are in
[`HCWDL_MHPE_TRI60_D000_CROSS_TRACK_ENSEMBLE_RUNBOOK.md`](HCWDL_MHPE_TRI60_D000_CROSS_TRACK_ENSEMBLE_RUNBOOK.md).

## TRI60 outside-lens strategy explainer (2026-08-23)

[`HCWDL_MHPE_TRI60_STRATEGY_EXPLAINER.md`](HCWDL_MHPE_TRI60_STRATEGY_EXPLAINER.md)
now gives a standalone scientific explanation of the running campaign. It
defines U/D coordinates, source-qualified specialists, singleton and
multi-member `E` artifacts, the `1+2+3+4+5` LOGIT triangle, the two `1+2+3`
representation triangles, probability-teacher/carrier separation, exact
losses, M1E-to-M2 compression, deployment boundaries, result interpretation,
and claim limitations. It also records both completed 300k predecessor result
families: C25P75 multi-horizon ensemble-versus-local-KD comparisons and the
fixed LOGIT+RSET+RREL ensemble's 50.8% AUC/78.2% R50 recovery at exact HLT.
The README links the explainer. This documentation does not alter the immutable
graph, recipe, execution, or active jobs.

## TRI60 composite-of-composite recovery ancestry repair (2026-08-23)

The exact cancellation of composite ledger root job `91376` and its 34
registered descendants succeeded without touching the five running RSET/RREL
parents.  Subsequent recovery creation failed closed with `TRI60 external
recovery dependency is unbound`.  The ordinary recovery ancestry walker knew
only ordinary/resource recovery contracts and therefore forgot authenticated
completed parents when its immediate subject was a composite recovery.

Completed-dependency discovery now recognizes a composite subject, traverses
both of its content-hash-bound subject specs, and imports only rows marked
complete by each bound immutable monitor.  Subject and monitor content hashes
are checked again before their tasks are admitted.  Active rows remain
excluded and continue to be represented by exact external Slurm dependency
IDs.  A focused regression covers completed parents from both composite arms.
The cancelled jobs produced no reusable outputs and will be recreated through
a new source-pinned recovery; already completed artifacts and the five running
representation fits remain untouched.

Local evidence passes: all 49 TRI60 tests and the broader HCWDL suite at 416
passed with 340 pre-existing Matplotlib/Pyparsing warnings.

## TRI60 true process-parallel preprocessing repair (2026-08-23)

The first 72-CPU composite reducer, Tigris job `91376`, proved that the prior
source-parallel repair did not deliver CPU parallelism.  During active train
cache construction it averaged 1.03 active cores out of 72 over a 30-second
window, read only 0.137 GiB, and logged NumExpr's rejection of a requested
thread count above its default 64-thread ceiling.  The allocation was correct;
the implementation was not.  Both the outer source fan-out and inner repair
fan-out used Python thread pools, while the row-wise repair kernel is
Python-heavy and therefore remained constrained by the GIL.

The balanced cache path now uses a bounded spawned `ProcessPoolExecutor` for
production source partitioning.  The 72-CPU plan retains eighteen concurrent
source producers, gives each producer one repair lane to prevent nested
oversubscription, keeps at most one source-sized result per producer in IPC,
and continues writing batches into authenticated fixed source slices.  Thus
process completion order cannot change canonical identities or epoch sampler
replay.  Spawn rather than fork prevents inherited CUDA state.  The original
thread backend remains available explicitly for compatibility and focused
tests, but production cache construction defaults to processes.  Both TRI60
workers now set NumExpr, OpenMP, MKL, and OpenBLAS thread counts to one before
the task Python process starts.

Local evidence passes: 56 focused TRI60/view-cache tests, 190 broader
homotopy/unified-balanced/MHPE tests, and the complete repository suite at
676 passed with 340 pre-existing Matplotlib/Pyparsing warnings.  Regressions
cover bounded spawned process scheduling, exact source identity, nested
numeric-thread caps, out-of-order canonical cache reassembly, and unchanged
epoch sampler replay.  No scheduler action was performed locally.  Live
reducer `91376` still runs the ineffective thread implementation and should be
replaced through an exact-ledger, source-pinned recovery after this repair is
committed and pushed; the five independent running RSET/RREL parents remain
out of scope.

## TRI60 completed inherited-parent race repair (2026-08-23)

While the split-ledger composite recovery was being prepared, inherited
representation parent job `91040` (`train_RSET_D000_from_U000`) completed
successfully.  That exact job is bound in the representation recovery command
plan but is not a row in its newer submission ledger.  Composite recovery now
loads the authenticated inherited dependency registry and retains that exact
job ID when the parent is absent from the newer ledger.  Consequently an
already-successful `afterok:91040` resolves immediately, rather than causing
the composite creator to reject the parent or redundantly retrain it.  The
same rule also remains valid if an inherited parent is still running; no broad
name-based scheduler discovery is introduced.

The regression removes the synthetic parent from pre-recorded completed
ancestry, binds it only through the prior command plan, and proves that the
new reducer command preserves job `91040` while the other active
representation fits remain outside the recovery closure.  The complete TRI60
test file passes 46 tests in 7.99 seconds; Python compilation and
`git diff --check` also pass with line-ending notices only.  No Tigris job was
cancelled, submitted, held, or otherwise changed locally.

## TRI60 split-ledger composite recovery (2026-08-23)

The active campaign has two legitimate execution generations: LOGIT tasks in
the `5717a3ce` recovery ledger and replacement representation tasks in the
`cfc70607` recovery ledger. A normal recovery against either ledger cannot
both repair cancelled LOGIT reducer `90660` and preserve running RSET/RREL
jobs `91035`, `91037`, `91044`, `91046`, and `91049`. The versioned
`HCWDL_MHPE_THREE_TRACK_60E_COMPOSITE_RECOVERY_SPEC/v1` now authenticates both
subjects and monitors, assigns task ownership by track, preserves active
exact-ID parents, replaces only terminal/cancelled descendants, and constructs
one new cross-track tail. Every recovered GPU command requests 72 CPUs and is
pinned to the repaired source. No cancellation or submission was performed
locally.

The synthetic split-ledger regression proves that active representation fits
are absent from the recovery closure, their exact job IDs are retained as
dependencies, stale LOGIT dependencies are replaced, all three new M1 tasks
join at `reduce_M1E`, and command-plan validation is exact. The recovery
submitter also accepts the composite per-task superseded-job registry instead
of assuming every replacement belongs to the primary ledger.

Local acceptance is complete: the focused TRI60/cache suite passes 54 tests;
the complete repository suite passes 674 tests in 324.63 seconds with only
the existing 340 Matplotlib/Pyparsing warnings; both composite/recovery CLIs
compile and expose help; and `git diff --check` is clean.

## TRI60 full-population preprocessing parallelism repair (2026-08-23)

Full-data reducer job `90660` established an execution bottleneck: after more
than two hours it had read about 49 GiB with roughly one CPU of effective
usage despite a 16-CPU reservation. The existing thread pool covered only the
final balanced-view transform; ROOT projection, selection, assignment joins,
and coupling lookup were produced by one serial generator upstream.

The shared unified-balanced cache builder now has a bounded source-parallel
mode. A 72-CPU recovery defaults to eighteen independent source producers with
three view-transform workers each. Produced batches may complete out of order,
but the RAM cache assigns every source an authenticated fixed slice and
restores exact split/source-entry order before sampler replay. Per-source
entries must increase strictly, source counts must equal the selection
manifest, and all malformed, duplicate, cross-source, missing, or excess rows
fail closed. The queue is bounded to two batches per source worker; only the
coordinator writes the one final train/validation cache, and no repaired view
is persisted.

Recovery creation now accepts independently validated CPU overrides for
LOGIT, reducer, and representation resource classes. The immutable original
campaign remains at 16 CPUs; a deadline-oriented source/resource-pinned
recovery requests all 72 effective GH200-node CPUs for fit, reducer, and
representation classes without changing graph, rows, batch size, passes,
losses, seeds, coordinates, or output ordering. Only cache construction is
expected to use most of that allocation; the single-GPU phase remains
GPU-bound. The execution-only source allowlist now includes the
shared balanced runner, ephemeral cache, and recovery creator needed for this
repair. No live job was cancelled, requeued, or submitted locally. Real Tigris
speedup and utilization remain to be measured after the current reducer
reaches a safe cutover point.

Local evidence for the repair is complete: the cache/TRI60 focused suite
passes 51 tests; the broader unified-balanced/TRI60 surface passes 85 tests;
and the complete repository suite passes 672 tests in 358.00 seconds (only
the existing 340 Matplotlib/Pyparsing warnings). Python compilation and the
updated recovery CLI help pass, and `git diff --check` is clean. Tests cover
bounded 72-CPU planning, concurrent source execution, out-of-order canonical
reassembly including all TRI60 metadata fields, strict within-source order,
and byte-identical HLT sampler replay across three epochs.

## TRI60 full-data 60-pass exact-HLT CE control (2026-08-22)

The additive `M0CE60` control anticipated by Section 17 of the three-track
plan is implemented without changing the immutable TRI60 graph or any active
recovery ledger. It is one fresh ordinary unified 21-channel HLT Particle
Transformer trained on every authenticated mapped train row for 60 passes
with unweighted CE only and validation every pass. It has exact `D000` HLT
inputs, no teacher/KD/offline/representation target, no final-test capability,
and the same initialization/training/sampler seed alias as `M2`. Thus the
registered validation comparison is the paired `M2 - M0CE60`; imported
20-pass `M0paired` remains contextual.

The new seven-contract `/v1` family, one-node graph/spec/command plan,
additive no-resume training authority, strict selected-checkpoint loader,
worker, and create/run/submit CLIs are present. The live plan is exactly one
dependency-free job named `hcwce60_train_M0CE60`, using 16 CPUs, 256 GiB RAM,
one GH200, and a three-day walltime. It reads the source TRI60 campaign and
foundation immutably, writes only beneath its own root, contains no source job
IDs or cancellation operation, and journals its one submission. Train and
validation views are built once in RAM; durable outputs are only selected and
final model envelopes, reports, contracts, the ledger, and attestation.

Final local evidence:

- focused CE-control tests: 3 passed (39 deselected), including a synthetic
  no-resume fit and authenticated custom-checkpoint reload;
- complete repository suite: 667 passed in 320.12 seconds;
- Python compilation: 7/7 changed Python entry points/modules;
- CLI help: 3/3; contract inventory: 7/7;
- `git diff --check`: clean (line-ending notices only).

The implementation-authoritative
[plan](plans/HCWDL_MHPE_TRI60_CE60_CONTROL_PLAN.md),
[contract](contracts/HCWDL_MHPE_TRI60_CE60_CONTROL.md), and
[runbook](HCWDL_MHPE_TRI60_CE60_CONTROL_RUNBOOK.md) are indexed. No job,
checkpoint, source-campaign artifact, final-test row, or remote scheduler state
was touched during local implementation. No donor files were copied. The
control reuses the already accepted TRI60 production cache/training path;
submission still requires an exact pushed source, canonical dry ledger, and
the two explicit control authorization phrases.

## Dense C25P75 frozen-D100-on-HLT diagnostic (2026-08-20)

An additive validation-only evaluator now answers the collaborator's clean
domain-transfer question.  It loads the exact selected component checkpoints
of the plotted dense `U100E` D100 ensemble, evaluates every component on the
canonical HLT validation view without fitting or checkpoint reselection, and
recombines them with the campaign's authenticated anchor-50 rational weights.
The content-hashed report compares the same frozen ensemble on native D100 and
HLT inputs against `M0paired`, exact-HLT `D0E`, and `M1`; it also retains every
D100 component's HLT metrics and exact report/checkpoint lineage.  The command
is `scripts/evaluate_hcwdl_mhpe_d100_on_hlt.py`.  It permits only the dense
C25P75 300k/60-pass profile, uses validation identities only, and records
`final_test_accessed=false`.
Focused ROC tests pass 9 tests; the broader MHPE-plus-ROC regression passes
42 tests. Python compilation, CLI help, and `git diff --check` are clean. No
Tigris inference or job submission has occurred for this diagnostic yet.

## HCWDL full-data three-track 60-pass plan (2026-08-20)

The implementation-authoritative
[three-track plan](plans/HCWDL_MHPE_THREE_TRACK_60E_FULL_IMPLEMENTATION_PLAN.md)
is now implemented and locally acceptance-ready. The immutable graph registers
one fresh shared all-mapped U000 root; a 15-specialist triangular LOGIT path;
six-specialist triangular RSET and RREL paths; one M1 per track; the fixed
three-M1 probability ensemble; and one C10P90/T1 M2. It contains exactly 32
fresh 60-pass fits, 12 scientific ensemble reducers, the real one-member U000
probability publication, and 50 topologically ordered Slurm tasks.

The corrected representation implementation was integrated from clean commit
`acecf9f74dab3d4ac675d8160cfb5decf83ba680`; exact donor blobs and source
SHA-256 values are recorded in `docs/LEGACY_SOURCE_MAP.md` and authenticated
again by the campaign integration lock. There is no runtime import or
published path to either development worktree. Ordinary unified logits and
RSET/RREL surfaces share one student forward, and installed-Weaver parity is a
required source-pinned launch parent.

The new `/v1` contract family, graph/recipe, RAM target adapter, no-resume
training engine, compact probability publisher, runner/workflow, aggregate,
deployable M2 extraction, source-pinned creation/submission, exact monitor and
cancellation, restart-from-zero source/resource recovery, bounded production
acceptance, ten thin CLIs, and three Slurm workers are present. Train/validation
particle views and all jet/set/relation targets are process-local RAM only.
Rolling optimizer/RNG/model generations and partial checkpoint reuse are
forbidden. Durable output is limited to compact 15-class FP32 probabilities,
selected/final checkpoints, and small lineage/audit/report artifacts; the
campaign projects 8 GiB durable usage and reserves at least 16 GiB free space.

Final local evidence after self-review:

- representation plus TRI60/UB integration regression: 108 passed;
- TRI60 and worker-focused regression after failure injection: 24 passed;
- complete repository suite: 647 passed in 300.00 seconds;
- all 240 source/script Python files compiled in memory;
- CLI help: 10/10;
- contract inventory and repository-relative Markdown links: 2/2;
- Git-Bash syntax: all three TRI60 workers;
- `git diff --check`: clean (line-ending notices only).

The implementation-authoritative [contract](contracts/HCWDL_MHPE_THREE_TRACK_60E_FULL.md)
and [runbook](HCWDL_MHPE_THREE_TRACK_60E_FULL_RUNBOOK.md) are linked from the
repository indexes. No Tigris job, Git push, cancellation, or final-test access
was performed locally. The only remaining pre-submission evidence is external:
on the exact pushed commit, run the source validator, installed-Weaver parity,
one bounded real-GH200 RAM/no-resume production-worker acceptance, and the
complete 50-job dry ledger. The bounded worker is an operational acceptance,
not a reduced scientific campaign; live science submission remains separately
phrase-authorized.

The first read-only Tigris foundation-authentication attempt at pushed commit
`c5a31a04` exposed a pre-submission integration defect before any Slurm job or
campaign artifact was created. The TRI60 adapter incorrectly expected a
`selection_manifest_sha256` entry in the full-foundation spec's `parents`
mapping. The canonical full-foundation contract stores only the split hash
there; the authenticated row-selection hash is the `content_hash` of the
artifact named by `artifact_paths.selection_manifest`. The adapter now loads
and validates the split and selection artifacts, proves the selection binds
the exact split, and records both actual hashes. A regression uses the real
contract shape without the nonexistent parent field. Both changed Python
files compile and `git diff --check` passes. The local default Python lacks
NumPy, so the focused regression must be rerun in `atlas_kd_tigris` on the
corrected pushed commit before proceeding to recipe selection or acceptance.

That corrected foundation regression passed on Tigris at pushed commit
`d218961c`. The source-pinned evidence job `90374` and acceptance job `90375`
then completed successfully. Dependent autolaunch job `90390` authenticated
the evidence, created the immutable full campaign, passed the complete dry run
(`50` jobs, `32` fresh fits), and submitted the exact live ledger. Its final
status-print snippet subsequently raised a local `NameError`; that happened
after submission and did not prevent the DAG from being registered.
`authenticate` (`90455`) and `preflight` (`90456`) completed. `train_U000`
(`90457`) failed before its first optimizer update, after building both RAM
caches, because the runner passed the versioned recipe field
`learning_rate_floor_fraction` as a constructor keyword while the canonical
`Tri60TrainingRuntime` field is named `minimum_lr_fraction`. The runner now
maps the versioned recipe key to the canonical runtime field, and a regression
constructs the runtime through this production adapter. The graph, recipe
value, foundation, views, losses, seeds, and checkpoint policy are unchanged.
Source-pinned recovery must cover the failed/downstream closure from the
original immutable ledger. No valid U000 selected checkpoint or rolling
resume exists, so U000 restarts from the beginning.

The first source-repair recovery at pushed commit `91a8da00` proved that the
runtime-field correction reached `train_tri60_node`, then exposed a second
pre-update production-adapter defect after rebuilding both full RAM caches.
Recovery job `90564` failed because the production U000 cache requested
`include_hcwdl_metadata=false`; consequently it omitted canonical identity
digests and would also have omitted the visible-index and family-code fields
required by the common strict TRI60 batch boundary. The bounded acceptance
had correctly built the same P0 view with HCWDL metadata enabled, so it did
not exercise this divergent production option. The production student-cache
adapter now requests the same strict HCWDL metadata for every track, including
LOGIT-only U000. LOGIT models still consume only features, vectors, and masks;
the additional fields authenticate identities and the shared batch contract
and do not add an offline inference input or representation loss. A focused
regression proves that the real U000 adapter cannot disable this metadata.
This retains the acceptance-measured memory topology and changes no graph,
view values, loss weights, seeds, schedule, or checkpoint policy. A new
source-pinned failed/downstream recovery is required; U000 again has no valid
partial state and restarts from update zero.

That second source-repair recovery at pushed commit `5717a3ce` successfully
completed the fresh full-data U000 fit (`90657`, 60 passes) and its compact
probability publication (`90658`). The independent LOGIT specialists then
entered training normally. The six first-generation RSET/RREL specialists
failed before target construction or their first optimizer update because the
carrier adapter treated `RowSelection.source_rows == -1` as an empty source.
In the authenticated all-mapped full-population selection, `-1` is the
canonical sentinel meaning every mapped row in that source; explicit
selections may also legitimately contain an exact zero-row source. The
carrier partition resolver now translates `-1` through each authenticated
split record's `mapped_entries`, skips only exact zero-row partitions,
preserves original source-file IDs, rejects per-source overcoverage, and fails
closed unless the nonempty partitions sum to the exact authenticated train
population. Regressions cover mixed all-row/explicit/empty sources and total
coverage mismatch. This is an execution-only population-adapter correction:
the graph, selected rows, views, targets, losses, seeds, schedule, and durable
storage policy are unchanged. U000, its reduction, and the running LOGIT work
must be preserved; source-pinned recovery is limited to the failed RSET/RREL
downstream closure. Because that closure later rejoins a still-active LOGIT
track at `M1E`, recovery command plans now also retain exact `afterok`
dependencies on healthy active jobs in the immutable subject ledger. Completed
external parents need no scheduler dependency because their task attestations
are authenticated; failed/downstream jobs remain bound to new recovery IDs.
This prevents recovery from either cancelling healthy LOGIT work or releasing
the cross-track reducer before that work completes.
Legacy repeated-recovery chains may also omit a dependency such as
`preflight` from the current ledger because its authenticated success belongs
to an older recovery ancestor. Dependency validation therefore walks the
immutable parent-recovery chain and accepts only task IDs carried by validated
completed-task attestations; an absent dependency that is neither completed
in that ancestry nor bound to an active subject job still fails closed.

The resulting source-repair jobs (`90920`, `90922`, `90925`, `90929`,
`90931`, and `90934`) then passed carrier population planning and reached the
RAM target runtime, where all six failed before a teacher forward because the
selected-checkpoint loader returned the U000 module in training mode. The
target runtime correctly rejects training-mode teachers: active dropout would
make representation targets nondeterministic. The bounded acceptance did not
expose this production divergence because its probability-target helper called
`model.eval()` before reusing the same object for representation targets.
`load_tri60_model` now establishes the canonical inference boundary centrally
after restoring the authenticated model/trimmer state:
`model.to(device).float().eval()`. Every existing caller uses selected
checkpoints only for inference, probability reduction, or teacher-target
construction. A real selected-checkpoint regression now proves the returned
model is FP32 and in evaluation mode. This changes no checkpoint bytes,
teacher outputs under the intended evaluation policy, graph, loss, seed,
schedule, or population.

The subsequent execution-only throughput pass adds bounded CPU/GPU overlap
and removes avoidable per-update device synchronization for the remaining
TRI60 work. A single producer thread now constructs at most one not-yet-
consumed batch ahead of the GPU for training, validation, compact probability
inference, and process-local representation-target generation. The producer
alone advances and closes its source iterator; values retain exact source
order, source/finalizer failures are delivered on the consumer thread, and a
clean end-of-stream is published only after source cleanup succeeds. The
depth-one bound adds only one batch of transient RAM and publishes no new
particle view or representation-target artifact.

Base-loss target validity and finite-value predicates are resolved in one
consolidated device-to-host transfer per update instead of one scalar transfer
per predicate. Report-only loss scalars accumulate as detached FP64 device
values and transfer once per completed pass instead of once per metric per
update. This accumulator is outside autograd. Every pass now emits its node,
pass/update progress, validation AUC, training seconds, and validation seconds
to the Slurm log, and terminal reports record the exact throughput mode.

No scientific or resource semantic changed: the effective/global batch size
remains 256, one GH200 remains requested, samples and batches retain their
registered order, the optimizer/update count and LR schedule are unchanged,
all fits still execute 60 passes, and selected/final checkpoint policy is
unchanged. A seeded two-pass U000 parity regression proves the synchronous and
prefetched paths have identical validation histories, mean loss reports,
selected/final updates, and tensor-equal selected/final model states. Further
tests prove depth-one lookahead, order preservation, early-close finalization,
source and finalizer error propagation, and the consolidated fail-closed
probability checks.

Local evidence after this optimization:

- focused TRI60 suite: 36 passed;
- broader MHPE/representation/UB/homotopy suite: 272 passed;
- complete repository suite: 660 passed in 313.04 seconds, with the existing
  Matplotlib/Pyparsing deprecation warnings only;
- TRI60 CLI help: 10/10;
- changed Python/test surfaces compile;
- `git diff --check` passes with line-ending notices only.

No donor code was copied and no contract version changed because this is an
execution-only implementation optimization. No Tigris job, cancellation,
submission, or final-test access occurred locally. A new source-pinned
recovery may use the changed runner/training files already permitted by the
TRI60 source-repair allowlist. The actual GH200 walltime improvement remains
external evidence: compare the new per-pass timing log against the existing
jobs before changing any walltime request or making a quantitative speedup
claim.

The six post-eval-mode source-repair jobs at commit `921d66d9` (`90945`,
`90947`, `90950`, `90954`, `90956`, and `90959`) all reached the first direct
balanced carrier batch and then failed at the same identity boundary. Direct
carrier streams publish their authoritative canonical
`source_path::tree::entry` keys but do not duplicate SHA-256 identity digests
at the top level; that duplication is added only when a complete student view
is materialized in `EphemeralPmardViewCache`. `_target_batch` had incorrectly
converted the absent optional value to an object array and compared it against
the correctly derived digest, making every valid direct carrier fail.

The target adapter now owns the deterministic key-to-digest projection for a
direct stream. When a caller does provide a digest array, its uint8 dtype,
shape, and exact equality to the canonical derivation remain mandatory. The
adapter also checks one-dimensional nonempty identity coverage and label-row
agreement before projection. A production-shaped regression covers the real
no-duplicate stream, an exact supplied duplicate, and a corrupted duplicate.
The fix neither changes identity bytes nor weakens a supplied-digest check; it
aligns the target boundary with the actual authenticated balanced-stream
contract. Focused TRI60 tests pass 37/37 and the broader
MHPE/representation/UB/homotopy suite passes 273/273. The complete repository
suite passes 661 tests in 319.80 seconds with only the existing
Matplotlib/Pyparsing deprecation warnings. These jobs failed before teacher
inference, target publication, student-cache construction, or any optimizer
update, so no partial scientific output is reusable.

The next six source-repair jobs at commit `a9ace4b4` (`90970`, `90972`,
`90975`, `90979`, `90981`, and `90984`) passed direct-carrier identity
projection and consumed all `95,117` selected rows in their first source
partition. They then failed at the shared unified-balanced stream's terminal
coverage assertion: the observed count was correctly `95,117`, while the
assertion compared it to the raw `RowSelection.source_rows == -1` sentinel.
That sentinel means every authenticated mapped row, not negative-one rows.
The common stream boundary now resolves `-1` through the split record's
nonnegative `mapped_entries`, preserves explicit zero-row selections, rejects
unknown negative sentinels, rejects selection beyond mapped coverage, and
retains the exact whole-role coverage assertion when no source partition is
requested. A production-shaped regression consumes a complete all-mapped
per-source stream and reaches terminal coverage; it would fail with the exact
production exception before this correction. The shared stream file is now
explicitly permitted by the TRI60 execution-only source-repair allowlist.

This is a fail-closed execution-boundary correction, not a selection or
scientific change. The graph, all-mapped population, views, teachers, losses,
seeds, schedules, and durable storage policy are unchanged. The six jobs did
not publish a complete process-local carrier/target population, build a
student cache, or execute an optimizer update. U000, its compact probability
publication, and healthy LOGIT work remain reusable. Local evidence after the
correction: affected homotopy/TRI60 suites pass 98/98; the complete repository
suite passes 663 tests in 347.23 seconds with only existing
Matplotlib/Pyparsing deprecation warnings; recovery CLI help, in-memory Python
compilation, and `git diff --check` pass. The next action is a source-pinned
failed/downstream recovery from the immutable ledger containing these six job
IDs after this correction is committed and pushed.

## Dense C25P75 validation ROC diagnostic (2026-08-20)

An additive validation-only plotting command now produces the requested
two-panel Hbb-versus-QCD and Hcc-versus-QCD rejection overlay for the completed
`C25P75_DENSE_ANCHOR50_300K60` campaign.  The collaborator-facing comparison
contains four curves: U000 (displayed as `Offline`), U100E (displayed as
`D100`), M1 (displayed as `KD-distilled HLT-only`), and M0paired (displayed as
`HLT baseline`).  The lineage report retains the qualification that `Offline`
means the projected-native-offline unified U000 model, not native TOFF.
The same command also writes a collaborator-requested five-curve progression
figure containing D100, D75, D50, D25, and D0, bracketed by dashed U000
`Offline teacher` and M0paired `HLT baseline` reference curves.  Both Hbb/QCD
and Hcc/QCD are shown as side-by-side panels, restricted to 40--60% signal
efficiency with logarithmic QCD rejection, a common color progression, and
explicit markers at 50% signal efficiency.  This is the exact dense ladder's gradual
feature-transition sequence; it does not substitute individual specialists
for the registered stage ensembles.
Every curve uses the campaign's frozen
`p_signal / (p_signal + p_QCD)` discriminant, the exact shared 100k validation
identity set, selected model checkpoints, and authenticated T1 ensemble
probabilities.  Zero observed QCD passes use the finite empirical ceiling
`N_QCD`; final test is forbidden.

Whole-campaign completion is not a plotting gate.  When the dense campaign is
still active, the command authenticates the exact four required products
directly: the imported U000 and M0paired reports/checkpoints, the U100E T1
validation probability bundle, and the M1 report/checkpoint.  Its report marks
this state as `authenticated_required_products_only`; an available completion
report is still validated and bound when present.

Reusable curve construction and authenticated evaluation live in
`scouting/hcwdl_mhpe_roc.py`; the thin entry point is
`scripts/plot_hcwdl_mhpe_dense_roc.py`.  It writes PDF, PNG, deterministic NPZ
curve arrays, and a content-hashed lineage report.  Final focused MHPE/ROC
tests passed 39 tests in 10.57 seconds; Python compilation, CLI help, and
`git diff --check` also passed.  The warnings were deprecations from the
installed local Matplotlib/Pyparsing combination.  No Tigris inference, job
submission, sealed-test access, or plot fabrication occurred locally.

## HCWDL-MHPE full-data D000 teacher-distance schedule screen (2026-08-19)

The additive validation-only screen requested after comparing full-data
20-pass and 300k/60-pass MHPE teacher ordering is implemented and locally
queue-ready. Its implementation-authoritative
[plan](plans/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN_FULL_PLAN.md)
registers 24 paired exact-HLT fits:

```text
teacher = {U000, U100E, D066E, D033E}
peak LR = {3e-4, 2e-4, 1.5e-4, 1e-4, 7.5e-5, 5e-5}
loss    = C25P75, temperature 2
budget  = one 80-pass warmup/cosine trajectory per fit
```

Every-pass validation preserves the best checkpoint available by passes 20,
40, 60, and 80. These are explicitly not relabeled as independent shorter
cosine schedules. After train and checkpoint-validation caches are released,
one untouched scoring-half HLT cache is reused serially for all four horizon
checkpoints. The graph therefore owns 24 expensive fits, 96 held-out
evaluations, one aggregate, and one completion task. Initialization, sampling,
dropout, repair, optimizer, and validation-order seed aliases are paired across
all teachers and LRs.

Campaign creation requires one authenticated all-mapped full-data `C25P75`
MHPE source with exactly the consumed teacher products ready. It binds the
full-data foundation, U000 report/checkpoint/train logits, and the T2 U100E,
D066E, and D033E probability bundles; whole-source-campaign completion is not
required. Missing teacher products, C10P90, 300k, and path-only sources fail
closed. All students consume byte-exact HLT. Ordinary final-test access is
zero. This source-readiness refinement versions the source-reuse lock,
operational waiver, and campaign spec to v2 and adds a v1 readiness artifact.

New reusable implementation surfaces are:

- `hcwdl_mhpe_d000_schedule_screen.py`: versioned graph, readiness/source-
  reuse, recipe, validation partition, waiver, campaign, command-plan, and
  report contracts;
- `hcwdl_mhpe_d000_schedule_screen_runner.py`: exact-HLT caches, four target
  kinds, 80-pass training, horizon authentication, scoring, and aggregate;
- `hcwdl_mhpe_d000_schedule_screen_recovery.py`: exact failed/downstream
  closure preserving horizon semantics;
- opt-in horizon-best state in `scouting/engine.py`, including rolling-resume
  state and four immutable selected checkpoints while leaving legacy callers
  unchanged;
- eight thin CLI surfaces, two Tigris workers, the reusable
  [contract](contracts/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN.md),
  and exact [runbook](HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RUNBOOK.md).

Resources are 8 CPUs, 96 GiB, 72:00:00, and one GH200 for each independent
fit; aggregate/completion use 4 CPUs, 32 GiB, and 01:00:00. GPU jobs receive
`USR1@120`. Both ordinary and recovery submitters have durable partial-
submission journals. No performance result can block completion.

Local evidence from the isolated scientific test environment:

- initial focused baseline before edits: 10 passed in 6.56 seconds;
- final focused engine/new-screen/legacy-screen suite: 37 passed in 76.55
  seconds;
- complete repository suite: 564 passed in 650.73 seconds, with one unrelated
  existing PyTorch scalar-conversion warning;
- graph probe: 24 nodes, 26 tasks, 96 evaluations, graph SHA-256
  `682d139a7a2d0000b569b5d743b31ebec88e229b63dbb1f4d93c24923e0af3ff`;
- all 16 campaign contract identities, eight CLI help surfaces, Python
  compilation, both Git-Bash `bash -n` worker checks, Markdown links through
  the full suite, and `git diff --check` passed.

The system-default Python lacked NumPy, so the initial baseline collection
failed before tests; all authoritative results above used the existing
isolated environment at `%TEMP%/hcwdl_schedule_screen_testenv`. No donor file
was copied and no donor commit applies. No installed-Weaver command, Tigris
job, Git push, cancellation, or final-test access occurred in this workspace.
The carried-evidence waiver truthfully records that no new standalone smoke
was run.

The exact next action is to commit and push the implementation and execute the
dry-run plus live block in the runbook as soon as U000, U100E, D066E, and D033E
authenticate. Unrelated source rungs and whole-campaign completion are not
gates. Live submission remains a separate explicit human action.

## HCWDL-MHPE D066 optimization schedule screen (2026-08-18)

The validation-only schedule study requested after comparing the 20-pass
full-data and 60-pass 300k MHPE results is now explicitly a full-data C25P75
study. The implementation-authoritative
[plan](plans/HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_PLAN.md) registers the exact
Cartesian product

```text
passes  = {20, 30, 40, 60, 80}
peak LR = {3e-4, 1.5e-4, 1e-4, 5e-5}
teacher = {U000, U050, U100E}
```

as 60 independent fresh D066 C25P75/T2 fits. It imports an authenticated
all-mapped full-data `C25P75` campaign and reuses its immutable U000/U050
train-logit manifests and U100E/T2 probability bundle. The outer source
campaign need not be complete; a v3 readiness artifact proves every consumed
teacher product is already complete. U000 and U050
target consumers bind both the target-manifest and teacher-report hashes;
U100E binds its probability-manifest hash. All fits share exact initialization,
sampler, dropout, repair, and optimizer seed aliases while varying only the
registered pass count, peak learning rate, and teacher identity.

The complete mapped full-data validation population is reconstructed from its
authenticated assignment shards and partitioned by label-free SHA-256 identity
rank into deterministic checkpoint-selection and untouched schedule-scoring
halves. The latter are evaluated once after loading each
selected checkpoint and cannot affect training or checkpoint selection.
Schedules rank by U100E scoring AUC, then CE, local-teacher contrast, passes,
LR, and lexical identity. Aggregate output also records U100E-U000 and
U050-U000 contrasts. Final test is absent from the task graph and ordinary
access counts are zero.

New reusable surfaces are:

- `hcwdl_mhpe_schedule_screen.py`: v3 graph, recipe, validation partition,
  source-reuse, waiver, campaign, command-plan, report, runtime, aggregate,
  completion, and recovery contracts;
- `hcwdl_mhpe_schedule_screen_runner.py`: once-per-job RAM view construction,
  exact target loading, 60-schedule training, untouched scoring, ranking, and
  completion;
- `hcwdl_mhpe_schedule_screen_recovery.py`: exact failed/downstream closure;
- eight thin create/run/submit/monitor/cancel/recovery CLIs and two Tigris
  workers;
- the reusable [contract](contracts/HCWDL_MHPE_D066_SCHEDULE_SCREEN.md) and
  exact [queue runbook](HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_RUNBOOK.md).

Operational requests are fixed at 8 CPUs, 96 GiB, 72:00:00, and one GH200 per
fit; CPU aggregate/completion jobs use 4 CPUs, 32 GiB, and 01:00:00. All 60
fits have no Slurm dependency on one another. The aggregate depends on all 60
and completion depends on aggregate. GPU workers receive `USR1@120` and use
the existing exact rolling-resume implementation. A separately authorized
carried-evidence waiver records that the source products already exercised
the same production D066 builder, target readers, engine, selection policy,
and Tigris worker; no standalone new smoke was run locally or submitted.

Validation evidence for v3:

- schedule-screen focused suite: 10 passed in 10.47 seconds;
- combined MHPE regression before the two final source-readiness tests:
  41 passed in 15.48 seconds;
- complete repository suite after the final implementation and tests:
  549 passed in 633.06 seconds, with one existing PyTorch scalar-conversion
  warning and no failures;
- eight CLI `--help` checks and all 15 executable v3 contract identities
  passed;
- all schedule-screen package/CLI surfaces compiled, both Slurm workers passed
  `bash -n`, and `git diff --check` passed.

The tests run in an isolated temporary Python 3.14 environment populated from
the repository's declared dependencies plus scikit-learn, which is imported by
existing matcher code. Earlier v2 evidence applies only to the retired 300k
design and does not authorize v3.

No donor files were copied and no donor commit applies. No installed-Weaver
or Tigris job was run for this additive screen in this workspace. Live Slurm
submission remains unauthorized until the user pushes an exact clean commit
and supplies the two phrases in the runbook.

The pushed but unexecuted v1 draft targeted C10P90/300k. The v2 correction
targeted C25P75/300k, but the user interrupted it during local Python import,
before campaign creation or Slurm submission. Both definitions answer the
wrong population-scale question and are retired. Full-data semantics are v3;
old roots must not be executed or relabeled.

## HCWDL-MHPE R-augmented refined continuation (2026-08-17)

The additive 300k/100k validation campaign requested after the completed
`C25P75_300K60` MHPE result is implemented and queue-ready. Its exact graph is:

```text
U100E -> U100R -> D066_from_U100R -> D066Eplus -> D066R
      -> D033_from_D066R -> D033Eplus -> D033R
      -> D000_from_D033R -> D000Eplus -> M1R
```

`U100R`, `D066R`, `D033R`, and `M1R` are fresh same-view C10P90/T1
refiners. The three projection children are fresh C25P75/T2 fits. The new
`D066Eplus`, `D033Eplus`, and `D000Eplus` reducers preserve exact uniform
underlying-specialist weights: four times 1/4, five times 1/5, and six times
1/6. Each reducer reloads every authenticated source component, reproduces the
durable source ensemble byte-for-byte, then performs one canonical all-component
FP64 reduction with a single little-endian FP32 publication. No validation
metric, confidence, class, or label controls an ensemble weight.

The new implementation consists of the plan, reusable contract, and runbook;
three reusable scouting modules; seven create/run/submit/monitor/recovery CLIs;
two Tigris workers; and focused contract, graph, numerical, target-corruption,
recovery-closure, campaign-publication, and CLI tests. The aggregate now
authenticates each PMARD report, outer HCWDL report, runtime record, E+ target
bundle, and stage-to-lock relationship before completion. Recovery is limited
to the exact failed/downstream closure and emits recovery-root attestations.

The first Tigris submission reached the authenticated 300k/100k student-view
caches for `U100R`, then job 89049 failed before its first training update. The
new loss helper had emitted `HCWDL_MHPE_REFINED_U100R`, while the reused
`GenerationalLossConfiguration` contract requires every arm identity to begin
with `HCWDL_UB_`. The helper now emits
`HCWDL_UB_MHPE_REFINED_<node_id>` for all seven nodes. A direct regression
constructs all seven executable loss objects and verifies their exact CE, KD,
and temperature values, closing the test gap that allowed the invalid string
through. No checkpoint, target, report, or scientific result was produced by
the failed fit; a clean source-pinned campaign must replace that submission.

Scientific and operational identities:

- graph: `HCWDL_MHPE_REFINED_CONTINUATION_GRAPH/v1`;
- campaign: `HCWDL_MHPE_REFINED_CONTINUATION_CAMPAIGN_SPEC/v1`;
- target shard/manifest/lock: refined-continuation v1;
- training/runtime/stage/aggregate/completion: refined-continuation v1;
- recovery specification/command plan: refined-continuation v1;
- graph SHA-256:
  `36f5e6a02cd6a78a031e71c659de7a0f65df368005555a3590f8d68369f1484a`;
- 7 fresh fits, 3 reducers, and 12 total tasks;
- GPU request: 8 CPUs, 96G, 06:00:00, one GH200;
- ordinary access: 300k train, 100k validation, zero final-test rows.

Final local evidence:

- expanded focused MHPE plus CLI suite: 123 passed in 215.40 seconds;
- complete repository suite after the Tigris loss-identity correction: 539
  passed in 269.81 seconds, with only the
  existing 14 Matplotlib/Pyparsing deprecation warnings;
- all seven new CLI help surfaces passed;
- scouting and script Python compilation passed;
- contract probe returned the exact graph hash, 12 tasks, 7 fits, 3 reducers,
  and the registered node/ensemble lists;
- repository-relative Markdown links passed in the complete suite;
- `git diff --check` passed with line-ending notices only.

The campaign reuses the authenticated completed source campaign, its 300k
foundation, production worker, installed-Weaver, view-cache, and ensemble
evidence. Per the implementation-authoritative additive plan, it does not
require another standalone smoke; this is evidence carry-forward, not a claim
that this exact continuation graph completed a smoke. No donor code was copied,
so `docs/LEGACY_SOURCE_MAP.md` did not change. No SSH, Slurm, cancellation,
push, or other Tigris mutation occurred. The next step is an explicit commit
and push, followed by the clean detached-worktree create/dry-run/live sequence
in `docs/HCWDL_MHPE_REFINED_CONTINUATION_RUNBOOK.md`.

## HCWDL-MHPE paired endpoint teacher-mixture add-on (2026-08-17)

A separate validation-only endpoint-refinement diagnostic is implemented for
the now-completed source campaign.  It performs one authenticated exact-HLT
validation inference pass through the registered source `M1`, reuses the
authenticated `D000E/T1` validation probability table, joins them by canonical
jet identity, and reports `D000E`, `75/25`, `50/50`, `25/75`, and `M1` fixed
probability blends.  Weights are declared before metric evaluation; only
aggregate metrics are durable and final test remains untouched.  The worker is
source-pinned, uses the repository Tigris activation contract, and requires one
GH200 with 8 CPUs, 96G, and a one-hour request.  This diagnostic asks whether
the raw ensemble retains complementary ranking information after its M1
distillation; it does not select or modify a campaign finalist.

The implementation-authoritative
[endpoint teacher-mixture plan](plans/HCWDL_MHPE_ENDPOINT_MIX_300K60_PLAN.md)
is implemented and queue-ready as a separate validation-only add-on over the
explicitly selected, completed `C25P75_300K60` campaign. It does not retrain or
mutate the source U/D ladder. The v2 target worker reuses the authenticated
`D000E/T1` train/validation probability bundle, performs exactly one
`M0paired` exact-HLT forward pass per role, joins the two components by jet
identity, and publishes four immutable teacher tables:

- `M1_D0only`: 100% D000E;
- `M1_mix90`: 90% D000E plus 10% M0paired;
- `M1_mix75`: 75% D000E plus 25% M0paired;
- `M1_mix50`: 50% D000E plus 50% M0paired.

The first queue attempt exposed that the unexecuted v1 draft had selected the
wrong source family: dense `D0E` rather than the intended original 300k
`D000E`. No endpoint-mixture campaign or Slurm job was created by that attempt.
The source semantics and the entire add-on artifact family are now version 2;
campaign creation accepts exactly `C25P75_300K60` and rejects dense sources.

Closing that source campaign exposed a separate reporting-only defect after
all 16 specialists and M1 had completed: the aggregate compared each outer
training report's semantic graph hash to the content hash of the complete
`graph.json` artifact. For this campaign those are correctly distinct
(`9b2635b0...` semantic identity versus `392cc1a8...` artifact identity).
The aggregate now authenticates `graph.json` against the campaign spec first
and then compares node reports to its authenticated inner semantic hash. It
does not waive, rewrite, or weaken any recovered training lineage. A focused
regression proves both the distinction and fail-closed artifact mismatch.

The first aggregate recovery then advanced to probability-bundle validation
and exposed a second compatibility defect.  All non-dense MHPE profiles reuse
the original v1 target shard/manifest/lock family; those payloads intentionally
omit `recipe_profile` and decode as `C25P75`, while their parent map binds the
exact campaign, graph, recipe, split, and selection.  The profile-aware bundle
validator incorrectly compared that artifact-family identity directly with
the outer `C25P75_300K60` campaign profile.  It now normalizes every non-dense
campaign to the shared v1 target identity while retaining exact lock,
manifest, consumer, parent, role, temperature, shard, and content-hash
validation.  Dense profiles still require their explicit v2 profile.  This is
an execution/reporting compatibility repair only; no target, model, metric, or
scientific contract changed.

The numerical contract is max-subtracted FP32 M0 softmax, exact-rational FP64
mixture accumulation, and little-endian FP32 publication. The four fresh,
exact-HLT students share initialization, sampler, dropout, optimizer, schedule,
population, and update seeds. Each uses unweighted `0.10 CE + 0.90 probability
KD`, `T=1`, runs 60 complete passes with every-pass validation, and selects by
macro AUC, CE, logR50, then earliest update. The retrained `M1_D0only` is the
paired baseline; the source campaign's M1 is contextual only.

After the source campaign completed, the first endpoint-mixture creation
attempt exposed a redundant lineage expectation in the add-on authenticator.
The immutable 300k reuse-lock schema binds `m0paired_report_sha256`; that
validated PMARD report binds `selected_checkpoint_sha256`, and creation hashes
the checkpoint file directly.  The add-on had additionally required a
nonexistent top-level `m0paired_checkpoint_sha256` in the reuse lock.  Creation
now follows the actual authenticated chain—reuse lock to report to checkpoint
file—without weakening or rewriting any source artifact.  A regression proves
valid acceptance plus fail-closed report and checkpoint tampering.

The first live endpoint-mixture training attempt then exposed an execution-
only arm-label defect shared by all four paired fits.  All four workers built
the authenticated 300k/100k exact-HLT RAM views and then failed before the
optimizer started because the new label `HCWDL_MHPE_ENDPOINT_MIX_*` did not
satisfy `GenerationalLossConfiguration`'s established `HCWDL_UB_*` namespace
guard.  The runner now emits `HCWDL_UB_MHPE_ENDPOINT_MIX_*`; CE/KD weights,
temperature, views, targets, seeds, schedules, and graph semantics are
unchanged.  A regression instantiates and checks the exact registered loss for
all four nodes.

The campaign is exactly seven jobs: one target builder; four parallel GH200
fits; aggregate; completion. GPU jobs request 8 CPUs, 96G, 06:00:00, and one
GH200; CPU jobs request 4 CPUs, 32G, and one hour. The implementation includes
content-authenticated targets, campaign/graph/recipe/report/completion
contracts, restart-safe dry/live submission journals, task attestations,
monitoring, exact-ID cancellation compatibility, and exact failed/downstream
same-source recovery. The [runbook](HCWDL_MHPE_ENDPOINT_MIX_RUNBOOK.md) binds
the source campaign explicitly and never discovers a mutable "latest" result.

Review found and fixed one integration defect before acceptance: the first
Slurm wrappers activated Conda directly instead of sourcing the mandatory
repository `sbatch/common.sh`. Both normal and recovery workers now use
`hlt_activate`, the required environment variables/library path, absolute
project paths, and `exec python -s`.

Post-correction local evidence with `PYTHONDONTWRITEBYTECODE=1` and pytest
caches disabled:

- new focused endpoint-mixture suite: 8 passed;
- combined MHPE regression: 40 passed;
- worker-contract plus new focused regression after the wrapper repair:
  8 passed;
- complete post-correction repository suite: 520 passed in 237.57 seconds, with
  only the 14 existing Matplotlib/Pyparsing deprecation warnings;
- all seven CLI help surfaces, Python compilation, all 15 v2 contract
  identities, the exact `D000E` source endpoint, and `git diff --check`:
  passed.

No donor file was copied. No Slurm command was submitted, no source campaign
artifact was modified, and no final-test row was accessed. The remaining
action is to commit/push the v2 correction, allow the already scientifically
complete `C25P75_300K60` source to publish its authenticated campaign-complete
tail, and use that exact source spec with the runbook block.

## HCWDL-MHPE 300k executable-recipe recovery (2026-08-16)

The first `C25P75_300K60` and `C10P90_300K60` jobs (`88447` and
`88470`) failed before building a view or starting an optimizer step.  Both
tracebacks ended at `recipe["batching"]`: the MHPE runner loaded the 300k
unified-balanced foundation's local scientific overlay `recipe.json`, which
intentionally contains only arm weights, pass count, and selection policy.
The foundation's own successful training workers instead load the authenticated
executable HCWDL recipe from `foundation_spec["artifact_paths"]["recipe"]`;
that artifact owns batching, optimizer, and schedule settings.

`hcwdl_mhpe_runner._training_recipe()` now follows the foundation-specific
runtime contract: legacy full-data MHPE profiles retain
`foundation_root/recipe.json` exactly, while both paired 300k profiles load the
executable recipe through the immutable foundation artifact registry.  A
regression constructs deliberately different overlay and executable recipes
and proves both 300k profiles select the latter while the legacy full profile
still selects the former.  Graphs, losses, seeds, populations, views,
checkpoints, resources, and final-test policy are unchanged; no contract was
versioned.  The registered source-repair workflow permits this runner-only
execution correction and must be used instead of requeueing the old pinned
jobs.

Local evidence with `PYTHONDONTWRITEBYTECODE=1`, repository `src` on
`PYTHONPATH`, and pytest caches disabled:

- focused MHPE suite: 22 passed in 5.14 seconds;
- combined MHPE and unified-balanced regression: 51 passed in 31.86 seconds;
- complete repository suite: 501 passed in 238.22 seconds, with only the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- MHPE recovery creation/submission CLI help: passed;
- `git diff --check`: passed.

## HCWDL-MHPE paired 300k/60-pass campaigns (2026-08-16)

The implementation-authoritative
[paired 300k/60-pass plan](plans/HCWDL_MHPE_300K_60E_PAIRED_PLAN.md) is
implemented and queue-ready. It registers two complete, independent MHPE
campaign profiles over the same authenticated 300k train, 100k validation,
and sealed 100k final-test population:

- `C25P75_300K60`: all 15 specialists use `0.25 CE + 0.75 KD, T=2`;
- `C10P90_300K60`: all 15 specialists use `0.10 CE + 0.90 KD, T=2`;
- both keep M1 at `0.10 CE + 0.90 KD, T=1`.

Each profile runs the entire 16-fit/four-ensemble triangular graph for 60
complete passes with validation every pass. They import the completed 300k
unified `U000`, CE-only exact-HLT `M0paired`, and U000 logit bank read-only.
Their graph topology, coordinates, teachers, ensemble semantics, checkpoint
policy, and target-coordinate seeds are paired; roots, artifacts, ledgers, and
job prefixes are separate. The old full-data v1 graph hash remains exactly
`3399cdf7f19e3461b9f5cfdcee2e38257a567d5bdb8547b8deb9dbddd856daf9`,
and the v2 C10 graph remains
`34a35d539e54e4b1983b3a6a62563d720149ba64b8bb5c83ae1eb734e89c04f7`.

New graph/node/recipe/campaign/training-report/waiver contracts are v3 for
C25P75 and v4 for C10P90. The 300k foundation reuse lock is v2 and embeds the
authenticated U000 target-lineage classification, including the one known
legacy digest-shadow form when present. Runtime dispatch retains the old
full-data repair seed and 224-GiB cache ceiling unchanged; the new profiles use
the original 300k balanced repair seed and a 72-GiB cache ceiling under exact
8-CPU/96G/6-hour/GH200 requests. CPU reports use 4 CPUs, 32G, and one hour.

The [paired queue runbook](HCWDL_MHPE_300K_60E_RUNBOOK.md) creates one clean
source-pinned worktree, discovers exactly one completed authenticated 300k
foundation, creates separate v3/v4 roots, materializes both canonical dry-run
ledgers, and submits both 23-job DAGs under `hcwmhpe25p_` and `hcwmhpe90p_`.
Recovery recognizes both new campaign contracts and retains their resources
and separate namespaces. No Slurm command was run locally and no final-test
row was accessed.

Local acceptance evidence with `PYTHONDONTWRITEBYTECODE=1` and pytest caches
disabled:

- pre-change focused MHPE suite: 18 passed;
- final focused MHPE suite: 22 passed;
- broader MHPE/UB/contracts/CLI regression before final review: 136 passed;
- post-review complete repository suite: 501 passed in 258.58 seconds, with only the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- all four graph identities, all CLI help surfaces, Python compilation,
  synthetic paired 23-task dry runs, profile/reuse dispatch, population-specific
  seed/cache behavior, Markdown links, and `git diff --check` passed.

No donor file was copied. Unrelated dirty/untracked user work was preserved.
The remaining action is to commit and push only this implementation, then run
the exact Tigris block in the paired runbook. That block performs both dry runs
before explicit live submission; it does not touch the running full-data
campaigns.

## HCWDL-MHPE C10P90 parallel companion (2026-08-16)

The additive
[C10P90 parallel plan](plans/HCWDL_MHPE_C10P90_PARALLEL_PLAN.md) is fully
implemented and queue-ready. It adds one independently source-pinned
`HCWDL-MHPE-C10P90-FULL` companion to the already running primary campaign.
All 15 non-M1 specialists change from `0.25 CE + 0.75 KD, T=2` to
`0.10 CE + 0.90 KD, T=2`; M1 remains exactly
`0.10 CE + 0.90 KD, T=1`. The authenticated FULL3 foundation, imported U000
and M0paired, 16-fit teacher graph, coordinates, teacher identities, uniform
probability ensembles, seeds, 20-pass schedule, validation/checkpoint policy,
resources, finalists, and final-test boundary are unchanged and paired.

The changed graph, node, recipe, campaign, outer training-report, and
operational-waiver semantics use explicit v2 contracts. Unchanged foundation,
target, stage-report, aggregate, lock, attestation, monitor, submission, and
recovery formats remain v1 while binding the v2 campaign hashes. The original
v1 graph hash remains exactly
`3399cdf7f19e3461b9f5cfdcee2e38257a567d5bdb8547b8deb9dbddd856daf9`.
Profile-aware training, ensemble reduction, aggregation, monitoring,
failed/downstream recovery, and resource recovery use the selected registry;
the default remains byte-compatible C25P75. A review found and fixed an
executable-validation bug that initially compared the v2 campaign against the
v1 creation phrase.

The new [parallel runbook](HCWDL_MHPE_C10P90_RUNBOOK.md) creates a separate
detached worktree, `hcwdl_mhpe_c10p90_full_*` campaign root, dry/live ledgers,
and `hcwmhpe90_` jobs. It reads the completed FULL3 foundation but never writes
to the foundation or the running C25P75 root. It performs a canonical dry run
before the independently authorized live submission. Per the user's explicit
decision, this paired recipe-only ablation requires no new smoke or 300k run.

Local acceptance evidence with bytecode and pytest caches disabled:

- focused `tests/test_hcwdl_mhpe.py`: 18 passed in 5.80 seconds;
- complete repository suite: 497 passed in 291.85 seconds, with only the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- all 13 MHPE CLI help surfaces and all 8 MHPE Python modules compiled;
- all six v2 contract identities have source/documentation traceability;
- all links in the five changed/new Markdown surfaces resolve;
- existing Slurm workers retain absolute `PROJECT_DIR`, isolated environment,
  `LD_LIBRARY_PATH`, and `exec python -s` behavior;
- `git diff --check` passed.

No donor file was copied, no SSH/Slurm command was issued, no campaign was
submitted locally, and no final-test row was accessed. The remaining action is
to commit and push only the listed C10P90 implementation/documentation files,
then run the exact source-pinned parallel queue block.

## HCWDL-UB completed-report aggregation repair (2026-08-16)

Recovery aggregate job `87370` failed after all registered `C05P95` training
outputs completed.  The traceback was reporting-only:
`UnifiedBalancedArmWorkflow` read `report["history"]`, but every authenticated
completed PMARD report contract accepted by the campaign (v4--v6) publishes
its interval-loss records as `training_history`; `history` is only an internal
rolling-checkpoint key.  The workflow now uses one fail-closed
`_report_training_history()` accessor and continues to publish the unchanged
outer aggregate field `loss_history`.  Metrics, selected checkpoints, graph,
losses, views, seeds, and final-test state are unchanged.  A regression rejects
the rolling-checkpoint key and malformed histories.

Focused local evidence after the repair: `tests/test_hcwdl_unified_balanced.py`
passes 29/29 in 34.36 seconds under the repository's `tagging-hlt` environment.
The pre-change focused suite passed 28/28.  The remaining operational action is
an aggregate-only, source-pinned retry for the affected arm; no model training
must be repeated.

## HCWDL multi-horizon projection-ensemble implementation (2026-08-16)

The implementation-authoritative
[multi-horizon projection-ensemble full-data plan](plans/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md)
is fully implemented as the additive `HCWDL-MHPE-FULL` contract family. The
immutable graph imports authenticated all-mapped `U000` and `M0paired`, trains
one U050, two U100, three D066, four D033, five exact-HLT D000 specialists,
and one exact-HLT M1: 16 fresh fits total. Four stage reducers form lexical,
uniform probability ensembles by max-subtracted FP32 softmax, FP64 canonical
accumulation/division, and one little-endian FP32 publication. Specialists use
the locked `C25P75/T=2` recipe; M1 alone uses `C10P90/T=1`. Every fit is cold,
paired by target-coordinate seed, runs 20 complete passes with every-pass
validation, and selects macro AUC then CE, logR50, and earliest update.

The implementation adds versioned graph, node, recipe, foundation-reuse,
probability shard/manifest/lock, campaign, plan, report, aggregate, finalist,
execution, completion, final-evaluation, recovery/resource-recovery, and
operational-waiver contracts. The probability-target training adapter is
additive: old logit KD remains the default and has an exact numerical
regression. The reuse lock requires every unaffected FULL3 scientific
model/view/cache/checkpoint/resume core file to remain byte-identical.
Corrected execution builders are pinned as current campaign source while
their imported products are authenticated by the completed foundation lock,
avoiding a false comparison with pre-recovery producer bytes. It records both hashes
for the two generic training entry points extended by the new target type.
The selected U050 logits are published once for four direct consumers. Each
ensemble reducer prepares train and validation views once, loads each
component once, and publishes authenticated T1/T2 train and validation target
bundles. No repaired particle dataset or final-test row is touched by the
ordinary 23-task DAG.

Operational surfaces include source-pinned creation, a complete nonmutating
dry run, resumable journaled live submission, exact-ID monitoring and
cancellation, immutable task attestations covering reports/checkpoints/target
bundles, failed-and-downstream source recovery, monotone resource recovery,
and repeated recovery. Reporting includes every specialist and ensemble,
leave-one-out metrics, scalar ensemble deltas, local-versus-skip comparisons,
pairwise correlation/Jensen-Shannon/classwise disagreement, cache bytes,
target-build time, GPU-hours, and un-clipped M0paired-to-U000 recovery. The
ordinary finalist lock freezes exactly M0paired, five D000 specialists,
D000E, and M1. Final test remains a separate exact-HLT job requiring two
explicit human locks; no final evaluation was run locally.

Local acceptance evidence, with bytecode and pytest caches disabled:

- focused MHPE, FULL3 reuse, unified-balanced, PMARD evaluation/resume,
  contracts, and CLI regression suite: 153 passed in 239.73 seconds;
- complete repository suite: 496 passed in 268.30 seconds, with only the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- all 13 MHPE CLI help surfaces and the final 45-test contract/scaffold/MHPE
  check passed;
- Python compilation, all Slurm worker invariants, contract-version tests,
  bounded synthetic 16-fit/four-ensemble flow, probability/KL gradients,
  target corruption/role/consumer checks, and `git diff --check` passed.

No donor file was copied, no SSH/Slurm command was issued, no campaign was
submitted, and no final-test row was accessed. Per the user's explicit plan,
there is no new smoke or 300k prerequisite. The remaining action is to push
one clean commit, use the runbook's exact authenticated FULL3-foundation
finder to create the Tigris campaign, inspect its canonical 23-task dry-run
ledger, and only then separately authorize live submission with
`SUBMIT HCWDL MHPE FULL EXACT LEDGER`.

## HCWDL full-data coarse factorized/joint three-arm campaign (2026-08-16)

The additive `HCWDL-UB-FULLCOARSE3` campaign is implemented and locally
queue-ready. It reuses—but never mutates—the authenticated completed FULL3
foundation. A new reuse lock checks byte-identical model/view/training core
source, endpoint projection, all foundation parent hashes, all-mapped role
counts, U000/M0 reports and checkpoint bytes, and the durable U000-logit
manifest. It explicitly authorizes all eight direct U000 target consumers:
the two first rungs in every arm and the two second-rung grandparent consumers
in `C10P75G15`.

The graph contains exactly 36 fresh fits. Each of `C25P75`, `C10P90`, and
`C10P75G15` runs a six-edge factorized path at exact thirds and a six-edge
joint path at exact sixths. Transition-index seeds are paired across paths and
recipes, the FULL3 discrete repair seed is retained, the first-edge
grandparent allocation is folded into parent KD, and all later grandparent
edges use the actual two-rung predecessor. Training remains cold-started,
unweighted, temperature 2, 20 passes, every-pass validation, and macro-AUC
checkpoint selection. `D0F` and `J100` are exact HLT endpoints. No M1 or
final-test task exists.

New implementation includes versioned graph/reuse/recipe/spec/report/
completion/recovery contracts, source-pinned arm creation, two-path Slurm
DAGs, independent three-arm submission ledgers, production workers, runtime
and aggregate reports, monitor, exact-ID cancellation, and same-source or
resource-only failed-closure recovery. The runbook intentionally requires no
new smoke: no preprocessing or view semantics changed, and the completed
FULL3 production foundation is the operational evidence. Creation still
fails closed if that foundation or the current scientific core differs.

No donor files were copied and no Slurm job or final-test access was performed
locally. Focused graph/full-data/unified-balanced tests pass 39/39; the new
coarse module tests pass 7/7; and the complete repository suite passes 478/478
in 249.44 seconds. All nine new CLIs pass `--help`; focused compilation,
11 module contract identities plus the runtime contract, Markdown links,
tracked `git diff --check`, untracked whitespace checks, and the 36-fit graph
identity check pass. The only warnings are 14 existing Matplotlib/Pyparsing
deprecations and one local pytest-cache permission warning.

## HCWDL full-data balanced-sidecar wiring repair (2026-08-15)

The source-pinned mapped-identity recovery completed scale calibration, every
train/validation residual-base and legacy-switch shard, both manifests, the
exhaustive coupling audit, coupling lock, and balanced-switch configuration.
Its `train_balanced` and `validation_balanced` arrays then failed uniformly in
seconds. Review found one missed external caller of the corrected
assignment-locked iterator: `build_balanced_sidecar_for_source()` constructed
a validated `DenseAssignmentStore` but did not pass it to
`_selected_source_chunks()`. The call now forwards that exact store. No data,
coupling, coordinate, model, loss, or graph semantics changed.

Because the failed ledger is itself a source-pinned mapped-identity recovery,
ordinary recovery cannot safely describe the next execution. A versioned
second-generation recovery now authenticates the original foundation, the
canonical first recovery, its exact ledger and monitor, and the completed
coupling lock/balanced configuration. It accepts only the three semantic-file
changes for `balanced_assignment_store_wiring_v1`, requires the closure to
begin with `train_balanced` and `validation_balanced`, and preserves the full
completed coupling prefix. Its contracts are
`HCWDL_UNIFIED_BALANCED_FULL_BALANCED_WIRING_REPAIR_EVIDENCE/v1` and
`HCWDL_UNIFIED_BALANCED_FULL_BALANCED_WIRING_RECOVERY_SPEC/v1`.

Files changed for this repair are the balanced builder, FULL3 recovery and
contract modules, recovery creation and monitor CLIs, the FULL3 regression
test, reusable contract, runbook, and this handoff. No external donor is
involved. Final local evidence: the direct wiring regression and synthetic
repeated-recovery closure pass; the homotopy/unified-balanced/FULL3 focused
suite passes 92/92 in 62.60 seconds; and the complete repository suite passes
471/471 in 262.41 seconds with 14 existing Matplotlib/Pyparsing warnings plus
one local pytest-cache permission warning. Recovery create/monitor/submit/run
help, focused compilation, contract-identity checks, Markdown links, and
`git diff --check` pass. No Slurm mutation or final-test access was performed
locally.

## HCWDL full-data mapped-identity preprocessing repair (2026-08-15)

The all-mapped FULL3 foundation at source commit
`8403b1a1c3c0e3bdaef9d63a858163a1922d85fb` completed its complete
train/validation assignment prefix and assignment lock (job `87461`), then
scale-calibration job `87462` failed on raw ROOT entry
`H0HpHm_mixed_new/dnnTuples_nanov15_0000.root::46`. That entry is absent
from the authenticated assignment shard because it is not in the mapped
population. The compact `all_rows: true` row selection had been interpreted
by coupling preprocessing as every raw ROOT entry instead of every
authenticated mapped entry. Bounded 300k selections store explicit entry
lists and therefore did not expose this full-data-only execution bug.

`hcwdl_upper_builder._selected_source_chunks` now uses each validated dense
assignment shard's sorted entry array as the selected-population carrier,
checks its exact per-source count against the split/selection inventory, and
independently proves that every entry is authorized by `RowSelection` and
present in the streamed ROOT source. Scale calibration, coupling construction,
full audit, and sampled audit all use this one path. Other training/data
streams were reviewed and already apply `baseline_mask & labels >= 0` before
the row-selection mask, so no second all-mapped leak was found. The fix is
label-free and does not change assignment construction, row identities,
coupling science, graph, losses, seeds, endpoints, or final-test access.

A classified execution-only recovery was added because ordinary recovery
correctly rejects changed semantic source. Its exact identities are
`HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_REPAIR_EVIDENCE/v1` and
`HCWDL_UNIFIED_BALANCED_FULL_MAPPED_IDENTITY_RECOVERY_SPEC/v1`, with repair
classification `all_mapped_assignment_identity_filter_v1`. It is authorized
only for a foundation closure beginning at `scale_calibration`, requires the
completed all-mapped assignment lock, preserves that prefix, and fails closed
unless the only changed semantic files are `hcwdl_upper_builder.py`, the
FULL3 recovery module, and the FULL3 contract constants. The recovered
foundation starts at scale calibration. After its foundation lock completes,
the three unchanged scientific arms must be launched from the original
source-bound worktree; they do not execute the repaired preprocessing path.

Files changed for this repair:

- `src/hlt_classification/scouting/hcwdl_upper_builder.py`;
- `src/hlt_classification/scouting/hcwdl_unified_balanced_full_recovery.py`;
- `src/hlt_classification/scouting/hcwdl_unified_balanced_full_contracts.py`;
- `scripts/create_hcwdl_unified_balanced_full_recovery.py`;
- `tests/test_hcwdl_homotopy.py` and
  `tests/test_hcwdl_unified_balanced_full.py`;
- the FULL3 reusable contract, runbook, and this handoff.

Local evidence after the final review:

- direct all-mapped regression proves raw entries absent from assignments are
  excluded; a bounded-selection regression rejects assignment drift;
- recovery closure/evidence regression starts at scale calibration, reuses the
  exact assignment lock, and rejects any fourth semantic-file change;
- post-review homotopy/unified-balanced/full-data focused suite: 91 passed in
  76.53 seconds;
- complete repository suite: 470 passed in 290.96 seconds, with the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- recovery create/submit CLI help, focused Python compilation, the two new
  contract identities, and `git diff --check` passed.

No external donor file or donor commit is involved. No local SSH, Slurm
submission/cancellation, ROOT write, final-test access, or Tigris execution of
the repaired code occurred. Exact next task: commit and push the repair, make
a clean detached Tigris recovery worktree, authenticate the failed foundation
ledger/monitor, submit the classified closure beginning at scale calibration,
and bind a replacement arm autolaunch to the recovered foundation-lock job.

## HCWDL unified-balanced all-mapped three-arm scale-up (2026-08-15)

The implementation-authoritative
[full-data three-arm plan](plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md)
is implemented locally as the additive `HCWDL-UB-FULL3/v1` campaign family.
Its reusable semantics are frozen in the
[contract](contracts/HCWDL_UNIFIED_BALANCED_FULL_DATA.md), and the exact
foundation-to-autolaunch procedure is documented in the
[runbook](HCWDL_UNIFIED_BALANCED_FULL_DATA_RUNBOOK.md). Existing 300k U/J and
six-arm artifacts are immutable inputs or contextual results; this work does
not edit or relabel them.

The campaign derives exact train, validation, and sealed-final-test counts
from the authenticated split and selects every mapped train/validation row.
It rebuilds full-population assignments, residual couplings, corrected
balanced switch sidecars, endpoint/resource evidence, fresh unified `U000`
and `M0paired` roots, and one compact identity-ordered FP32 U000 target bank.
Durable reconstructed particle views remain forbidden, and no ordinary task
selects, assigns, couples, caches, or evaluates a final-test row.

The scientific registry contains exactly 38 fresh fits:

- shared CE roots `U000` and `M0paired`;
- factorized-only arms `C25P75`, `C10P90`, and `C10P75G15`;
- in each arm, `U020 -> U040 -> U060 -> U080 -> U100 -> D80F -> D60F ->
  D40F -> D20F -> D0F -> M1F`, plus `D100direct`.

Every fit uses 20 natural-population passes, validation every pass,
macro-AUC-first checkpoint selection, and unweighted per-jet CE. The three
declared homotopy losses are respectively `0.25/0.75/0`, `0.10/0.90/0`, and
`0.10/0.75/0.15` for CE/parent/grandparent KD; unavailable first-edge
grandparent weight transfers to the parent. Homotopy KD temperature is two.
`M1F` remains fixed at `0.25 CE + 0.75 parent KD` with temperature one.
Twenty full-data passes are intentional: at roughly eight to nine times the
300k train population, they process about three times as many jet examples as
a 60-pass 300k fit while avoiding a prohibitive 60-pass full-data bill.

The initial full-data envelopes are 8 CPUs, 256 GiB, 24 hours, and one GH200
for GPU training/target jobs; 16 CPUs, 192 GiB, 24 hours for assignment and
coupling arrays; and 4 CPUs, 64 GiB, 4 hours for reducers and reports. A
measured all-row endpoint/resource gate must stay below 75% of the locked GPU
RAM request before either shared root trains. Resource recovery may only
increase CPU, RAM, or walltime while preserving the GPU class and scientific
identity.

Operationally, one source-pinned campaign submitter registers the foundation
and one `afterok` autolaunch job. Only after the immutable foundation lock
exists does the autolaunch create and submit the three independent arm
ledgers. Foundation, arm, autolaunch, and recovery submissions resume from
immutable journals/artifacts after interruption; no accepted job is silently
resubmitted. Exact-ID cancellation, monitor/task-attestation validation, and
failed/downstream source-pinned or resource-only recovery are present.

Implementation and review evidence:

- focused HCWDL-UB/full-data/ladder suite: 46 passed;
- complete repository suite: 468 passed in 280.19 seconds, with the 14
  existing Matplotlib/Pyparsing deprecation warnings;
- the bounded synthetic all-mapped closure created a 2.6M/1M/1M inventory,
  exact 38-fit graph, foundation plan, three arm specs, three independent
  14-task dry-run ledgers, idempotent arm reuse, and an exact resource
  recovery closure;
- all 12 full-data CLI help surfaces and Python compilation passed;
- all 22 full-data/recovery contract constants are explicitly versioned;
- all three Slurm workers pass Git-Bash syntax checks and use absolute
  `${PROJECT_DIR}` activation plus `exec python -s`;
- the five-document Markdown-link audit and `git diff --check` passed.

There are no external donor files or donor commits for this block, so
`docs/LEGACY_SOURCE_MAP.md` is unchanged. No SSH, push, Slurm submission,
cancellation, final-test access, or new smoke occurred locally. The existing
production-worker evidence is reused exactly as allowed by the plan; the new
all-row endpoint/resource gate is the production preflight and cannot be
bypassed.

Exact next task: commit and push this implementation, create a clean detached
Tigris worktree at that full commit, bind the authenticated prepared 300k U/J
template, create the all-mapped foundation spec, run the nonmutating campaign
dry run, and then use the single explicit full-campaign submission command in
the runbook. That live command submits the foundation and automatically
releases all three arms after its lock; it does not submit a smoke or touch
final test.

## HCWDL unified-root balanced six-arm campaign (2026-08-15)

The implementation-authoritative
[HCWDL unified-root balanced homotopy plan](plans/HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
is now implemented locally as a new additive campaign family. It does not
mutate or relabel HCWDL-UJ. The reusable v1 semantics are frozen in the
[contract](contracts/HCWDL_UNIFIED_BALANCED_HOMOTOPY.md), and the exact
source-pinned foundation/six-arm procedure is in the
[runbook](HCWDL_UNIFIED_BALANCED_HOMOTOPY_RUNBOOK.md).

The shared 300k foundation owns exactly `U000` and `M0paired`, balanced
train/validation structural-switch sidecars, endpoint/resource gates, and one
identity-ordered FP32 U000 target cache. `U000` is the freshly initialized
unified 21-channel CE root on the exact P0 particle multiset; native TOFF is a
contextual control, not the primary root. Uniform Shell Exact moves every
matched continuous field with one exact rational offline strength and switches
discrete/validity groups with deterministic confidence-independent nested hash
variates. The original cost-CDF U schedule and confidence-warped D coordinate
remain paired controls only in `C25P75`.

Six separately authenticated 300k-train/100k-validation arms are registered:
`C25P75`, `C10P90`, `C05P95`, `C10P75G15`, `C05P80G15`, and `C00P100`.
Each has its own root, spec, command plan, ledger, monitor, cancellation
surface, recovery closure, aggregate, and completion artifact. There are no
cross-arm teachers or scheduler dependencies. The exact registry is 151 fits:
two shared fits, 34 reference-arm fits, and 23 fits in each of the other five
arms. Training GPU jobs are locked to 8 CPUs, 96 GiB RAM, six hours, and one
GH200. Every arm uses 60 natural-population passes, validation every pass,
macro-AUC-first checkpoint selection, and its declared CE/parent/grandparent
loss. M1 remains fixed at 0.25 CE plus 0.75 parent KD at temperature one.

The final audit fixed several launch-relevant issues rather than accepting
scaffolding:

- all new Slurm workers now use the absolute `${PROJECT_DIR}/sbatch/common.sh`
  activation contract and `exec python -s`;
- the foundation lock binds both shared report hashes, both selected
  checkpoint hashes, the compact U000 target manifest, and a semantically
  validated target lock;
- each arm validator reauthenticates its exact foundation, recipe arm, graph,
  waiver, project, source, and frozen semantic-source map;
- the operational waiver freezes the full model/input/training/cache/runner
  and worker surface, not only the headline U/D modules;
- partial foundation, arm, and recovery submissions resume from immutable
  journals instead of resubmitting already accepted jobs;
- finalist, execution, sealed-evaluation, and campaign-completion artifacts
  now receive semantic validation in addition to content-hash validation;
- recovery is restricted to the exact failed/downstream closure. Source and
  resource recovery remain separate, and resource recovery may only increase
  CPU, RAM, or walltime while retaining the GPU class.

Local readiness evidence at this checkpoint:

- focused HCWDL-UB/homotopy/training/worker suite: 110 passed;
- complete repository suite: 462 passed in 586.68 seconds, with the 14 existing
  Matplotlib/Pyparsing deprecation warnings plus one local pytest-cache
  permission warning;
- bounded synthetic end to end: foundation plus all six arm specs/plans, the
  real six-arm dry-run wrapper, six validated dry-run ledgers, and an exact
  resource-recovery closure/command plan all passed;
- all 17 new CLI help surfaces passed;
- all 30 new contract constants are versioned identities and appear in the
  implementation plan or reusable contract; the launch-boundary correction
  versions foundation spec and operational waiver to `/v2`;
- Python compilation, Git-Bash syntax checks for all three new Slurm workers,
  six-document Markdown-link validation, and `git diff --check` passed.

There are no external donor files or donor commits for this block, so
`docs/LEGACY_SOURCE_MAP.md` is unchanged. Existing repository HCWDL and PMARD
primitives were extended in place; the new campaign modules, scripts, workers,
contracts, runbook, and focused tests are additive. No SSH, Slurm submission,
cancellation, remote push, or final-test access occurred during this local
implementation.

Exact next task: commit and push the target-digest execution repair described
below, create a clean detached Tigris worktree at that full commit, publish one
immutable monitor and execution-repair recovery spec per old arm ledger,
preview then cancel only those six ledgers' exact IDs, dry-run all six recovery
closures, and submit the six replacement closures. The completed shared
foundation must be reused; it must not be rebuilt or edited. No additional
HCWDL-UB smoke is required or claimed.

The first Tigris launch attempt at `c05e3b7` correctly submitted no jobs but
exposed two launch-boundary defects: parent discovery required the unrelated
U/J campaign-completion report, and it named the real coupling lock as
`locks/coupling.json` instead of `locks/coupling_lock.json`. The intended
parallel experiment only requires the already authorized fixed-preprocessing
prefix. Foundation-spec v2 and operational-waiver v2 therefore authenticate
the exact coupling, endpoint-equality, and graph/recipe locks plus their
manifests without waiting for old U/J training descendants. This changes no
particle, graph, loss, seed, or resource semantics and still fails closed if
any required preparation lineage is absent or mismatched. The corrected
HCWDL-UJ/UB/recovery focus passes 86/86; the complete repository suite passes
463/463 in 204.10 seconds with one existing PRAD tensor warning and the local
pytest-cache permission warning. Python compilation, the revised waiver CLI,
and `git diff --check` also pass.

### HCWDL-UB U000 target-manifest digest execution repair (2026-08-15)

The first six-arm launch proved that all shared preparation and training
completed successfully through foundation lock `86841` and autolaunch `86842`.
Every U000-supervised arm root then failed before optimizer training with
`HCWDL-UB shared target manifest is not foundation-locked`, after unnecessarily
building its 300k/100k RAM views. The shared foundation artifacts authenticate
independently. Tigris reported:

- target manifest content hash:
  `5de8bde03d6813a686e9e0042d589537e6f0e1be5095d11193f86f73ed93aa68`;
- U000 training-report hash:
  `e4a6296e551d369be6d06ecb1bdf270251d4b5eb6545898841322186c883ddc3`;
- the foundation and target lock both recorded the latter report hash as the
  manifest hash;
- U000 report/checkpoint and target-lock/foundation-lock parent hashes matched.

Root cause was a local-variable shadow in `validate_target_manifest()`: after
authenticating the manifest, its parent-validation loop overwrote `digest` and
returned the final sorted parent digest, which was the teacher-report hash.
The manifest, FP32 logits, U000 checkpoint, U000 report, M0paired result,
balanced couplings, and endpoints are not corrupt and must be reused.

The validator now returns a dedicated `manifest_digest`. The arm runner
preflights shared-U000 lineage before constructing any RAM view. A distinct
`HCWDL_UNIFIED_BALANCED_EXECUTION_REPAIR_RECOVERY_SPEC/v1` path reconstructs
and binds the exact legacy defect, requires an explicit repair phrase, and
accepts a changed semantic-source map only when the changed files are exactly
the target validator and arm runner. It never rewrites the old locks. Ordinary
execution/source/resource recovery still rejects the mismatch. Recovery
reports bind the recovery spec, actual target-manifest hash, and repair-evidence
hash.

Local verification for this repair: focused HCWDL-UB tests pass 28/28; the
complete repository suite passes 464/464 in 238.65 seconds with the 14 existing
Matplotlib/Pyparsing deprecation warnings; five affected CLI help surfaces,
Python compilation, versioned-contract documentation, and `git diff --check`
pass. No SSH, Slurm cancellation/submission, lock mutation, or final-test access
occurred locally.

## HCWDL-UJ combined execution/resource recovery (2026-08-14)

The linear endpoint-preparation correction changes semantic-source bytes while
the operator also requested that replacement training jobs use 16 rather than
8 CPUs. The pre-existing source-recovery v2 contract preserves resources, and
the resource-only v2 contract preserves source, so neither may be repurposed
or bypassed. A distinct
`HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_{SPEC,COMMAND_PLAN}/v1`
family now binds both reviewed changes without changing the original campaign's
scientific identity.

The combined recovery authenticates the exact cancelled/failed downstream
closure, original ledger and monitor, complete old/new semantic-source maps,
execution-only human classification, and complete old/replacement resource
maps. Resources may only increase; the GPU type cannot change; at least one
resource class used by the closure must increase. The intended 300k recovery
changes only `gpu_training.cpus` from 8 to 16 and retains 96G, six hours, and
one GH200. A dedicated worker environment and distinct authorization and
submission phrases prevent old v2 recovery artifacts from being interpreted
under the combined semantics.

Local evidence at this checkpoint: the focused recovery/resource/worker suite
passes 7/7; the broader HCWDL/stream/repair/cache suite passes 185/185; and the
complete repository suite passes 436/436 with the 14 existing
Matplotlib/Pyparsing warnings. CLI help and `git diff --check` pass. Tigris
accepted corrected commit `f698889`, cancelled only the exact old-ledger IDs,
authenticated a 41-task failed closure, verified 39 training commands at 16
CPUs/96G/six hours/one GH200, and published the new live recovery ledger under
`recovery/linear_16cpu_f6988892_r1/live`. Corrected phase timing and completed
training evidence remain pending.

The reusable diagnosis and implementation guidance is in
[HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md](HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md).

## HCWDL-UJ linear endpoint preparation and ordered CPU overlap (2026-08-14)

Live 300k recovery jobs `84176`, `84177`, `84179`, `84182`, `84193`, and
`84204` each remained before their first validation checkpoint after roughly
four hours, while HLT-only job `84211` reached update 38,676/70,320 in about
one hour.  Slurm accounting showed approximately one CPU-hour per elapsed
hour for every blocked job.  This excludes ordinary optimizer training and
the already locked matcher/coupling production as the dominant cost.

The concrete defect was repeated whole-chunk ragged conversion.  The public
one-row offline projector converts every required Awkward branch into all
rows.  P0, D100, U, J, calibration, and audit callers invoked that projector
inside a row loop over a 4,096-row chunk; U/J partition construction also
reconverted every raw HLT branch per row.  The resulting work was quadratic
in chunk population.  Cache allocation itself was already preallocated and
linear.

The execution repair leaves `repair.py`, Shell Exact v1, coupling artifacts,
coordinates, graph, loss, inputs, endpoints, and checkpoint selection
unchanged.  `hcwdl_homotopy.py` now prepares every offline and HLT branch once
per chunk, validates the same count/length rules, and reuses exact raw feature,
validity, and float32-to-float64 p4 endpoints across P0, Shell, partition, and
carrier construction.  Coupling calibration/audit reuse the same prepared
chunk.  The homotopy stream bounds one in-flight chunk per worker, overlaps
chunk construction with a deterministic thread pool sized from
`SLURM_CPUS_PER_TASK`, and always consumes futures in canonical submission
order.  `HCWDL_UJ_VIEW_BUILD_WORKERS` may request fewer workers but is rejected
if it exceeds the Slurm allocation.  Workers now print explicit train-cache,
validation-cache, teacher-target, and optimizer-training phase boundaries.

Local evidence:

- pre-change focused baseline: 77/77 passed;
- post-change endpoint/homotopy/cache focus: 80/80 passed;
- broader HCWDL/CLI/contracts/smoke/repair/cache focus: 168/168 passed;
- complete repository suite: 435/435 passed with the 14 existing
  Matplotlib/Pyparsing warnings;
- exact prepared-versus-legacy raw feature, validity, and p4 arrays are locked
  across a seven-row fixture, and every required branch is converted exactly
  once rather than once per row;
- ordered one-worker versus four-worker emission is locked by regression;
- synthetic P0 preparation at 1,024 rows improved from 4.837 s to 0.364 s
  (13.30x) with exact p4 equality; because the removed term was quadratic,
  the production 4,096-row chunks should benefit more;
- Python compilation and `git diff --check` pass at this checkpoint.

This is local execution evidence, not Tigris acceptance.  The running jobs
remain pinned to the old source and cannot acquire this fix.  After exact-ID
closure handling, the corrected commit requires the existing human-authorized
source-recovery path; completed reports and compatible rolling checkpoints
remain reusable. The eight requested CPUs are now available to bounded
concurrent chunk construction; raising the request above eight should wait for
the corrected Tigris timing rather than masking the eliminated algorithmic
defect.

## HCWDL-UJ exhaustive-audit source parallelism repair (2026-08-14)

The 300k v2 coupling audit timed out twice: original job `83434` exhausted its
four-hour request, and resource-recovery job `83884` exhausted twelve hours.
This is an operational implementation failure, not a failed coupling
invariant or a scientific result. The audit task requested eight CPUs but
`audit_full_roles()` executed the entire 300,000-train plus
100,000-validation source population in one Python process. For every jet it
also accumulated all twenty structural thresholds, reconstructed and compared
the P0/D100/HLT endpoints, rebuilt the residual cost matrix, reran the exact
Hungarian optimum check, and later performed the independent ROOT reread. The
extra walltime therefore left seven requested CPUs unused by the dominant
loop.

The exhaustive audit is now partitioned into one process task per
authenticated train or validation source unit, using up to
`SLURM_CPUS_PER_TASK` workers. Every source still performs the complete row,
endpoint, conservation, transition, and solver-optimum checks. Results are
reduced with exact integer addition in canonical train-then-validation source
order; endpoint and solver proofs use domain-separated, length-framed
per-source hashes. Independent sample rereads use the same source partition.
Completion order and worker count are explicitly excluded from scientific
identity, while the reduction scheme is recorded in the audit and validated
fail-closed. The frozen single-process implementation remains private as a
reference for future Tigris parity audits. No coupling, switch, endpoint,
graph, loss, or dataset semantics changed, and no contract version was
bumped.

Self-review also found that the pre-existing source-recovery worker would have
rejected any corrected file listed in the campaign's semantic-source map,
including this execution-only repair. Source recovery now preserves that
original map inside the unchanged scientific identity and separately binds
the complete corrected execution-source map, the exact old/new hash pair for
every changed semantic file, and an explicit human-authorized execution-only
classification. Both the recovery validator and worker checkout validate
that second map; resource-only recovery continues to require byte-identical
campaign source. This closes the recovery path without disguising changed
source bytes as the original campaign source.

Local evidence for this repair:

- focused HCWDL homotopy and CLI suite: 114/114 passed before the final
  execution-lineage regression, followed by 4/4 targeted reducer,
  process-pool, and validator tests;
- complete repository suite: 432/432 passed with the 14 existing
  Matplotlib/Pyparsing warnings plus one non-scientific pytest-cache warning;
- changed-source AST parsing, four recovery/task CLI help surfaces, v2 graph,
  pilot, and recovery contract identity checks, and `git diff --check` passed;
- no donor code or donor commit was used; unrelated user worktrees, figures,
  and plotting files remain untouched.

The current resource-recovery ledger must be monitored immutably and cancelled
by its exact IDs before a source-pinned failed-closure recovery is submitted.
That recovery reuses the completed calibrations, shards, manifests, switches,
and TOFF targets, retains the twelve-hour CPU-coupling envelope, and starts at
`coupling_audit`; no earlier data work or completed artifact is recomputed.
Real Tigris validation of the parallel audit remains the next required runtime
evidence.

## HCWDL architecture-factorial measured P0 walltime (2026-08-12)

The first 300k architecture-factorial pilot measured a preprocessing-bound
failure: `O_U` job `83191` received Slurm `SIGUSR1` at `05:57:39` under its
six-hour request, with exit `0:10`, empty Python stderr, peak RSS about 24 GiB,
and no `rolling_resume.pt`. It had not reached the first optimizer update; it
was still projecting the 300k train plus 100k validation native-offline P0
population and building the process-local RAM caches. This is neither OOM nor
a model/scientific failure, and a same-resource requeue would repeat the work.

The replacement campaign policy therefore keeps exact-HLT cells at 8 CPUs,
96 GiB, six hours, and one GH200, while P0 cells use the same CPU/RAM/GPU with
a sixteen-hour walltime. Tasks bind explicit `training_hlt` or `training_p0`
resource classes from their registered input domain; tests assert all four
exact command rows. The four-cell graph, inputs, seeds, losses, schedule, and
reports are unchanged. The old campaign must be cancelled only through the
exact IDs in its immutable submission ledger, and the replacement must use a
fresh source-pinned root.

## HCWDL-UJ canonical pilot-parent role-count repair (2026-08-12)

The first waived v2 300k dry-run attempt stopped safely before campaign
publication or submission with `HCWDL-UJ parent role counts differ`. The
canonical HCWDL 300k parent correctly records 300,000 train, 100,000
validation, and 100,000 sealed final-test rows; the validation-only U/J child
correctly records 300,000 train, 100,000 validation, and zero final-test rows.
`authenticate_parent()` had incorrectly compared the parent against the child
projection. It now authenticates the parent with its own HCWDL mode contract
and retains the existing child-side zero-final-test checks. A regression locks
both populations explicitly. The already published waiver and source parity
cannot be reused because this validator is semantic source; the corrected
commit requires new parity and waiver identities. The partial v2 smoke exact
IDs were cancelled only after its completed infrastructure evidence had been
bound into the old waiver, and no 300k jobs were submitted.

Local compilation and `git diff --check` pass. Focused pytest cannot collect
in the current Windows Python because NumPy is not installed; the pinned
Tigris environment must run the focused parent-population and waiver tests
before constructing the corrected dry run.

That corrected attempt then reached the next independent validator boundary
and stopped before publication with `dense D0c control has different HCWDL
parent lineage`. The implementation plan requires the selected coarse cold
`D0c` from the primary HCWDL graph; dense10/dense5 reports are separate,
possibly empty contextual imports. The canonical primary report does not and
should not carry the dense-supplement `parent_campaign_spec_sha256` field.
The importer now authenticates canonical path, primary graph/node payload,
unweighted recipe, split and source snapshot, assignment and qualification
locks, sole D25c teacher report, wrapper/engine equality, and checkpoint
bytes through `validate_completed_hcwdl_node()`. A focused regression locks
that schema. The second failed dry run again published no campaign and
submitted no 300k jobs.

## HCWDL-UJ v1 operational-evidence carry-forward (2026-08-12)

The human rejected repeating all graph-thinned v2 smoke fits after the
completed 80-fit v1 production-worker smoke. That decision is now represented
honestly by `HCWDL_STRUCTURAL_FEATURE_OPERATIONAL_EVIDENCE_WAIVER/v1`, not by
relabeling the partial v2 run as complete. The waiver binds the v1 campaign
and completion, completed v2 installed-Weaver parity, coupling/endpoint/TOFF
target/graph locks, the exact new source and semantic hashes, the 8 CPU/96G/
six-hour/GH200 pilot request, and the exact human authorization phrase. It
does not change any scientific graph, loss, dataset, endpoint, or final-test
rule. A completed measured v2 resource profile remains an alternative, not a
mandatory gate when this explicit waiver is present.

No jobs were cancelled or submitted by the implementation. The operator may
cancel only the exact partial-v2 smoke ledger IDs after publishing the waiver,
then create/dry-run/submit a new source-pinned 300k root. The two unrelated
worktree entries remain untouched.

## HCWDL-UJ reduced v2 graph (2026-08-12)

After the successful 80-fit v1 production-worker smoke, the user selected a
smaller primary screen before any 300k U/J submission. The implemented v2
graph retains ten equal nominal-L1 predecessor-KD transitions per primary
path: factorized `U020..U100 -> D80F..D0F -> M1F`, and joint
`J010..J100 -> M1J`. Its controls are five stationary D100 fits, ten
temperature-2 stationary HLT fits plus temperature-1 `S0_11`, and the same
seven direct/adapter controls with `U020P0KD`. The exact registry is 45 fits
and 66 total tasks. The completed v1 smoke remains immutable historical
evidence and is not relabeled as v2 acceptance.

Graph-bound coordinate, node, graph, recipe, graph-lock, campaign, command,
aggregate, completion, and both recovery families now use explicit `/v2`
contract identities. Coupling, endpoint, view, target, node-wrapper/runtime,
resource, and resume contracts remain v1 because their underlying semantics
did not change. Cache miniature keys, full-role displacement coordinates,
TOFF target consumers, seed aliases, temperature routing, dependencies,
report comparisons, and same-input metadata all derive from the reduced v2
registry. The earlier comparison-report bug that marked several exact-input
controls `false` is fixed by comparing authenticated input-domain signatures.

The requested 300k `gpu_training` row is enforced in campaign creation and
validation as exactly 8 CPUs, 96 GiB, six hours, and one GH200. The v2 smoke
must still demonstrate 25% resource headroom before its profile can authorize
the pilot. No Slurm job was submitted by this change.

Final local evidence from the implementation tree:

- focused homotopy contracts/graph/campaign/reporting: 45/45 passed;
- complete HCWDL/CLI focus: 107/107 passed;
- complete repository suite: 423/423 passed with the 14 existing
  Matplotlib/Pyparsing warnings;
- bounded synthetic end to end: all 45 fits completed, no final-test access;
- 13/13 changed graph-bound v2 contract identities and 7/7 primary CLI help
  surfaces passed;
- Python compilation and `git diff --check` passed (line-ending notices only).

The existing unrelated `.worktrees/hcwdl-rkd-local` modification and
`.worktrees/hcwdl-u-rkd-local/` directory remain untouched. No donor code or
donor commit was used. The next acceptance layer is one clean, pushed,
source-pinned v2 Tigris smoke using the already authenticated HCWDL parent;
only after completion, measured resource profiling, and a complete dry run may
the user separately authorize the 300k pilot.

## HCWDL architecture-factorial BF16 repair (2026-08-11)

The first genuine Tigris architecture-factorial smoke at source `c2a5510d`
authenticated its parent and passed architecture-check job `82058`. Unified
cells `H_U` (`82059`) and `O_U` (`82061`) completed, while split cells `H_S`
(`82060`) and `O_S` (`82062`) failed before producing scientific results. In
production BF16 autocast, each Weaver encoder returned a BF16 embedding while
`SplitScoutingParticleTransformer._encode_nonempty()` allocated its scatter
destination from the FP32 cached input. PyTorch correctly rejected the
cross-dtype indexed assignment. This is an implementation-only failure; it
does not implicate either authenticated input view or the factorial design.

The split encoder now allocates from the encoded embedding and uses
differentiable `index_copy`, preserving the active autocast dtype and gradient
flow. The architecture preflight now executes both models under the same BF16
autocast policy as training and records input, autocast, and output dtypes. A
focused regression exercises the split model's FP32-cache/BF16-output path.
Source compilation and `git diff --check` pass. The local Windows Python does
not provide PyTorch, so focused/full pytest cannot run in this checkout; the
replacement source-pinned Tigris smoke must first pass its strengthened GPU
preflight and then all four two-update cells. The completed old unified cells
are not mixed with repaired-source cells: the scientifically clean smoke is a
new campaign root, and the old failed campaign's aggregate/completion jobs may
be cancelled only by their exact IDs `82063` and `82064`.

## HCWDL common-schema architecture–input factorial (2026-08-11)

The validation-only four-way architecture/input ablation requested after the
HCWDL-UJ smoke is implemented. It registers four fresh, paired, unweighted
CE-only cells: exact HLT and projected-native P0 inputs under either the
canonical unified 21-channel ParT or a new common-schema charged/noncharged
two-stream ParT. The split control retains every visible token exactly once,
routes unknown/unclassified tokens to the noncharged stream, uses two full
eight-block 21-channel encoders plus the canonical TOFF fusion topology, and
handles an absent stream without inventing a particle. Native 19/7 `TOFF`
remains a fifth imported reference and is never relabeled as a factorial cell.

The separate `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_*/v1` family implements the
exact four-cell graph, content-hashed parent/source/spec/command lineage,
process-local one-time HLT/P0 caches, shared sampler and within-architecture
initialization seeds, 60-pass/every-pass validation, macro-AUC checkpoint
selection, all five prespecified factorial effects, runtime parameter-count
disclosure, zero final-test rows, journaled submission, immutable task
attestations, and generic rolling-checkpoint requeue recovery. The Slurm
training request is exactly 8 CPUs, 96 GiB, six hours, and one GH200. Campaign
creation and execution are thin CLIs; no job was submitted locally.

Focused architecture/graph/campaign tests pass 5/5. The complete repository
suite passes 423/423 with the 14 existing Matplotlib/Pyparsing warnings. All
three new CLI help surfaces pass, the eight contract identities are asserted,
Markdown links resolve through the complete suite, and `git diff --check`
passes with Windows line-ending notices only. The next acceptance layer is a
real source-pinned Tigris smoke using the completed authenticated unweighted
HCWDL smoke parent. No donor code or donor commit was used.

## HCWDL-UJ first Tigris dry-run repair (2026-08-11)

The authenticated unweighted parent smoke completed through aggregate job
`80916`, and installed-Weaver CUDA FP32 parity passed for source
`c70a34e4edb515a9636110ad920dd21ff84d2e07`.  The first HCWDL-UJ campaign
creation then completed, but its immediate dry-run validation failed closed
before any of the 101 campaign jobs were submitted.  The immutable graph JSON
reloaded each node's `teachers` tuple as a JSON list, while
`HomotopyNodeSpec.payload()` regenerated a Python tuple; the validator
therefore rejected its own byte-valid artifact as
`HCWDL-UJ immutable local campaign artifacts differ`.

`HomotopyNodeSpec.payload()` now emits a JSON-native teacher list before
hashing or publication.  This does not change the canonical graph hash or any
scientific graph semantics because tuples and lists had the same canonical
JSON encoding; it makes the in-memory contract representation agree with its
immutable reloaded representation.  A regression checks JSON round-trip
identity for all 80 node payloads.  A dependency-isolated execution of the
actual graph module confirms all 80 payloads round-trip exactly, source/test
compilation passes, and `git diff --check` passes.  The local Windows Python
lacks NumPy, so the focused pytest module cannot be collected locally; the
focused and complete suites must be rerun in `atlas_kd_tigris` on the pushed
repair before the smoke is relaunched.  Failed launcher jobs `80998`, `81219`,
`81220`, and `81221` submitted no HCWDL-UJ campaign tasks and do not alter the
completed parent.

## HCWDL dense measured-resource reschedule (2026-08-11)

The pending corrected dense10 and dense5 recovery roots inherited the pilot's
conservative `gpu_single` request of `320G` for 72 hours. Tigris jobs `77534`
and `77546` provide direct completed-rung evidence: D90c/D95c finished in
8,983/9,187 seconds with peak RSS 50,285,376/50,287,104 KiB. The new
`HCWDL_DENSE_RESCHEDULE_SPEC/v1` therefore freezes a right-sized `96G`,
six-hour, eight-CPU, one-GH200 request while leaving the original CPU aggregate
request unchanged.

The reschedule is operational only. It binds the exact prior recovery spec and
live submission ledger, retains the original dense scientific spec, failed
closure, graph, recipe, assignments, teachers, seeds, output root, and rolling
checkpoints, and records the prior ledger's exact IDs as superseded. Arbitrary
resource edits and recursive resizing fail closed. Operators create and
dry-run both replacements before cancelling either old ledger, cancel only
through `cancel_hcwdl_campaign.py`, confirm removal, and then submit the
right-sized closures. No local or remote Slurm job was mutated during this
implementation.

Implementation is in `hcwdl_dense_recovery.py` with the thin
`create_hcwdl_dense_reschedule.py` surface and the existing authenticated
submitter/worker. Both dense runbooks and the recovery contract document the
new path. Focused dense/recovery/CLI tests pass 71/71; the complete repository
suite passes 415/415 with the 14 existing Matplotlib/Pyparsing warnings; both
CLI help surfaces and `git diff --check` pass (Windows line-ending notices
only). No donor code or donor commit was used.

## HCWDL structural-feature homotopy implementation (2026-08-11)

The validation-only structural/feature homotopy in
[`docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`](plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md)
is now implemented locally and ready for its required real Tigris smoke. This
supersedes the earlier documentation-only status below; it does not mutate or
relabel any completed primary HCWDL, dense10, dense5, or recovery artifact.

Reusable implementation was added under `src/hlt_classification/scouting/`:
the exact 80-fit graph and recipe overlay; typed P0/D100 endpoint partition;
label-free maximum-cardinality minimum-cost residual coupling; immutable base
shards, switch sidecars, manifests, audits, and locks; the generalized
`V(s,f)` carrier; dedicated homotopy streaming; durable authenticated TOFF
targets; process-local student/teacher caches; per-node training; campaign,
workflow, reporting, source recovery, resource-only recovery, and bounded
synthetic-smoke modules. The existing all-21-field projector is exposed from
`repair.py` without changing Shell Exact v1. Generic PMARD checkpoint
publication now accepts only semantically identical interrupted retries, and
HCWDL training accepts an explicitly validated per-node loss resolver while
retaining the old default behavior.

Thin command surfaces were added for coupling calibration/build/finalization,
TOFF targets, campaign creation/submission/workers, monitoring, exact-ID
cancellation, recovery, resource profiling, installed-Weaver parity, and the
local smoke. Three Slurm workers use the pinned absolute `PROJECT_DIR`, Tigris
environment rules, `exec python -s`, and `USR1@120` on checkpointable GPU
jobs. The smoke/pilot command plan has 101 tasks: 21 infrastructure/reporting
tasks and all 80 fits. Train/validation roles are 4,096/4,096 for smoke and
300,000/100,000 for pilot; both register zero final-test rows and no final-test
task.

The reusable contract index is
[`docs/contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md`](contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md).
It records 36 unique v1 identities covering coupling/config/calibration/cache,
TOFF targets, coordinates/graph/recipe/locks, node/runtime/aggregate/resource
reports, installed-Weaver parity, completion, and both recovery families.
The operator procedure is
[`docs/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md`](HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_RUNBOOK.md).
No donor or exploratory file was copied, so there is no donor commit or new
`LEGACY_SOURCE_MAP` entry. The pre-existing unrelated modification at
`.worktrees/hcwdl-rkd-local` was left untouched.

Local evidence from the final tree:

- focused homotopy plus CLI tests: 101/101 passed;
- cross-module HCWDL/PMARD/Shell Exact tests: 144/144 passed before the final
  parity-only addition;
- complete repository suite: 412/412 passed in 142.47 seconds, with one
  unrelated existing PyTorch warning;
- bounded synthetic graph: all 80 cold-start fits completed two updates,
  every loss was finite, teacher-own-domain routing was exercised, and
  `final_test_accessed` remained false;
- 14/14 new CLI help paths, three/three Slurm shell syntax checks, Python AST
  parsing, all documentation links, 36/36 exact contract identities, and
  `git diff --check` passed (Windows line-ending notices only).

Local installed-Weaver runtime parity cannot be claimed because the Windows
development environment does not contain Weaver. The exact next acceptance
step is therefore the runbook's pinned Tigris GPU parity command followed by
the genuine 4,096/4,096 production-worker smoke. No Slurm job was submitted.
After the smoke completes, its exact ledger/monitor/runtime reports must be
converted into the measured resource profile; only then may the runbook emit
the nonmutating 300k dry run and request separate live-pilot authorization.

## HCWDL completed-node recovery idempotency (2026-08-10)

The source-pinned RNG recoveries retrained already-completed dense prefix nodes
(`D90c` in dense10 and `D95c` in dense5). Training finished successfully, but
publication then correctly rejected the newly serialized `final_model.pt`
bytes because an immutable final checkpoint already existed. This was a
recovery orchestration defect, not a model-training or scientific failure.

Primary and dense HCWDL workflows now authenticate and reuse a completed node
before constructing views or launching training. Reuse requires both the PMARD
engine report and HCWDL node report, valid content hashes, exact campaign,
graph, recipe, node, teacher, lock, split, source, and warm-parent lineage, and
byte-identical selected and final checkpoints. A partial, stale, corrupt, or
lineage-mismatched node fails closed. No checkpoint is overwritten, and no
matching, repair, model, loss, seed, schedule, or graph semantics changed.

Focused dense/primary/recovery/resume tests pass 65/65. The complete repository
suite passes 355/355 with the same 14 Matplotlib/Pyparsing warnings, and
`git diff --check` passes apart from Windows line-ending notices. Real Tigris
validation remains: submit a new source-pinned dense recovery from the pushed
fix; it should reuse D90c/D95c immediately and begin D80c/D90c respectively.

## HCWDL structural-feature homotopy implementation plan (2026-08-10)

The new implementation-authoritative supplemental plan is
[`docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`](plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md).
It specifies the missing upper bridge from native-offline support to exact
D100 support and an edge/update-depth-matched comparison between a factorized
`U then D` path and a joint `U+D` path. It is additive: completed primary,
dense10, dense5, and recovery artifacts retain their original contracts.

The planning audit found and closed a critical representation ambiguity.
Canonical TOFF is a two-stream 19/7-feature, 90/60-particle oracle, while all
U/J/D students use the unified 21-channel ParT. The plan therefore defines P0
as the exact TOFF-visible particle multiset under the existing 21-field
projection without claiming tensor/adapter identity. It then freezes typed
A/B/K/C_A/C_B/O/R endpoint records, a label-free maximum-cardinality
minimum-cost residual coupling, atomic substitution/removal/insertion edits,
an exact integer train-calibrated switch coordinate, a maximum-200 carrier,
and one `V(s,f)` builder. U100 must be byte-identical to current D100;
factorized D0 and joint J100 must be byte-identical to HLT, including canonical
raw-length metadata.

The factorized and joint paths each contain 20 cold predecessor-KD transitions
plus one born-again M1, use the locked unweighted 60-pass recipe, and are paired
by transition seed. The exact 80-fit registry includes P0 adapter diagnostics,
paired/direct endpoints, a ten-generation D100 chain, and a 21-generation HLT
stationary-depth chain so path gains are not confused with repeated
distillation depth. The initial 300k study is validation-only and structurally
has no final-test task. This step is documentation only: no coupling, graph,
worker, artifact, Tigris submission, or remote mutation was implemented or
authorized. After the independent science/implementation/editorial audit, the
focused high-coverage/HCWDL suite passed 60/60 in the local `tagging-hlt`
environment. The complete suite passed 355/355 with the 14 existing
Matplotlib/Pyparsing warnings plus one local pytest-cache permission warning;
`git diff --check` passed apart from line-ending notices.

## Primary HCWDL source-pinned closure recovery (2026-08-10)

The unweighted 1M campaign completed both D50 branches, then `D25c` job
`60508` and `D25w` job `60518` failed while restoring CUDA RNG byte states
through the old source-pinned checkout. The shared RNG implementation is
already repaired, but the ordinary HCWDL resumer intentionally cannot execute
an immutable parent spec through a different commit.

`HCWDL_FAILED_CLOSURE_RECOVERY_SPEC/v1` now provides the missing proper path.
It authenticates the complete original ledger and immutable Slurm monitor,
forms the exact union of failed descendants, requires every external parent to
be completed and artifact-valid, preserves the original scientific spec and
output root, and executes through a clean pushed repair worktree. For the 1M
failure this reuses D50c/D50w, resumes D25c/D25w independently from their
rolling checkpoints, and continues only their cold/warm descendants and
shared downstream stages. Separate recovery attestations and exact-ID
supersession lineage are retained.

The implementation is in `hcwdl_campaign_recovery.py` with thin create,
submit, worker, and Slurm entry points. The operational contract and commands
are documented in `docs/contracts/HCWDL_CAMPAIGN_RECOVERY.md` and
`docs/HCWDL_CAMPAIGN_RECOVERY_RUNBOOK.md`. Focused recovery/contract/dense
tests pass 33/33, all HCWDL CLI tests pass 44/44, and the complete repository
suite passes 352/352 with the same 14 Matplotlib/Pyparsing warnings. The repair
still requires an exact pushed commit and a real Tigris recovery run.

## GPU-mapped exact-resume RNG repair (2026-08-10)

The live ten-point dense recovery completed D90, then D80 job `77535` exposed
a shared exact-resume defect after a rolling checkpoint was loaded with
`map_location=cuda`. That mapping correctly moved model and optimizer tensors,
but it also moved the serialized CUDA-generator byte states onto CUDA;
`torch.cuda.set_rng_state_all()` requires CPU `ByteTensor` state even for CUDA
generators. This was an execution-only resume failure and did not alter dense
matching, repair, loss, model, sampler, or graph semantics.

Shared RNG restoration now validates every CPU and CUDA state as a
one-dimensional `uint8` tensor, normalizes it to contiguous CPU storage before
any generator is mutated, validates the CUDA device count, and only then
restores all generators. Existing rolling checkpoints and checkpoint contracts
remain compatible, so D80 can resume rather than restart. Regressions simulate
a device-mapped tensor without requiring a local GPU and reject malformed
dtype/shape before generator mutation.

Focused training/resume, PMARD, HCWDL ladder, and dense tests pass 48/48. The
complete repository suite passes 347/347 with the same 14
Matplotlib/Pyparsing warnings. The remaining runtime action is an exact pushed
commit followed by a new source-pinned dense recovery; the completed D90 report
and D80 rolling checkpoint must be preserved.

## HCWDL continuous Shell Exact and dense-closure repair (2026-08-10)

The live ten-point D90 and five-point D95 workers exposed a shared runtime
guard that still restricted every repair family to PMARD's seven-value alpha
grid. `HIGHCOV_SHELL_EXACT/v1` now accepts its declared continuous finite
coordinate on `[0,1]`; other repair families retain the legacy grid. Direct
regressions construct D90 and D95 confidence-warped views and preserve endpoint
and range checks.

Because the failed Slurm jobs are immutably pinned to the broken commit,
ordinary requeue is not valid. `HCWDL_DENSE_RECOVERY_SPEC/v1` binds each
complete original ledger and monitor report, authenticates the completed
D100offkd prefix, and submits only the failed/downstream closure from a clean
repair checkout. It writes into the original scientific campaign root under
the original graph while keeping separate recovery attestations and never
accessing final test. The exact contract is
`docs/contracts/HCWDL_DENSE_RECOVERY.md`.

Focused dense, matching, repair, and CLI validation passes 76/76. The complete
repository suite passes 344/344 with the same 14 Matplotlib/Pyparsing warnings;
Python compilation and `git diff --check` pass. The real ten-point and
five-point Tigris closures remain to be submitted from the exact pushed repair
commit.

## HCWDL interrupted sealed-final repair (2026-08-10)

The 300k unweighted pilot's sealed evaluation failed after its one-time claim
because `load_pmard_model()` restored checkpoint storage onto CUDA but left the
newly constructed ParT module on CPU. The loader now explicitly moves the
restored module to the requested device.

The repair does not reset or bypass the final-test seal. Exact-claim validation
and recovery were added alongside `HCWDL_FINAL_EVALUATION_MANIFEST/v2`, which
validates and reuses completed per-finalist reports and evaluates only missing
ones after an interruption. A separate source-pinned
`HCWDL_FINAL_RECOVERY_SPEC/v1` registers only sealed evaluation and aggregation,
binds the original failed job through an immutable monitor report, and forbids
new selection. The contract and operator procedure are in
`docs/contracts/HCWDL_FINAL_EVALUATION_RECOVERY.md` and
`docs/HCWDL_FINAL_RECOVERY_RUNBOOK.md`.

Focused final-recovery, HCWDL-contract, and CLI validation passes 65/65. The
complete repository suite passes 337/337 with the same 14
Matplotlib/Pyparsing warnings; Python compilation and `git diff --check` pass.
The real Tigris recovery remains to be run from the exact pushed repair commit.

## HCWDL five-point dense cold 300k supplement (2026-08-09)

The dense cold implementation now also registers an isolated five-point
screen: imported TOFF teaches D100offkd, then fresh D95/D90/.../D5/D0 and one
fresh M1. This is 22 new sequential GPU training nodes plus one CPU aggregate.
It retains the locked unweighted 0.25 CE plus 0.75 sole-teacher KD recipe,
temperature 2 for privileged rungs and temperature 1 for M1, 60 passes,
per-pass validation, macro-AUC-first checkpoint selection, and the shared
nested Shell Exact repair coordinate. It reuses the completed unweighted 300k
pilot read-only and has no final-test task.

The new scientific identities are the `HCWDL_DENSE5_* /v1` graph, node,
specification, command-plan, training-report, and aggregate contracts. Its
Slurm jobs use `hcddp5_*`, so it can run beside the primary campaign and the
ten-point supplement without artifact or scheduler-name collision. The same
thin CLIs dispatch by immutable spec contract; creation requires
`--rung-step 5` and separate exact authorization phrases.

This extension audit also fixed a latent generic-training bug before the
ten-point ladder reached D90: custom dense domain registries are now passed
through training configuration rather than looking up only the original
25-point HCWDL domains. D90 and D95 input selection are regression tested.
The exact contract and launch procedure are in
`docs/contracts/HCWDL_DENSE5_COLD_PILOT.md` and
`docs/HCWDL_DENSE5_COLD_RUNBOOK.md`. Focused HCWDL/dense/high-coverage/repair
validation passes 118/118; the complete repository suite passes 330/330 with
the same 14 Matplotlib/Pyparsing warnings. CLI help, Python compilation, and
`git diff --check` pass. No Tigris jobs were submitted by this implementation.

## HCWDL dense cold 300k supplement (2026-08-09)

The new isolated `HCWDL_DENSE_COLD_GRAPH/v1` implements the requested
validation-only 300k descent: imported TOFF teaches fresh D100offkd, followed
by fresh D90/D80/.../D10/D0 and one fresh born-again M1. Every D node uses the
locked unweighted 0.25 CE plus 0.75 KD recipe at temperature 2; M1 uses the
same weights at HLT temperature 1. All nodes train for 60 passes, validate
every pass, and select macro-AUC first. M2--M6 and warm nodes are absent.

The supplement authenticates and directly reuses the completed pilot's split,
row selection, recipe, train/validation assignment stores, assignment and
Shell Exact locks, and M0/D100/TOFF reports. It does not rematch, copy repaired
datasets, or register final-test access. Twelve sequential checkpointable GPU
jobs feed one CPU aggregate.

The implementation audit found that the original coarse runner used a
domain-derived discrete repair seed. Individual coarse views remain valid,
but their discrete switches were not strictly nested across alpha. The dense
runner explicitly uses one shared seed across every D domain, making the new
10-point descent a monotone discrete repair trajectory. The old campaign is
retained as the coarse empirical control and is not mutated or relabeled.

Implementation lives in `hcwdl_dense.py`, `hcwdl_dense_runner.py`, and
`hcwdl_dense_workflow.py`, with three thin CLIs and
`sbatch/run_hcwdl_dense_task.sh`. The exact scientific contract and Tigris
commands are in `docs/contracts/HCWDL_DENSE_COLD_PILOT.md` and
`docs/HCWDL_DENSE_COLD_RUNBOOK.md`. Focused HCWDL/dense/repair validation
passes 95/95; the complete repository suite passes 329/329 with the same 14
Matplotlib/Pyparsing warnings; Python compilation and `git diff --check` pass.
No remote job was submitted by this change. The remaining runtime boundary is
an exact pushed clean commit and execution of the creation/dry-run/submission
sequence in the dense runbook against the completed unweighted pilot root.

## HCWDL unweighted primary and automatic continuation (2026-08-09)

The primary HCWDL recipe is now `HCWDL_RECIPE/v4`: qualifiers, privileged
teachers, cold/warm ladder nodes, controls, and confirmation runs use
unweighted natural-population per-jet loss through an authenticated vector of
fifteen exact ones. The deterministic train selection and all class counts
remain bound. Earlier square-root-weighted campaigns are ablations and cannot
be resumed as primary parents.

All six endpoint qualifiers share one stochastic trajectory seed, pairing
initialization, sampler order, repair randomness, dropout, and training RNG
across views. Campaign/submission contracts advance to v7 and the command plan
to v3. In `preauthorized_automatic` mode the full `afterok` DAG is queued once;
the endpoint gate creates a lineage-bound nonselecting waiver only after all
six finite reports and endpoint invariants validate. Poor finite science
continues, while job failures, nonfinite metrics, corrupt artifacts, and
lineage mismatches stop descendants. Manual acknowledgement remains supported.

The weighted 300k/500k/1M/2M campaign identities are stale for this primary
recipe and require fresh recipes, specs, worktrees, and exact job IDs.

Local verification after this change: focused HCWDL/high-coverage tests pass
78/78 and the complete repository suite passes 322/322 with the same 14
Matplotlib/Pyparsing warnings. A fresh unweighted Tigris smoke and the four
new campaign identities remain to be launched from an exact clean pushed
commit; no remote jobs were mutated from this local implementation.

## HCWDL named midscale campaign modes (2026-08-08)

HCWDL now registers three distinct immutable midscale populations:
`midscale500k` uses 500,000 train, 250,000 validation, and 250,000 final-test
jets; `midscale1m` uses 1,000,000 train, 400,000 validation, and 400,000
final-test jets; and `midscale2m` uses 2,000,000 train, 500,000 validation, and
500,000 final-test jets. Each reuses the
same count-agnostic matching, persistent assignments, two-phase endpoint gate,
23-node ladder, confirmation, and sealed-test machinery, but receives its own
campaign identity and requires a recipe whose class weights are bound to its
exact deterministic train selection. Future midscale sizes must
receive different mode names rather than editing these counts.

Campaign specs and submission authorizations advance to v6; v3 through v5
artifacts remain readable. Validation requires the stored
role counts to exactly equal the registered mode, closing a previously
implicit integrity check. The creation CLI exposes the new mode and focused
tests cover its exact counts, uncapped dry-run DAG, authorization, tamper
rejection, and v3 compatibility. This change does not alter or cancel the
currently running or prepared v3-v5 campaign.

The named modes require independent deterministic preselections, recipes,
resource/storage evidence, authorization, worktrees, assignments, locks, and
two-phase submissions. None may reuse another mode's recipe lineage.

Verification after registering all three named midscale modes: the focused
high-coverage/HCWDL suite passes 72/72 and the complete repository suite passes
316/316 with the same 14 Matplotlib/Pyparsing warnings. `midscale2m` has not
yet been executed on Tigris. Its exact next step is a clean commit and push,
then a dedicated worktree, deterministic 2m/500k/500k preselection, a recipe
bound to that 2m train-selection hash, scaled storage/profile evidence, exact
v6 candidate dry run and authorization, and an independent two-phase launch.

## PMARD all-model exploratory test comparison (2026-08-08)

The user explicitly authorized evaluation of all completed T100-sweep and
paired-schedule models and reclassified the pilot's 100,000-jet `final_test`
role as an exploratory comparison set. The new isolated contract freezes the
36 plus 27 distinct report/checkpoint identities before access, publishes dedicated
authorization and execution locks, builds one deterministic test selection,
runs an uncapped HLT-only `0-62` GPU array, and aggregates only metrics. Every
evaluation must attest the same ordered identity hash; per-jet predictions are
never written. The aggregate permanently records that post-test rankings are
descriptive and confirmatory claims from this holdout are forbidden.

Implementation lives in
`src/hlt_classification/scouting/exploratory_test.py`, the three
`scripts/*pmard_exploratory_test*.py` entry points, and
`sbatch/run_pmard_exploratory_test.sh`. Scientific and operational semantics
are documented in
[`docs/contracts/PMARD_EXPLORATORY_TEST_COMPARISON.md`](contracts/PMARD_EXPLORATORY_TEST_COMPARISON.md)
and [`docs/PMARD_RUNBOOK.md`](PMARD_RUNBOOK.md). The existing confirmatory
single-finalist code path was not weakened. Current focused verification is
16/16 passing; the complete local suite is 310/310 passing with 15 existing
Matplotlib/pytest-cache warnings. Tigris execution remains to be performed
from exact pushed clean source.

The first Tigris creation attempt at commit `f9926ab` exposed a historical
lineage compatibility bug before any test access: the supplemental sweep's
immutable parent campaign was being recomputed against today's expanded PMARD
campaign registry. The exploratory validator now authenticates every archived
embedded artifact by its own versioned contract and recorded content hash,
while still validating the sweep/follow-up relationship, every training
report, and every checkpoint hash. It does not reinterpret the old
campaign with current constants.

The second pre-access creation attempt established that the completed follow-up
contains 27 rather than 28 models: its registered construction correctly
deduplicated one recipe that won both CE and utility roles. The fixed-64
assumption was therefore invalid. Contract v2 freezes the exact registered
inventory (36 + 27 = 63) and derives the
array bound from the immutable specification. No test-role access occurred in
either failed creation attempt.

The third pre-access creation attempt found that some historical training
reports predate `balanced_accuracy` and `always_qcd_accuracy`. These derived
fields are now optional only in copied validation summaries; the five common
selection/performance metrics remain mandatory, and every exploratory test
evaluation computes the full current metric set. This attempt also stopped
before publishing a specification or reading the test role.

## HCWDL matching-free representation-KD plan (2026-08-08)

The new implementation-grade registered-ablation plan is
[`docs/plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md`](plans/HCWDL_MATCHING_FREE_REPRESENTATION_KD_ASCENTS.md).
It adds four planned full M1--M6 ascents without changing the implemented
logit-only `HCWDL_GRAPH/v1` or `HCWDL_RECIPE/v3`: paired jet plus unordered
token-set KD (`RSET`) under cold and warm initialization, and the same package
plus differentiable latent-relation KD (`RREL`) under cold and warm
initialization. The existing logit cold/warm ascents remain the primary
controls.

The plan freezes 24 new node identities, exact privileged teachers, separate
native-offline charged/neutral representation spaces, fixed RFF set/relation
sketches, train-only gradient calibration, a `rho_repr=0.10` auxiliary budget,
pass-based ramps, 60 passes with validation every pass, macro-AUC-first
selection, paired stochastic streams, one-time compact target construction,
generation-aware just-in-time target cleanup/recovery, zero-coefficient,
jet-only, no-relation, and shuffled-pair controls, and a population-scoped
combined final-test reservation honored by both evaluator families. It includes the
proposed source/module map, CLI and Slurm boundaries, artifact layout,
contracts, failure semantics, complete test matrix, implementation blocks, and
definition of done.

The planning audit found a pre-existing parent-loss mismatch that is now an
explicit implementation blocker: the active HCWDL plan requires class-weighted
CE and unweighted KD row means, while the current shared `pmard_loss` runtime
class-weights CE and both KD terms. HCWDL-RKD requires a versioned
parent-loss attestation, runtime correction, and consistent parent artifacts;
affected reports cannot be relabeled as paired. The plan also requires a new
architecture attestation because current parent artifacts do not publish the
needed model/tap architecture hash.

This is documentation only. No representation-ascent contract, model tap,
target bank, training node, worker, smoke, or submission was implemented or
authorized in this step. Before editing, the focused existing HCWDL/PMARD
training suite passed 40/40 under the local `tagging-hlt` environment with
`PYTHONPATH=src`. After the final plan audit, the four focused HCWDL suites
passed 58/58 in the same environment; Markdown targets and `git diff --check`
also passed aside from line-ending notices. The separately authorized
primary-HCWDL Tigris-validation work remains independent; this new plan does
not submit, cancel, or alter it.

## Preliminary PMARD pilot evidence (2026-08-07)

The recovered 300k/100k/100k PMARD pilot has now completed its teacher ladder,
authenticated validation oracles, K2 alpha sweep, alpha selection, and K0--K6
logit-KD controls. Alpha `1.0` was selected. On validation, K2 improved over
the ordinary K1 self-KD control from CE `0.675880` to `0.675060`, macro AUC
`0.937334` to `0.937468`, and mean log QCD rejection `7.022452` to `7.034357`;
overall accuracy was effectively unchanged. The selective T100 endpoint
closed only about 16--21% of the native-offline oracle gap, and direct TOFF KD
in K6 improved CE/AUC slightly beyond K2 without improving rejection. These
are single-screening-seed validation observations, not confirmed or final-test
claims. Mechanism controls remain in progress; representation, generation,
confirmation, and sealed final-test stages remain pending.

The complete numerical tables, exact endpoint semantics, current limitations,
artifact locations, and code-symbol map are recorded in
[`docs/PMARD_PILOT_PRELIMINARY_RESULTS.md`](PMARD_PILOT_PRELIMINARY_RESULTS.md).
This section supersedes older statements below that no real PMARD scientific
output exists; it does not supersede their implementation or recovery history.

An authenticated supplemental T100 transfer sweep is now implemented. It
holds CE at 25%, scans T100 KD weights `0.15/0.25/0.35/0.50`, T100
temperatures `1/2/4`, and training exposures `10/20/40` complete passes,
assigns the remaining 75% teacher mass to T0 at temperature 1, and trains all
36 rows on HLT inputs only. A single prerequisite
pass caches T0/T100 train logits, so repaired alpha-one inputs are not rebuilt
for every student. The sweep is bound to the completed pilot prefix, compares
against both K1 and the original alpha-one K2 row, and cannot read final test.
Its specification and commands are documented in
`docs/contracts/PMARD_T100_KD_SWEEP.md` and `docs/PMARD_RUNBOOK.md`.

The first live supplemental target-cache job, `48034`, failed after six
minutes before publishing a cache because the selective authorization stores
the versioned family `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1`, while the tensor
builder historically accepted only the runtime selector without `/v1`.
`runtime_repair_family()` now performs an explicit allow-listed translation
for both full-endpoint contracts, and a regression proves the versioned
selective family produces byte-identical repaired tensors. The failed
source-bound sweep specification must not be resumed under corrected source;
recovery requires a new clean commit, worktree, sweep root, and specification.
The next clean attempt, target job `48142`, then showed that normalization at
the tensor builder was too late: the PMARD streamer had already omitted native
offline branches because its projection dispatch still compared the versioned
name literally. The streamer now normalizes at entry, before category gates,
branch projection, confidence dispatch, or tensor construction. Its
fitted-strict regression requires the versioned selective family to project
and forward native offline arrays under the unversioned runtime selector. The
supplemental target worker also evaluates the constrained T100 repaired path
before the ordinary T0 path so future endpoint incompatibilities fail early.

A validation-only paired schedule follow-up is now implemented for the
observed 20-pass CE-control instability. It consumes the completed 36-row
T100 sweep aggregate, freezes both its lowest-CE and best-utility recipe at
each of 10/20/40 passes, carries the 40-pass winners into a predeclared
60-pass extrapolation, and reruns those recipes beside fresh K0 CE-only and
K1 T0-self-KD controls. Every row validates once per complete train pass.
Ten-pass rows use the parent `3e-4` peak LR; 20/40/60-pass groups run both that
fixed LR and the predeclared `L*sqrt(10/passes)` schedule. The maximum grid is
28 uncapped jobs (smaller only when the two parent selection rules choose the
same recipe). It reuses the parent T0/T100 float32 target cache and opens only
the 300k train and 100k validation HLT views; it does not rematch, construct
repaired views, or access test. Contracts and launch procedure are in
`docs/contracts/PMARD_KD_SCHEDULE_FOLLOWUP.md` and `docs/PMARD_RUNBOOK.md`.
Local source compilation passed; the dependency-bearing pytest suite was not
available under the current Windows Python environment and still requires
execution in `atlas_kd_tigris` before push/Tigris submission.

## Active PMARD implementation (2026-08-05)

The user explicitly activated the PMARD mandate. The active scientific plan is
`docs/plans/SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md`; PRAD is a
separate prior campaign and cannot redefine PMARD.

Implemented locally in the current worktree:

- version-4 Scouting schema with the fitted-strict donor's native
  constituent-count boundary (`n_cpfcands` plus `n_lts`) as the authoritative
  charged/lost-track layout; the auxiliary `cpfcandlt_isLostTrack` model
  feature is deliberately not a matching-projection or boundary requirement;
  plus versioned source, label, row-identity, split, artifact, and
  role-capability contracts;
- version-2 60/20/20 source-file split construction: the source audit streams
  per-file 15-class counts, an exact MILP minimizes the worst class-by-role
  fraction deviation under whole-file disjointness, and validation fails
  closed above a predeclared 0.02 absolute tolerance;
- exact 15-class mapping, 21-channel CMSSW transform, 200-token HLT view, and
  90/60 native-offline no-SV view;
- projected bounded ROOT streaming with rank/worker file partition and sealed
  pre-lock final-test branch access, plus eight-file round-robin train reads
  and deterministic natural-population 32,768-row RAM shuffle buffers;
- v2 sparse candidate graphs with audited node/edge/track-validity features,
  three-round bipartite message passing, positive-unlabeled bootstrap,
  two ultra-conservative pseudo-label rounds, held-out edge and post-assignment
  calibration, deterministic score quantization, exact alternative margins,
  Hungarian/OT controls, and `M0`--`M5`;
- adversarial synthetic loss/fake/split/merge/category-confusion tests,
  genuinely different event-mixed jets, independent HLT/offline perturbation
  stability, confidence Brier and matching-only category gates;
- the canonical `fitted_strict` selective matcher as a standalone production
  module: immutable semantic hashes for the 7,500-jet edge/confidence/audit
  bundle, exact empirical LLR scoring, strict PID/charge and broad kinematic
  gates, rectangular Hungarian private dummies, calibrated ten-diagnostic
  confidence at threshold `0.9828147479721088`, native-index restoration after
  lost-track exclusion, batch inference, and persistent-cache construction;
- deterministic class-proportional row-selection manifests and immutable
  fitted-strict assignment caches computed once per selected jet: compact
  per-source CSR/DEFLATE shards store only entry, uint8 HLT index, uint16
  offline index, and uint16 confidence; consumers use a bounded lazy LRU and
  verify split/selection/matcher/data hashes before reuse;
- exact `P4_ONLY/v1` alpha endpoints and byte-identical `alpha=0`, plus charged
  eligibility, shuffled/corrupted matches, direction/response/wrong/random
  controls, log-angular interpolation, and confidence weighting;
- versioned `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1` mixed-type repair: unique
  accepted assignments, exact all-21-channel plus p4 offline replacement for
  accepted tokens at `alpha=1`, byte-identical unmatched HLT tokens, raw
  continuous interpolation, SHA-256 identity-bound nested discrete/validity
  switches, charged/neutral track-applicability coherence, and unchanged HLT
  count/order/mask without durable repaired datasets;
- a hash-chained selective-assignment authorization boundary binding the
  canonical fitted-strict artifact and threshold, split, deterministic row
  selection, compact train/validation assignment manifest, all five input
  categories, and the exact unmatched-as-HLT policy before any privileged
  teacher or student;
- 21-input/15-output canonical Weaver adapter, native-offline diagnostic model,
  K0--K6 loss semantics including a valid alpha-zero K2 collapse, class
  weights, separate `R4_PAIR`/`R4_GRAM` representation arms, corrected
  per-signal QCD discriminants, and a fixed three-generation registry whose
  representation and anchored-companion paths remain executable when the
  selector legitimately chooses alpha zero via an exact HLT identity alias;
- cross-file full-batch packing, Weaver no-decay optimizer exclusions,
  named RNG domains, FP32 CE/KL/recursive representation and pair/Gram math
  plus FP32 RAM logits under BF16 ParT autocast, eight-check validation cadence,
  durable 50-update interval-mean loss history, validation/preemption-only
  rolling checkpoints, batch-shell-to-Python `exec` signal delivery, exact
  batch-offset/RNG/optimizer/best-state resume, a shared deterministic
  4,096-row smoke selection, one bounded RAM-cache miniature, and HLT-only post-cache streams
  with unused GPU teachers/matchers released for R0 and zero-coefficient
  controls, plus the HLT-only inference signature,
  fixed observer-only stratification and final nested paired statistics;
- campaign-spec v10 process-local repaired-view caching: nonzero-alpha
  teachers retain only FP32 privileged train/validation model views, positive
  representation KD retains aligned HLT/privileged train views, and
  identity-indexed replay reproduces the exact file/chunk/32,768-row-buffer
  epoch sampler and batch tails; the 320-GiB/75%-of-Slurm preallocation gate,
  192-GiB smoke/pilot and 384-GiB production requests, zero durable output,
  v9 validation support, and exact-source enforcement keep the active
  recovered pilot scientifically unchanged;
- hash-chained freeze locks and a minimum-storage Tigris DAG whose smoke graph
  cannot access final test, with exact-ID monitor/resume/cancel, authenticated
  resource/storage/miniature evidence, nonmutating production dry run, and
  explicit production authorization gates; a pilot graph uses exact
  300k/100k/100k row selections, constructs final-test selection/assignments
  only after the execution lock, and reports HLT, native-offline, and
  matched-offline endpoint oracles alongside deployable finalists.

Local verification passes `228 tests` with 14 Matplotlib warnings. The new
RAM-view-cache coverage includes three-epoch online-versus-cached
identity/batch/tensor equivalence, memory fail-before-allocation, alpha-zero
and native-offline bypass, campaign v9 compatibility, and campaign v10
resource wiring. Existing coverage includes selective
all-field endpoint, persistent sparse join, pilot-DAG, matcher, KD, resume, and
legacy campaign regressions. CLI help and `git diff --check` also pass; the
remaining notices are repository line-ending notices.

The first genuine pilot DAG, campaign `pmard_pilot_62dd448731cdd932` at pushed
commit `00f03a6b1818ad16f9538e8e9e155082293677cd`, reached the persistent
assignment-cache array (`43465_0`--`43465_42`) after its source, split,
feature-audit, row-selection, Weaver, and matcher-artifact prerequisites
completed. Every assignment shard failed before publication because the port
required `cpfcandlt_isLostTrack` to equal a positional collection boundary.
That check was stricter than the fitted-strict donor, which slices regular
charged candidates from the authenticated scalar counts. Schema v4 corrects
the projection and decoder contract: `n_cpfcands + n_lts` defines the charged
family and the appended lost-track suffix, while the auxiliary model feature
cannot redefine that boundary. No matcher-quality or PMARD result can be
inferred from this infrastructure failure, and the immutable v3 pilot must not
be resumed under v4 source.

The schema-v4 replacement pilot, campaign `pmard_pilot_3f379cb66398fcc8` at
commit `775bed70`, completed its persistent assignment cache, baseline budget
grid, budget selection, and temperature grid. Privileged-teacher array
elements `43733_1`--`43733_5` then failed before launching the trainer because
the workflow appended the authenticated matcher threshold to `argv` as a
Python float after `_script` had stringified the initial arguments;
`subprocess` accepts only text, bytes, or path-like arguments. The same latent
fault existed in the nonzero-alpha student builder. Both builders now
normalize their complete final command vectors to strings, with regressions
covering the exact threshold in teacher and student commands. This is an
execution-only correction and does not change matcher, repair, loss, model, or
selection semantics. The replacement pilot remains immutable and cannot be
continued in place under the corrected source snapshot. A version-4 prefix
import now provides the narrow recovery path: it proves Git/AST compatibility,
reuses independent source/Weaver/HLT-budget/temperature artifacts, rebuilds
the assignment cache plus its dependent authorization and training locks, and
then submits from `teachers` onward. Failed teacher/descendant artifacts are
categorically excluded.
The first prefix-resume submission exposed a Slurm-controller lifetime edge:
the resume helper attached the newly submitted `teachers` job to historical
completed job `43732`, although the imported `training_lock` artifact already
authenticated that predecessor. Completed IDs may age out of the controller
and are not valid scheduling dependencies. Resume commands now include only
dependencies newly submitted by the same resume invocation; imported reusable
predecessors remain artifact dependencies, and scheduler failures retain the
actual `sbatch` diagnostic. Focused resume/recovery/campaign tests pass 10/10,
and the complete suite passes 218/218 (plus one local pytest-cache warning).
The next recovered teacher launch exposed a second execution-only defect:
assignment manifests intentionally store shard paths relative to
`matcher/assignments` (or `matcher/final_assignments`), while the persistent
consumer resolved them relative to the manifest's `matcher` parent. The
recovery had correctly imported every shard, but the consumer looked one
directory too high and failed closed before the first training update. The
store now resolves both canonical initial and final assignment roots while
retaining colocated custom-manifest behavior. Prefix-import contract v4 proves
the narrowly authorized execution corrections by AST comparison against pilot
commit `775bed70`; no trained model or matcher output is reinterpreted.
The subsequent teacher launch reached selective full-field repair and exposed
an over-broad HLT identity check. Selective repair validated all visible HLT
tokens, including unmatched ambiguous-PID tokens that the contract requires it
to preserve.
Repair now validates chargedness only on `matched_tokens`; complete repair
still validates every token because its matched set is complete. A regression
proves that an invalid unmatched identity remains byte-identical while the
same identity fails when matched, and the v4 recovery AST proof authorizes
exactly this scope correction in addition to the two earlier execution fixes.
The following real-data launch then established that fitted-strict's exact PID
gate could accept unknown-to-unknown (`-1`) pairs. Those pairs are outside the
authorized five-category endpoint and can carry invalid offline identities.
Assignment-cache construction now requires exact categories 0--4, exact charge
in -1/0/+1, and charged/neutral charge coherence on both endpoints. Because
the original cache violates that scientific eligibility rule, recovery v4
does not reuse it. It resubmits only the inexpensive 43-way compact cache,
manifest, full-endpoint lock, and training lock while retaining the expensive
HLT-only budget and temperature checkpoints across their scheduling-only gate.

Still required externally before production: commit and push the exact clean
source; authenticate the native 53-file dataset and split; run installed-Weaver
FP32 parity; complete the genuine streamed Tigris smoke graph; capture measured
RAM/GPU/I/O/wall/storage evidence; and inspect the full production dry run.
The smoke DAG ends at `miniature_summary` and structurally contains no
`final_test` task. No PMARD scientific result, real matcher lock, checkpoint,
resource measurement, or final-test output is claimed in this handoff.

Track-only and p4-plus-track repair remain deliberately unavailable until the
real Stage-A units/definition audit creates a compatibility lock; attempting
either fails closed. The old complete full-particle implementation remains
locally verified but is not the selected scientific endpoint. The selected
fitted-strict path is intentionally partial and uses the new selective
all-field contract; it never claims that the unmatched fraction became offline.

## Repository state

Transfer Blocks 1--4, 6, and 7 are complete locally. Transfer Block 5 is
code-complete and its authoritative installed-Weaver FP32 equivalence check
passed on Tigris. A complete real PRAD miniature is still required.

Implemented:

- frozen JetClass schema, canonical identities, bounded ROOT reads, and
  deterministic balanced splits;
- registered `fixed_hlt_v3_track_dominant_proxy/v1` degradation and replica
  contracts with donor-v1 deterministic parity;
- immutable authenticated offline and HLT cache shards;
- atomic no-overwrite publication and exact completed-shard reuse;
- streaming cache range/batch readers that do not materialize the campaign;
- source, split, schema, profile, replica, generator, offline-parent, and
  source-array lineage;
- aggregate HLT diagnostics with construction indices excluded from deployable
  artifacts;
- exact label, identity, role, validity-state, padding, and parent validation;
- canonical 17-feature Particle Transformer input construction;
- frozen from-scratch standard-four Weaver baseline adapter and standalone
  FP32 logits/training-gradient/mask/state-dictionary attestation, including
  exact topology comparison for Weaver's nonfinite auxiliary Lorentz-input
  derivative.
- deterministic shard-local epoch sampling and fixed-update AdamW training;
- exact model, optimizer, schedule, scaler, sampler, replica-cycle, RNG, and
  history checkpoint resume;
- deterministic validation checkpoint selection without early termination;
- ordered label-free prediction shards and authenticated label joins;
- frozen accuracy, CE, efficiency, AUC, Brier, 15-bin top-label ECE,
  QCD-rejection, and paired-bootstrap definitions.
- clean-source snapshots, immutable campaign specifications, storage gates,
  task attestations, per-job durable submission journals, exact-ID Slurm
  ledgers, monitoring, and resume plans;
- thin absolute-path Tigris workers and two-lock final-test authorization;
- a smoke-only deliberate update-one training interruption followed by
  dependent exact-checkpoint resume through the ordinary training worker.
- the active PRAD implementation: proportional 500k/150k/500k split contract,
  paired-view audit/cache, replica-aligned charged/neutral Hungarian matching,
  exclusive C/A K=2/3/4 targets, and train-only moments/positive weights;
- a parity-oriented offline-teacher/HLT-student Weaver extension with symmetric
  contextual relations, bounded centered head bias, zero-initialized gates,
  HLT-only deployment signature, and explicit nondeployable oracle path;
- optional validated float16 dense teacher relation/bias caching and a
  mandatory installed-Weaver CUDA PRAD mechanics attestation before E0;
- configuration-driven E0--E10 and all registered V1--V10 switches, fixed
  staged budgets, frozen teacher, exact resume, deterministic identity-bound
  in-batch shuffled control, and five confirmation seeds;
- PRAD validation selection, bounded relation-fidelity diagnostics, label-free
  inference, validation benchmarks, pre-test five-seed final selection,
  restart-safe finalist/execution claims, paired bootstrap reporting for every
  final five-seed graph, and complete required plot assembly;
- a source-bound PRAD Tigris DAG using `reu-aisocial`/`tigris`, absolute worker
  paths, exact numeric dependencies, durable partial-submit recovery, exact-ID
  monitoring/resume/cancellation, and production requests bound to completed
  smoke `sacct` evidence plus conservative source-bound storage headroom.
- the versioned `prad_minimum_durable_storage_v1` profile: the 15-node DAG
  rebuilds paired views, structural targets, and teacher outputs in Slurm
  job-local temporary storage and deletes them on worker exit; completed runs
  retain optimizer-free best/final checkpoints, incomplete runs retain only
  one rolling exact-resume checkpoint, and prediction shards store logits with
  split-row-bound implicit identities.

Still not implemented or accepted:

- a completed real Tigris minimum-storage miniature.
  The first authentic PRAD smoke submission, source commit
  `8bb20be8cb351f0a0fd71f50dabcc03467790161` and jobs `41126`--`41147`,
  failed closed before data/model execution: sorted JSON role keys exposed an
  order-sensitive split-builder check, and the independent Weaver worker read
  the absent split manifest before dispatch. Both defects now have local
  regressions and fixes. A direct installed-Weaver diagnostic then established
  that every auxiliary Lorentz-input derivative is NaN for all-valid, singly
  padded, and multiply padded fixtures alike, while logits, feature gradients,
  and parameter gradients remain finite. The v3 parity contract compares that
  diagnostic topology exactly and keeps all training-required quantities
  strictly finite. The v3 authoritative rerun passed on Tigris at source
  commit `2b3e34c961067f1d458561d67b51b2ffd780704a`. A disk-backed smoke from
  that source was submitted, but complete monitor/resource evidence has not
  been supplied and it cannot validate the new storage profile;
- PRAD's real `reports/data_audit.md`, exact full split and ephemeral-data
  attestations,
  trained checkpoints, experiment results, sealed 500k test metrics, plots,
  and final scientific recommendation. Those are intentionally not fabricated
  locally and require the clean committed source plus the real miniature first;
- all twelve HOSD implementation steps. The migrated plan is authoritative,
  but `src/hlt_classification/hosd/`, its target extractors/caches/teachers,
  taps/heads/probes, auxiliary and feedback training, combinations, metrics,
  confirmation/scale, final selection/seal, and HOSD production DAG do not
  yet exist.

No file in this repository imports `Fresh_check` at runtime. The repository
has an independent Git history. No remote is configured; direct Tigris access
from the implementation session reached the cluster but was rejected at the
required interactive Duo MFA step.

The exact PRAD execution sequence, including clean-source transfer, smoke dry
run, monitored miniature, reviewed resource requests, and separately reviewed
production dry run, is in `docs/PRAD_RUNBOOK.md`.

## Contracts added

```text
scientific HLT profile:
fixed_hlt_v3_track_dominant_proxy/v1

serialized HLT profile:
hlt_classification_hlt_v3_profile_v1

replica policy:
hlt_classification_hlt_replica_manifest_v1

cache shard:
hlt_classification_cache_shard_v1

offline cache manifest:
hlt_classification_offline_cache_manifest_v1

HLT cache manifest:
hlt_classification_hlt_cache_manifest_v1

canonical ParT inputs:
hlt_classification_part_inputs_v1

Weaver adapter/parity report:
hlt_classification_weaver_part_v1
hlt_classification_weaver_part_v2
hlt_classification_weaver_part_v3

training checkpoint:
hlt_classification_training_checkpoint_v2

training report:
hlt_classification_part_training_report_v2

prediction manifest:
hlt_classification_prediction_manifest_v1

evaluation report:
hlt_classification_evaluation_report_v1

metrics:
hlt_classification_metrics_v1

source snapshot and compact campaign:
hlt_classification_source_snapshot_v1
hlt_classification_baseline_campaign_spec_v1

execution evidence:
hlt_classification_storage_measurement_v1
hlt_classification_slurm_resource_evidence_v1
hlt_classification_runtime_environment_v1
hlt_classification_task_attestation_v1
hlt_classification_submission_job_record_v1
hlt_classification_submission_ledger_v1
hlt_classification_monitor_report_v1
hlt_classification_resume_plan_v1

smoke interruption evidence:
hlt_classification_training_interruption_evidence_v1

final-test locks:
hlt_classification_finalist_lock_v1
hlt_classification_final_test_execution_lock_v1
hlt_classification_final_test_execution_claim_v1

PRAD campaign surfaces:
hlt_classification_prad_split_manifest_v1
hlt_classification_prad_paired_view_v1
hlt_classification_prad_structural_targets_v1
hlt_classification_prad_teacher_outputs_v1
hlt_classification_prad_relation_v1
hlt_classification_prad_checkpoint_v1
hlt_classification_prad_model_checkpoint_v1
hlt_classification_prad_ephemeral_dataset_v1
hlt_classification_prad_training_report_v1
hlt_classification_prad_training_v3
hlt_classification_prad_prediction_manifest_v2
hlt_classification_prad_teacher_prediction_manifest_v1
hlt_classification_prad_evaluation_report_v1
hlt_classification_prad_campaign_spec_v2
hlt_classification_prad_submission_ledger_v1
hlt_classification_prad_task_attestation_v1
hlt_classification_prad_resource_evidence_v1
hlt_classification_prad_final_selection_v1
hlt_classification_prad_runtime_validation_v3
hlt_classification_prad_statistical_report_v2

PMARD/Scouting reviewed surfaces:
hlt_classification_scouting_schema_v4
hlt_classification_fitted_strict_matcher_v1
hlt_classification_scouting_feature_audit_v2
hlt_classification_pmard_row_selection_v1
hlt_classification_pmard_selective_assignment_shard_v1
hlt_classification_pmard_selective_assignment_manifest_v1
hlt_classification_pmard_ephemeral_assignment_v2
hlt_classification_pmard_matcher_report_v2
hlt_classification_pmard_matcher_validation_v2
hlt_classification_pmard_full_role_coverage_v2
hlt_classification_pmard_training_report_v4 (legacy readable parent)
hlt_classification_pmard_resume_checkpoint_v4 (legacy readable parent)
hlt_classification_pmard_training_report_v5
hlt_classification_pmard_resume_checkpoint_v5
hlt_classification_pmard_lock_v4
hlt_classification_pmard_campaign_spec_v9
hlt_classification_pmard_t100_kd_sweep_spec_v2
hlt_classification_pmard_t100_kd_targets_v1
hlt_classification_pmard_t100_kd_sweep_report_v2
hlt_classification_pmard_t100_kd_sweep_ledger_v2
hlt_classification_pmard_kd_followup_spec_v1
hlt_classification_pmard_kd_followup_report_v1
hlt_classification_pmard_kd_followup_ledger_v1
```

Changing registered-v1 event, substream, replica-cycle, degradation, or cache
semantics requires a new applicable scientific or serialized contract version.

## Readiness matrix

| Capability | Status | Evidence |
|---|---|---|
| Package and documentation | Passed locally | `tests/test_scaffold.py` |
| JetClass data foundation | Passed locally | `tests/test_data_foundation.py` |
| HLT-v3 mechanism contracts | Passed locally | `tests/test_hlt_v3.py` |
| Donor-v1 deterministic parity | Passed locally | frozen array/diagnostic hashes |
| Authenticated cache construction/resume | Passed locally | `tests/test_cache_pipeline.py` |
| Cache corruption/cross-profile rejection | Passed locally | `tests/test_cache_pipeline.py` |
| HLT shard/batch layout invariance | Passed locally | bitwise concatenated arrays |
| Canonical ParT transforms | Passed locally | `tests/test_part_inputs.py` |
| Wrapper/parity logic | Passed with interface-faithful test double | `tests/test_particle_transformer.py` |
| Installed-Weaver FP32 parity | Passed on Tigris | v3 report at `2b3e34c` matched logits, masks, model state, training-required gradients, and auxiliary Lorentz-gradient topology |
| Real Tigris minimum-storage miniature | Attempt 6 completed through E4 | core-screen array `42278` reached Stage B in E5--E10 and exposed an optimized-SDPA mixed-freeze backward incompatibility |
| Baseline training/resume | Passed locally | `tests/test_training_engine.py` |
| Inference and metrics | Passed locally | `tests/test_inference.py`, `tests/test_metrics.py` |
| Production DAG dry run | Passed locally | `tests/test_campaign.py` |
| Source-drift/all-reused validation | Passed locally | `tests/test_campaign.py` |
| Failure injection and exact-ID recovery | Passed locally | `tests/test_campaign.py` |
| Worker static contracts | Passed locally | `tests/test_campaign.py` |
| Full production authorization | Not authorized | Transfer Block 8 |

## Donor provenance

Inspected donor commit:

```text
bfcda5bd6fd48037ab198bcca1eadf6f4d87e131
```

Block-4 cache references were inspected, not copied unchanged:

```text
jetclass_fresh/hlt_cache.py
5631ac115a4f352739492e0db06c5e9c3111fa52d522d273243504e0731823c4

teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py
57d0dde4e4aad5aa622fc62c2020954ae69a0108cf89178d7304dabd2f6345e4
```

Exact migration details for all blocks are in `docs/LEGACY_SOURCE_MAP.md`.

## Verification evidence

Validated with bytecode and pytest caches disabled:

```text
python:
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe

focused Block 4:
8 passed

focused Block 7:
8 passed

complete discovered suite:
205 passed, 14 warnings

focused PRAD suite:
60 passed

focused PMARD/Scouting suite:
54 passed
```

The Block-4 suite proves byte-identical clean versus interrupted/resumed
offline and HLT artifact trees, fail-closed corruption and lineage drift,
cross-profile rejection, exact parent-role validation, deployable-artifact
hygiene, and bitwise HLT array equivalence across different source shard,
output shard, and processing-batch layouts.

No real Tigris data, GPU, or installed Weaver path was exercised in this
block. The local environment has PyTorch 2.5.1 but no Weaver installation.

The current PMARD/Scouting focused evidence is 54 passing tests. It includes exact
uninterrupted-versus-resumed state equivalence, a deliberately poor model that
still completes, nonfinite failure injection, bitwise batch-invariant
prediction shards, all frozen metric edge cases, full-endpoint authorization
rejection with full-role count-only lineage, recursive FP32 representation KD
under BF16 inputs, durable interval-mean history, batch-shell signal delivery,
bounded cross-file reads, sparse validation/checkpoint cadence, and an
end-to-end zero-alpha selection through representation KD, anchored companion
training, and the next dual-teacher generation student.
The fitted-strict additions authenticate the canonical model bundle, reproduce
reference scores/margins/confidences numerically, exercise both rectangular
assignment directions, private dummies, post-assignment rejection, strict
PID/charge gates, wrapped phi, invalid PID, native lost-track boundaries,
one-to-one integrity, unmatched p4 preservation, artifact tampering, and the
online PMARD-stream dispatch.

The campaign portion includes source drift after an
all-reused graph, cross-campaign lineage rejection, storage insufficiency,
exact dependency IDs, task-byte corruption, dry-run nonmutation,
failed-descendant recovery, stale exact-job cancellation lineage, and
final-test lock authorization. Python compilation, all 21 PRAD script `--help`
paths, all 11 shell syntax checks, `git diff --check`, and the whitespace audit
pass.

The PRAD-focused suite covers exact split constants and proportional/disjoint
construction, Hungarian thresholds and pair masks, C/A targets, symmetric
relations, masked centering/bounds, exact zero-gate parity, class-gradient
flow, replica-aligned target selection, frozen teacher behavior, HLT-only
inference, test locks, exact checkpoint state, frozen primary metrics,
train-only statistics, source-bound campaign guards, numeric dependencies,
partial-submit reuse, smoke-bound production resources, dense teacher-output
shape/storage validation, accumulated training time, semantic-only causal
isolation, teacher semantic validation, exact final-claim recovery, exact
array-element attestations, and synthetic installed-interface runtime
attestation. Minimum-storage coverage additionally proves in-memory
reconstruction does not publish arrays, task-local cache paths cannot fall
inside the campaign/project tree, completed training removes full optimizer
state while retaining loadable best/final model-only checkpoints, and compact
predictions bind identities through authenticated split row ranges. The two
Tigris-derived regressions prove split construction is
invariant to canonical JSON key sorting and independent Weaver dispatch does
not touch a split artifact. A direct installed-Weaver run at corrected commit
`22c15fc2051f103e50361beb2759731cb956dc8a` then reported zero maximum error
for logits and feature/parameter gradients but a nonfinite Lorentz-input
gradient difference. This is consistent with the fixture's multiple identical
padded zero four-vectors. The v3 contract instead authenticates exact nonfinite
topology for that unused derivative while requiring every training surface
finite; its Tigris run at `2b3e34c` passed with exit code zero.

The first minimum-storage miniature at `288aafc` authenticated split, audit,
installed-Weaver parity, CUDA PRAD runtime, and task-local train statistics in
jobs `41820`--`41824`. E0 job `41825` completed training and published compact
model checkpoints, then failed closed when evaluation restored Weaver's
registered tensor `_counter` buffer by assigning a Python integer. The restore
helper now mutates tensor counters in place while preserving legacy integer
counters, with a regression that proves registered-buffer identity and value.

The replacement miniature at `6bcda2d` then completed split, audit, parity,
runtime, statistics, and the entire E0 train/evaluate task in jobs
`41967`--`41972`. E1 job `41973` failed before training because its CLI passed
the intended `model_factory` keyword to an engine signature that still named
the parameter `model`, even though the engine body already invoked
`model_factory()`. The signature is aligned, the CLI now binds its complete
keyword set against the real engine signature in a regression, and a tiny
full-budget teacher test exercises factory construction through compact
best/final checkpoint publication and rolling-checkpoint removal.

The `58ee660b` miniature passed the repaired teacher path, oracle, and E3/E4.
Core-screen array `42015` then failed E5--E10 during their relation-only Stage
A with PyTorch `RuntimeError: LSE is not correctly aligned (strideH)`. Those
are exactly the pair-supervised graphs; E3 begins directly in all-trainable
Stage C and completed. The Stage-A loss had anchored zero-valued hard/KD
placeholders to final logits, unnecessarily asking installed CUDA attention to
backpropagate a zero gradient through frozen later blocks while only its
additive bias path remained differentiable. The zero anchor now uses the
pre-attention relation tensor. A regression requires final logits to receive
no Stage-A gradient, and runtime-validation v2 reproduces the registered
freeze schedule on installed CUDA before E0.

The next `d2237422` miniature completed split, installed-Weaver parity, data
audit, and train statistics. Its runtime-validation subprocess returned exit
code zero in job `42084`, which establishes that the new Stage-A CUDA
backward attestation passed. The task wrapper then rejected that valid report:
it requested the v2 contract name while the reusable content-hash validator
still defaulted to schema version 1. The validator now accepts an explicit
expected schema version while preserving version 1 as its default, the PRAD
runtime worker explicitly requests version 2, and a regression exercises the
real dispatch path with a content-authenticated v2 report.

The `f90cf5fe` miniature then completed its runtime worker, E0 baseline, E1
teacher, E2 oracle, and E3/E4. Core-screen E5--E10 all failed after the same
roughly four-minute interval with
`RuntimeError: LSE is not correctly aligned (strideH)`. Their shared timing
and exclusive pair-supervised stage schedule identify the transition from the
repaired relation-only Stage A to partially unfrozen Stage B as the failure
surface.
Stage B intentionally freezes early ParT blocks while training later blocks
and the differentiable relation bias; this differs from both Stage A and E3's
fully trainable Stage C. Training contract v3 records
`math_for_stage_b_mixed_freeze_v1`: CUDA Stage B selects PyTorch's reliable
math SDPA backend, while Stage A, Stage C, CPU, and inference retain automatic
selection. Runtime-validation v3 now performs a full mixed-freeze Stage-B
forward/backward with the production loss and requires finite nonzero
gradients before E0.

## Historical PRAD next task (superseded as the active conversation task)

Commit the Stage-B backend policy and runtime-validation v3, cancel only the
pending descendants of `smoke_min_f90cf5fe` through its v2 ledger, transfer
the exact fix commit to Tigris, and execute a fresh PRAD real miniature exactly
as `docs/PRAD_RUNBOOK.md` specifies. Runtime-validation v3 must authenticate
the mixed-freeze CUDA backward before E0; the miniature must still complete
training, selection, locked-test, and aggregation.

```bash
export PYTHONNOUSERSITE=1
python -s scripts/validate_weaver_parity.py --device cpu
```

Do not mark PRAD production readiness accepted until every new smoke task
attestation authenticates and measured resource/storage evidence is captured.
A full PRAD campaign remains unauthorized in this handoff.

## Exact active next task: corrected PMARD pilot

Commit and push the prefix-import recovery utility, update the clean Tigris
checkout to that exact commit, create a fresh pilot spec, import the compatible
completed prefix from `pmard_pilot_3f379cb66398fcc8`, inspect the recovery
dry run, and submit from `teachers` onward. Do not reuse failed teacher or
descendant outputs. Cancel only the old campaign's exact dependency-doomed
jobs through its own submission ledger.

## HCWDL local implementation closure (2026-08-08)

The first real smoke reached and completed all six endpoint qualifiers, but
Tigris represented the submitted `--hold` gate as `JobHeldAdmin`; the ordinary
submitting user could not release it. The same execution also demonstrated
that a campaign bound to the shared main checkout fails if another experiment
pulls that checkout mid-DAG. Campaign/command-plan contracts advance to v3/v2.
Creation and submission now bind the exact invoking worktree, the initial
submission stops after endpoint qualification, and
`continue_hcwdl_campaign.py` validates the complete attested prefix plus the
human acknowledgement before submitting the gate and downstream ladder with
no Slurm hold. This correction passes the 58 focused orchestration tests and
the complete 311-test suite locally; it still requires a fresh exact-commit
Tigris smoke.

The active conversation task is now the local implementation of
`docs/plans/HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`. The previous PRAD
and corrected-PMARD operational next tasks above are historical context and are
not authorization to contact Tigris in this task.

All six locally implementable HCWDL blocks now exist:

1. The exact high-coverage donor runtime and three fitted resources are
   packaged under `src/hlt_classification/scouting`; every copied donor file is
   recorded in `docs/LEGACY_SOURCE_MAP.md` at exact clean donor commit
   `64be1a82f11f42949fdffa639a869ccea2528bfa`.
2. Cross-fitted empirical matching, native-index restoration, lost-track
   exclusion, lexicographic one-to-one assignment, 18 confidence diagnostics,
   compact int16/uint16 shards, complete-role manifests, deterministic sampled
   recomputation, and strict sub-10% dustbin authorization are implemented.
3. Shell Exact, Shell Soft, and HC Exact repair all 21 fields. Shell Exact has
   confidence-warped intermediate alphas, byte-exact D0, exact assigned D100,
   exact HLT dustbins, deterministic identity-bound discrete switches, and a
   fixed HLT skeleton.
4. The immutable 23-node cold/warm graph, HLT/D25/D50/D75/D100/TOFF domains,
   fresh/warm initialization, one/two-teacher FP32 KD, recipe-bound AdamW
   epsilon and microbatch accumulation, 60 passes, every-pass validation,
   AUC/CE/logR/earliest selection, final and selected checkpoints, and exact
   resume are executable.
5. Smoke/pilot/midscale500k/midscale1m/midscale2m/production DAGs, source-byte reauthentication, split validation,
   locks, fixed endpoint qualification, exact human acknowledgement, 55-row
   confirmation registry, reporting, one-claim final evaluation, exact-ID
   recovery/cancellation, resource/storage evidence, and Slurm command
   generation are implemented. Submission stops after the six diagnostics;
   after inspection and lineage-bound acknowledgement, a separately invoked
   continuation validates the complete prefix and submits the remaining DAG.
   A measured non-executable candidate
   spec now hashes the exact future Slurm commands independently of the
   enclosing spec; submission authorization v3 binds that command plan and
   exact resource-request hash, and the executable spec must reproduce both.
   The first explicitly authorized smoke miniature uses conservative bootstrap
   requests without claiming they were measured; pilot/production still
   require a genuine measured resource profile.
6. Donor parity, tamper/failure guards, assignment/repair/graph/training tests,
   all CLI help surfaces, a complete bounded local smoke, and a full planning
   pilot dry run are implemented locally.

The primary two-teacher ladder decision was locked on 2026-08-08 from the
completed `pmard_kd_followup_b8a493547de8bd7e` results: 60-pass maximum,
validation every pass, macro-AUC-first checkpoint selection, dual-teacher peak
LR `3e-4`, and CE/predecessor/privileged weights `0.25/0.40/0.35` with
privileged temperature `2`. `HCWDL_RECIPE/v3` now locks the complete primary
profile: single-teacher CE/KD `0.25/0.75`; privileged/HLT temperatures `2/1`;
all peak LRs `3e-4`; effective/microbatch 256 with accumulation one; fixed
AdamW and five-percent warmup/cosine/minimum schedule semantics; authenticated
sqrt-inverse train-count class weights; constant coefficients; and the warm
label-only control. D0 teaching M1 is routed to the HLT temperature one while
sole privileged D teachers use two. A changed value requires a separately
identified `registered_ablation` and cannot silently alter the primary.

New contract families are `HIGHCOV_MATCHER_RESOURCES/v1`,
`HIGHCOV_DENSE_ASSIGNMENT_{SHARD,MANIFEST}/v2`,
`HIGHCOV_DENSE_ASSIGNMENT_LOCK/v1`,
`HIGHCOV_ASSIGNMENT_RECOMPUTATION_AUDIT/v1`, `HCWDL_ARTIFACT/v1`,
`HCWDL_{NODE_SPEC,GRAPH,TRAINING_REPORT,CHECKPOINT_SELECTION}/v1`,
`HCWDL_RECIPE/v3`,
`HCWDL_EPHEMERAL_{VIEW,TARGET}_BANK/v1`, `HCWDL_LOCK/v1`,
`HCWDL_EXECUTION_CLAIM/v1`, `HCWDL_ENDPOINT_{QUALIFICATION,DIAGNOSTIC_ACK}/v1`,
`HCWDL_{SCREEN,CONFIRMATION,FINAL}_AGGREGATE/v1`,
`HCWDL_FINAL_{EVALUATION,EVALUATION_MANIFEST}/v1`,
`HCWDL_CAMPAIGN_SPEC/v6` (v3-v5 readable), `HCWDL_COMMAND_PLAN/v2`,
`HCWDL_SUBMISSION_LEDGER/v2`, `HCWDL_MONITOR_REPORT/v1`,
`HCWDL_{RESOURCE_PROFILE,STORAGE_ESTIMATE}/v1`,
`HCWDL_SUBMISSION_AUTHORIZATION/v6` (v3-v5 readable),
`HCWDL_CACHE_MINIATURE/v1`, and `HCWDL_LOCAL_SMOKE_REPORT/v1`.
PMARD training/resume contracts advance to v6 for explicit microbatch,
accumulation, and Adam-epsilon semantics; validators retain v4/v5 report
compatibility.

Local verification evidence at the final post-handoff audit:

- repository-wide suite: 299 passed with the same 14 Matplotlib/Pyparsing
  deprecation warnings;
- all 51 HCWDL/high-coverage Python surfaces compile, all 25 thin CLIs have
  tested help surfaces, and Slurm shell invariants pass;
- standalone bounded smoke:
  `%TEMP%/hcwdl_local_smoke_final_500de1f8e45f44ba8fb84b0eebca898a/smoke_report.json`,
  content hash
  `8f3157b434d6b371f3513de83ffa175d13cbb9e5a9d6e9a196849013b13dae0e`;
  all 23 nodes and five alphas completed, endpoint exactness passed, and the
  final-test role was not accessed;
- the pilot dry-run regression renders exact 300,000/100,000/100,000 roles,
  every dependency and resource class, uncapped source arrays, the held human
  gate, and no submission.

The placeholder/donor-import/development-path static scans were clean and
`git diff --check` passed with line-ending notices only. During that local
implementation audit, no SSH, Slurm, remote push, ROOT write, or Tigris action
occurred.

The first separately authorized Tigris smoke attempt reached source audit but
stopped at split validation (`source_audit` job 56135 completed; `splits` job
56136 failed before matching or training). The cause was an order-sensitive
role-key check: immutable JSON publication canonically sorts object keys while
the validator required the pre-serialization insertion order. Split role maps
are now validated by their exact key set and consumed in the canonical
`SPLIT_ROLES` order. A regression publishes, reloads, and validates a real
split manifest through the immutable JSON path. Focused HCWDL/split tests pass
69/69 and the repository-wide suite remains 299 passed with 14 warnings.

The fresh smoke at corrected commit `68406fd9c9077ead912555fe85ff7fb0e6208d00`
then exposed a second adapter-only split failure (`source_audit` job 56179
completed; `splits` job 56180 failed). The authenticated source-manifest file
rows contain audit metadata such as `baseline_selected_entries`, branch count,
and branch-schema hash in addition to the six fields that define
`SourceFileRecord`; HCWDL had incorrectly expanded the richer row directly
into that narrower dataclass. A reusable explicit projection now preserves
the six split-identity fields while the source manifest and all of its richer
metadata remain independently content-hash authenticated. Both HCWDL and the
canonical split-building CLI use that projection, with a rich-row regression.
Focused HCWDL/split tests pass 70/70 and the repository-wide suite passes 300
with 14 warnings. That fix was committed as `659ebb6294038497d5da874f92beba05927021e8`
and exercised by the next fresh smoke identity.

The next smoke at commit `659ebb6294038497d5da874f92beba05927021e8`
passed both source/split boundaries and produced every train/validation
assignment shard, then stopped while aggregating job 56236. Real HLT data can
contain category flag rows that are not exactly one-hot; the established
decoder represents these tokens as category `-1`, and the broad matcher gate
correctly forces them to the HLT-preserving dustbin. Assignment v1 counted
these tokens in the dense HLT skeleton but had only five category counters,
making its own conservation equation impossible. Assignment shard/manifest
contracts advance to v2 and now record `unclassified_hlt_tokens` explicitly:
five-category visible totals plus unclassified equal all visible tokens,
five-category assigned totals equal all assigned tokens, and an unclassified
token is forbidden from carrying an offline assignment. Dustbin fraction
continues to cover the complete HLT skeleton, including these tokens, so the
strict `<10%` authorization criterion is unchanged.
Focused matcher/HCWDL tests pass 64/64 and the repository-wide suite passes
301 with the same 14 warnings. The v2 fix requires an exact clean commit and a
fresh Tigris smoke identity; v1 assignment shards must not be relabeled or
reused.

The v2 smoke then completed assignment manifest publication and the strict
assignment lock (`56351` and `56352`) before cache miniature job 56353 exposed
a separate bounded-stream bug. The miniature supplied the authenticated
4,096-row validation selection and also asked the PMARD stream for a
class-balanced `max_rows=4096`; that second uniform per-class quota is not the
same as the selection's authenticated natural class proportions, so it
discarded selected rows and could not cover the promised bound. PMARD streams
now expose an explicit `stream_prefix` bound policy. The cache miniature uses
that policy for D0 and D100, taking exactly the requested deterministic prefix
after row-selection filtering without rebalancing or changing training
semantics. The original `class_balanced` policy remains the default for all
existing callers. A skewed-class regression proves the prefix reaches its
exact bound where the former second quota could not.
Focused cache/stream/HCWDL tests pass 47/47 and the repository-wide suite
passes 302 with the same 14 warnings. The fix still requires a clean pushed
commit and a fresh smoke identity.

## Exact active next task: separately authorized HCWDL Tigris validation

Resolve and sign the immutable optimization recipe from independent evidence;
push an exact clean commit only under a separate authorization; run installed
Weaver parity and a genuine production-worker Tigris miniature; capture
measured RAM, walltime, I/O, GPU, and storage evidence; build a measured
resource profile and a fresh complete dry run; then request explicit pilot
submission authorization. Do not claim runtime acceptance or release the held
endpoint gate until those steps and the six real endpoint diagnostics exist.

## 2026-08-17: dense anchor-50 MHPE 300k/60-pass implementation

The additive implementation-authoritative
`docs/plans/HCWDL_MHPE_DENSE_ANCHOR50_300K60_PLAN.md` is implemented as
profile `C10P90_DENSE_ANCHOR50_300K60`. It imports the authenticated 300k
U000/M0paired foundation and registers 29 fresh cold fits over U033, U066,
U100, D75, D50, D25, D0, and M1. Six reducers use the frozen anchor-50 rule:
one half for the immediate-predecessor specialist and the remaining half
equally divided among skip specialists. The uniform reduction is computed
only as a validation diagnostic and is not published as a teacher.

Implemented surfaces:

- graph/coordinate/teacher/seed/finalist registration in
  `hcwdl_mhpe_graph.py`;
- v5 graph, node, recipe, campaign, training-report, and waiver identities;
- v3 300k foundation-reuse identity;
- v2 weighted target shard/manifest/lock, stage report, aggregate, finalist,
  completion, and sealed-final identities;
- exact-rational lexical FP32-softmax/FP64 weighted reduction with a single
  normalization and little-endian FP32 publication;
- profile-aware U033 logit publication, specialist training, reducer,
  aggregate, exact-HLT final evaluation, task attestations, monitoring, and
  failed/downstream source/resource recovery;
- 38-job Slurm plan: 29 fits, six reducers, aggregate, finalist lock, and
  completion, with 8 CPUs/96G/06:00:00/GH200 for GPU work and
  4 CPUs/32G/01:00:00 for CPU reporting;
- the versioned contract documentation and
  `docs/HCWDL_MHPE_DENSE_ANCHOR50_300K60_RUNBOOK.md`.

Existing MHPE v1-v4 graph hashes and uniform target semantics remain
unchanged. Deployable/finally evaluable nodes consume exact HLT only. Ordinary
execution has zero final-test access. No donor code was copied and
`docs/LEGACY_SOURCE_MAP.md` therefore did not change.

Final local evidence on the completed implementation:

- focused MHPE suite: 27 passed in 4.07 seconds;
- complete repository suite: 506 passed in 236.45 seconds, with the existing
  14 Matplotlib/Pyparsing deprecation warnings only;
- create/submit/recovery CLI help surfaces passed;
- contract probe reported graph
  `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v5`, graph SHA-256
  `4b53cace3e19dd8e4c3e3de09a747c821b46d9aa0f117a9a2b0919c0bcc53f1e`,
  campaign/recipe/training v5, 29 fits, six ensembles, and 38 tasks;
- all MHPE Python/CLI surfaces compiled;
- repository-relative Markdown links passed as part of the complete suite;
- `git diff --check` passed with line-ending notices only.

Per the active plan and explicit user direction, this additive profile carries
the completed foundation, installed-Weaver, production-worker, and prior MHPE
reducer/KD evidence and does not require another standalone smoke. This is an
operational waiver, not a claim that the new graph completed a smoke. No SSH,
Slurm submission, cancellation, push, or Tigris mutation occurred during this
implementation. The exact next task is to commit and push these changes, make
a clean detached Tigris worktree at that commit, create the canonical campaign
spec from the completed 300k foundation lock, materialize and inspect the
38-job dry-run ledger, and only then perform the separately authorized live
submission.

## 2026-08-17: paired dense C25P75 anchor-50 MHPE implementation

The additive `C25P75_DENSE_ANCHOR50_300K60` profile is implemented under
`docs/plans/HCWDL_MHPE_DENSE_C25P75_300K60_PLAN.md`. It is a paired loss-only
comparison against `C10P90_DENSE_ANCHOR50_300K60`: the same authenticated
300k/100k/100k population, 29-fit graph, six exact anchor-50 probability
reducers, teacher routes, coordinates, stochastic seed aliases, 60-pass
schedule, checkpoint policy, resources, and task dependencies are retained.
Every specialist changes from C10P90/T2 to C25P75/T2. M1 intentionally stays
C10P90/T1.

New semantic identities are graph/node/recipe/campaign/training-report/waiver
v6 and foundation-reuse v4. The weighted target/stage/aggregate/finalist/
completion/final-evaluation v2 contracts are reused because the reducer
semantics are byte-identical and every payload binds the new recipe profile.
The old v5 dense C10 graph SHA-256 remains
`4b53cace3e19dd8e4c3e3de09a747c821b46d9aa0f117a9a2b0919c0bcc53f1e`;
the new dense C25 graph SHA-256 is
`1cedac2dd9b94d576aa44cbfca0053386ae681127c3df902b287823a81b9d47b`.
An independent `hcwmhpe25d` Slurm namespace and `hcwmhpe25d_r` recovery
namespace prevent collision with the running dense C10 campaign.

Review evidence showed that all 28 specialist nodes differ only in CE/KD
weights and node-contract version; M1 differs only in node-contract version.
Task lists and exact ensemble weights compare equal. The complete synthetic
campaign path publishes and authenticates an independent 38-command ledger
with 29 fits, six reducers, aggregate, finalist lock, and completion.

Final local evidence:

- focused MHPE suite: 32 passed in 4.23 seconds;
- complete repository suite: 511 passed in 247.77 seconds, with the existing
  14 Matplotlib/Pyparsing deprecation warnings only;
- create, submit, and recovery CLI help checks passed and the create CLI lists
  `C25P75_DENSE_ANCHOR50_300K60`;
- all scouting and script Python surfaces compiled;
- contract probe reported graph/recipe/campaign/training v6, foundation-reuse
  v4, 29 fits, six ensembles, 38 tasks, identical paired seeds, and exact
  anchor-50 weights;
- repository-relative Markdown links passed in the full suite;
- `git diff --check` passed with line-ending notices only.

No donor code was copied, so `docs/LEGACY_SOURCE_MAP.md` did not change. No
SSH, Slurm submission, cancellation, push, or Tigris mutation occurred. Per
the active plan and user direction, no standalone smoke is required for this
loss-only paired profile. The next operation is a clean commit/push, detached
Tigris worktree, canonical campaign creation from the completed 300k
foundation lock, dry-run ledger inspection, and separately authorized live
submission using `docs/HCWDL_MHPE_DENSE_C25P75_300K60_RUNBOOK.md`.

## 2026-08-17: endpoint-refinement exact-HLT validator correction

Tigris job 89013 failed before inference because the validation-only D000E/M1
blend diagnostic looked for `scientific_config.node.input_domain`. PMARD
training reports serialize the registered HCWDL `NodeSpec` under that key,
whose authoritative field is `student_domain`. The failure therefore rejected
an exact-HLT M1 despite no privileged or offline input access.

The diagnostic now fails closed across three independent report facts:
`scientific_config.node.student_domain == "hlt"`,
`scientific_config.input_key == "hlt"`, and
`config.model_input == "hlt"`. A regression covers the accepted report and
each individual privileged-input substitution. Job 89005's durable endpoint
mixture targets completed successfully and are unrelated to this diagnostic
validator failure. No training, target generation, or final-test access must
be repeated; only the single validation diagnostic job needs a source-pinned
resubmission after commit and push.

## 2026-08-26: TRI60 five-seed CE ensemble reviewer control

The additive full-data reviewer control in
`docs/plans/HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md` is implemented.
It contains five fresh exact-HLT, CE-only, 60-pass Particle Transformers; one
fixed equal-probability reducer; one C10P90/T1 distilled student; and one
CE-only student with the exact same initialization, sampler, and training seed
alias as the distilled student. The graph therefore uses seven fresh fits and
directly tests the alternative explanation "ordinary seed ensemble followed
by KD" without changing or depending on scheduler state from the running
TRI60/dense ladders.

The implementation adds versioned graph, node, campaign, training,
probability, aggregate, completion, monitor, and recovery contracts. Only
canonical identity digests and 15 FP32 probabilities are durable teacher
targets; raw component logits, particle views, hidden representations, and
rolling-resume state are not persisted. Five CE teachers and the paired CE
control launch in parallel; only `CE5_KD` waits for the five-teacher reducer.
All models use exact HLT D000 inputs, the unified 21-channel ParT, the retained
full mapped foundation, 60 passes, batch 256, and the established TRI60
optimization/checkpoint semantics. Final test is absent.

The campaign has its own `hcwce5_` job names, worktree, checkpoint root,
command plan, exact dry/live journal, task attestations, and restart-from-zero
failed/downstream recovery. It never holds, cancels, reprioritizes, or writes
into another campaign. The queue procedure and result reader are in
`docs/HCWDL_TRI60_CE5_SEED_ENSEMBLE_RUNBOOK.md`.

Focused implementation evidence is 57 passing tests across the new CE5 suite
and the unchanged TRI60 suite. The complete repository suite is 711 passed in
325.28 seconds with 340 pre-existing Matplotlib/Pyparsing deprecation
warnings. This includes exact graph/loss and paired-seed checks,
hand-calculated probability averaging and identity joins, command-DAG
isolation, aggregate semantics, recovery closure, and worker boundaries. All
new Python/CLI surfaces compile; every create/run/submit/monitor/recovery CLI
help surface passes; both worker scripts pass `bash -n`; and repository link
validation passes in the complete suite. No donor file was copied, so
`docs/LEGACY_SOURCE_MAP.md` does not require an entry. No SSH, push, Slurm
submission, cancellation, or remote mutation occurred during implementation.

## 2026-08-27: standalone TRI60 M1 compression screen

The additive study in
`docs/plans/HCWDL_TRI60_M1_COMPRESSION_SCREEN_PLAN.md` is implemented. It
reports one imported `M1_LOGIT` control and runs 19 fresh exact-HLT,
full-mapped, 60-pass fits spanning cold/warm/polish initialization,
C10P90/C25P75/C50P50 plus one exact C75P25-to-C10P90 ramp, T=1/T=2, and peak
learning rates `3e-4`, `1e-4`, and `5e-5`. Warm initialization inherits only
the selected `LOGIT_D000_from_D033E` model state; polish inherits only the
selected `M1_LOGIT` model state. All optimizer, schedule, RNG, and sampler
state is fresh.

The source lock authenticates only the completed source subset required by
the screen: canonical campaign/foundation/recipe/endpoint evidence,
`LOGIT_D000E` probability artifacts and stage report, selected `M1_LOGIT`, and
selected `LOGIT_D000_from_D033E`. Source campaign aggregate/completion and
source Slurm job IDs are not dependencies. Source artifacts are read-only.
T=2 is the exact in-memory transformation
`softmax(log(p_LOGIT_D000E)/2)`; no component logits, copied probability bank,
particle views, hidden states, or representation targets are made durable.

The independent 23-job DAG is authenticate, preflight, 19 sibling GH200
fits, aggregate, and completion. It uses its own `hcwm1scr_` namespace, root,
ledger, task attestations, monitor, and restart-from-zero recovery. Every job
has Slurm nice 5000 so the opportunistic screen does not outrank the original
TRI60 ladder. Final test is absent, poor metrics cannot fail completion, and
no automatic top-three follow-up is launched. Per explicit user direction,
there is no standalone smoke requirement; established TRI60 production
evidence is carried forward.

Implementation evidence:

- the focused new/unchanged TRI60 suites pass 62 tests in 6.73 seconds;
- the complete repository suite passes 717 tests in 326.95 seconds with the
  existing 340 Matplotlib/Pyparsing deprecation warnings only;
- all new Python surfaces compile and seven create/run/submit/monitor/recovery
  CLIs pass their help/import checks;
- a synthetic canonical publication validates the graph/spec/command plan and
  a complete 23-task dry submission ledger;
- `git diff --check` passes apart from existing Windows line-ending notices.

The local Windows Bash executable could not be started, so both small worker
scripts still require `bash -n` in the clean Tigris worktree before live
submission. No donor file was copied, so `docs/LEGACY_SOURCE_MAP.md` is
unchanged. No SSH, push, Slurm submission, cancellation, or remote mutation
occurred. The queue procedure is in
`docs/HCWDL_TRI60_M1_COMPRESSION_SCREEN_RUNBOOK.md`.

## 2026-08-28: full-data TRI100 four-spine LOGIT path-density campaign

The implementation-authoritative plan in
`docs/plans/HCWDL_TRI100_FOUR_SPINE_LOGIT_IMPLEMENTATION_PLAN.md` is now
implemented locally. It registers four isolated immediate-parent-only paths
from one authenticated, read-only full-data U000 anchor:

```text
DIRECT:     U000 -> D000
COARSE:     U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:      U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
ULTRADENSE: U000 -> U020 -> U040 -> U060 -> U080 -> U100
                  -> D090 -> D080 -> D070 -> D060 -> D050
                  -> D040 -> D030 -> D020 -> D010 -> D000
```

The graph contains 29 fresh C25/P75, T=2 LOGIT-KD fits and 25
single-selected-checkpoint probability reducers. It contains no ensemble,
weight continuation, M1, final-test access, rolling resume, source completion
dependency, or source mutation. Same-coordinate seed domains are matched
across branches, including all four D000 endpoints.

The core training engine gained additive, opt-in support for
`warmup_hold_cosine_floor_tail_v1` and `macro_auc_patience_v1`. Existing
fixed-budget TRI60 behavior remains the default. This campaign alone binds
warmup passes 1--3, peak hold through pass 45, cosine decay through pass 60,
a constant 1.5e-5 tail through at most pass 100, minimum 60 passes, patience
15, patience delta 5e-5, and exact-best checkpoint restoration independent of
the patience threshold.

The campaign owns the `hcwsp4_` namespace, detached worktree, checkpoint root,
graph, recipe, source lock, command plan, journal, exact ledger, attestations,
monitor, and restart-from-zero recovery. Its four head fits depend only on its
own four-node NCCL acceptance task and can run concurrently. Each fit is a
four-node, one-GH200-per-node synchronous-DDP allocation; reducers remain
single-GPU. The global batch remains 256, every global batch is partitioned
without padding or duplication, and unequal final-rank row counts receive
exact loss weighting before DDP's gradient average. Rank zero alone evaluates
the complete canonical validation population and publishes artifacts. A
failed fit is recovered from update zero on all four ranks.

The campaign imports only the already complete U000 report/checkpoint/
probability bank and has no Slurm dependency on the source campaign. Durable
target banks contain identity digests and 15-class probabilities only;
particle views and optimizer/resume state remain RAM-only or absent. The
first registered GPU gate performs a real NCCL collective and DDP backward
pass, checks four distinct hosts and one visible GH200 per rank, and publishes
an immutable acceptance lock before any branch head may run.

The DDP extension has focused evidence for exact branch/teacher chains,
matched seeds, LR boundaries, minimum-pass early stopping, single-component
target round trips, source authentication, the 58-task isolated command DAG,
four-node resource projection, inherited-parent restart-zero recovery, exact
256-row and 255-row four-rank partitions, and a real two-rank CPU/Gloo training
comparison. The Gloo check exercises an uneven final global batch, rank-zero
publication, synchronized full validation, and final-weight parity with the
single-process global-batch reference. The final focused regression is 68
passing tests across the four-spine, unchanged TRI60, and view-cache suites.
The complete repository suite is 767 passed in 355.08 seconds with the 580
existing Matplotlib/PyParsing deprecation warnings only. All 17 relevant
Python surfaces parse through the AST, all seven create/run/submit/monitor/
recovery CLIs pass `--help`, both DDP Slurm workers pass `bash -n`, and
`git diff --check` passes with line-ending notices only. The queue procedure is in
`docs/HCWDL_TRI100_FOUR_SPINE_RUNBOOK.md`. Per user direction there is no new
standalone smoke campaign; live submission still requires canonical Tigris
source authentication, a full dry-run ledger, a clean exact pushed commit,
and success of the in-campaign NCCL acceptance gate. No remote job or artifact
was changed during local implementation.

## 2026-08-29: full-cardinality bottleneck-matching four-spine control

The implementation-authoritative plan in
`docs/plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md`
is implemented locally as a completely separate `hcwsp4b_` foundation and
science campaign. The matcher selects exactly
`min(n_hlt, n_offline)` one-to-one pairs. It lexicographically minimizes the
descending vector of delta-R values after 1e-7 ties-to-even canonicalization,
then applies the registered pT-response, category, charge, and native-index
tie breakers. The production solver first finds the exact bottleneck threshold
and prunes infeasible edges before its exact mixed-radix Hungarian solve; an
independent exhaustive reference covers brute-forceable cases. Forced-pair
validity is stored separately and is never represented as correspondence
confidence.

The new immutable foundation reuses only authenticated split, selection,
recipe, and established-source parents. It rebuilds the train/validation
assignment shards, diagnostics, audits, calibration/base coupling, balanced
sidecars, and all assignment-dependent locks. Assignment construction uses
bounded ordered process parallelism without persisting dense edge matrices.
Durable diagnostics cover per-jet cardinality, exact smaller-side coverage,
delta-R tails and ranked worst-pair profiles, multiplicity slices, old/new
pair overlap, common-pair and worst-pair deltas, and established confidence as
a separate historical quantity. A complete ordered train/validation P0 stream
comparison must prove identical tensors, labels, identities, checkpoint, and
probability-bank lineage before the established U000 anchor can be reused.

After that lock exists, the science DAG reproduces the established single-GPU
four-spine graph exactly: 29 fresh C25/P75, temperature-2 LOGIT fits and 25
single-component reducers, global batch 256, 60--100-pass floor-tail schedule,
early stopping, matched seeds, and immediate-parent-only teachers. The sole
declared scientific change is the pairing foundation. There is no ensemble,
M1, DDP, rolling resume, final-test access, dependency on running jobs, or
write path into the established campaign. Its aggregate reports accuracy,
macro AUC, macro R50, M0CE60-to-U000 recovery, complete per-class rejection,
and authenticated established-campaign rows when available; missing historical
rows remain pending rather than controlling completion.

The foundation and science paths each use versioned, content-hashed artifacts,
parent/source lineage, atomic publication, exact command plans, ledger
journals, task attestations, and bounded output inventories. The science path
also includes monitor and restart-from-zero recovery closure over exactly its
58-task DAG. Foundation and completion locks reject resume/optimizer files,
enforce projected durable-size and free-space reserves, and keep particle
views and dense matching state transient. No donor file was copied, so
`docs/LEGACY_SOURCE_MAP.md` is unchanged.

Local evidence is 16/16 focused tests, 119/119 focused-plus-neighbor tests, and
785/785 repository tests in 412.81 seconds; the only output was 580 existing
Matplotlib/PyParsing deprecation warnings. Ten create/run/submit/monitor/
recovery CLI help surfaces pass, and 29 new Python surfaces parse through the
AST. A local worker import audit caught and corrected one wrong balanced-switch
helper import. All three Slurm workers have a focused static environment/path
contract test. The Windows Git-Bash executable could not reliably start, so
the runbook requires authoritative `bash -n` in the clean detached Tigris
worktree before submission. Installed-Weaver real-row matcher acceptance,
measured production resources, all-row U000 equivalence, and the genuine
single-GH200 forward/backward check are registered fail-closed production DAG
gates and remain to run on Tigris. No push, SSH action, Slurm submission,
cancellation, reprioritization, or remote artifact mutation occurred. Exact
commit, foundation, audit, science submission, monitoring, recovery, and
storage commands are in
`docs/HCWDL_TRI100_FOUR_SPINE_FULLCARD_BOTTLENECK_RUNBOOK.md`.

## 2026-08-30: bounded full-cardinality matcher-acceptance repair

The first production foundation matcher-acceptance job, `97292`, reached its
eight-hour limit before publishing an acceptance lock. The failure occurred
before assignment production or any scientific fit. Source review identified
an operational bug in the acceptance miniature: while searching for eight
rare brute-forceable low-multiplicity rows, it ran the exact production
matcher on every selected row encountered. The final gate required only a
bounded real-row sample, so this accidentally converted candidate discovery
into an unbounded expensive matching scan.

Candidate discovery now reads only scalar count, selection, baseline, and
label branches. It deterministically stops after registering 64 ordinary real
rows and eight real rows with smaller-side multiplicity at most nine, then
performs targeted full-branch reads and exact matching only for the bounded
unique union. Low-multiplicity rows that overlap the first 64 count toward both
targets. The worker logs completed discovery and every eight matched rows and
records scan rows/timing separately from exact-matching timing. Matching,
cardinality, exhaustive-reference, provenance, and fail-closed scientific
semantics are unchanged.

The focused matcher/four-spine set passes 17/17, including a new regression
that proves overlapping low-multiplicity candidates stop discovery after 64
rows. The focused-plus-homotopy/unified-balanced/established-four-spine set
passes 120/120 in 59.05 seconds, and `git diff --check` passes with line-ending
notices only. No donor file was copied. The timed-out root must not be reused;
after this repair is pushed, cancel only its exact still-pending ledger IDs and
create a new source-pinned foundation root at the repair commit.

## 2026-08-30: exhaustive-reference cardinality repair

The second production foundation matcher-acceptance job, `97450`, completed
candidate discovery in 0.823 seconds but timed out after eight hours inside
the bounded matching loop. Its log reached 64/70 ordinary production checks
but completed only three exhaustive-reference checks. The reference filter
incorrectly admitted a row whenever the *smaller* side had at most nine
particles, even though its enumerator has
`P(max(n_hlt,n_offline), min(n_hlt,n_offline))` candidates. Consequently a
highly rectangular real row could pass the gate while requiring an
astronomical reference search.

The exhaustive helper now fail-closes unless both particle sides contain at
most eight particles, bounding a square comparison by 8! assignments. The 64
ordinary acceptance rows remain deterministic selected TRAIN rows and still
exercise the unrestricted production solver. The eight independent reference
rows are now searched across all authenticated TRAIN rows because this is an
integrity comparison rather than a scientific-population measurement; each
is explicitly bound on both sides and only those registered rows invoke the
exhaustive solver. The acceptance artifact records both populations and the
reference ceiling. Matcher objectives, assignment semantics, selected
training population, and the four-spine scientific comparison are unchanged.
Review also found that the preceding candidate-discovery edit had displaced
the tensor portion of the U000 stream-hash helper below an unreachable return.
Direct tensor equality still failed closed, but the durable digest bound only
identities and labels; the helper and regression now bind identities, labels,
features, vectors, masks, and raw lengths as originally intended.

Local evidence after the repair is 13/13 focused matcher tests, 130/130
focused-plus-neighbor four-spine/homotopy/unified-balanced tests in 98.93
seconds, and 789/789 repository tests in 558.98 seconds. The complete run
reported only the 580 existing Matplotlib/PyParsing deprecation warnings plus
the local read-only pytest-cache warning. A measured square 8-by-8 exhaustive
reference takes about 3.6 seconds locally, so the eight registered reference
rows have a deliberately bounded worst-case enumeration count. No donor file
was copied and no remote job or artifact was changed by this repair.

## 2026-09-01: persistent-HLT all-prior MT20 four-spine campaign

The implementation-authoritative plan in
`docs/plans/HCWDL_TRI100_FOUR_SPINE_PERSISTENT_HLT_MT20_IMPLEMENTATION_PLAN.md`
is implemented as an isolated `hcwmt20_` campaign. It uses the authenticated
full-cardinality bottleneck foundation, persistent-HLT support semantics, and
the same DIRECT, COARSE, DENSE, and ULTRADENSE path geometry as the current
persistent-HLT four-spine experiment. It retrains its own 60-pass CE-only
persistent-HLT U000 anchor and registers 29 downstream fresh fits.

Every downstream fit is MT20: 20% CE plus 80% temperature-2 probability KD
from every earlier node on the same spine. The immediate teacher receives 50
loss points; the older-teacher pool receives 30 points with an exact
nearest-first geometric ratio of one half. A one-teacher edge assigns all 80
KD points to that teacher. Teacher order and rational weights are graph-bound,
and every required ancestor reducer is an explicit Slurm dependency.

The 26 durable teacher banks contain only identity digests and 15-class
probabilities. Each fit constructs its weighted target once in float64 RAM,
converts it to float32, and persists only a small lineage registry—not the
mixture array, particle views, hidden states, optimizer state, or rolling
resume. The campaign has 61 tasks, its own roots/contracts/job namespace,
single-GH200 execution, exact-ledger monitoring, and restart-from-zero
recovery. It has no scheduler dependency on, cancellation/reprioritization
authority over, or write path into any existing campaign. Existing
immediate-parent reports are validation-only controls and may remain pending.

Local evidence is 48/48 focused-plus-neighbor tests, covering exact weight
examples, all-prior same-spine closure, RAM mixture order and normalization,
non-publication, the isolated 61-task DAG, restart-zero recovery, unchanged
full-cardinality persistent-HLT behavior, and matching repair. Python syntax
compilation passes for the new production surfaces. No donor file was copied,
so `docs/LEGACY_SOURCE_MAP.md` is unchanged. No SSH, Slurm submission, remote
artifact mutation, job cancellation, hold, or reprioritization occurred.
Authoritative installed-Weaver/Tigris preflight and measured production
resource evidence remain required in the campaign DAG before scientific fits.

The first production preflight, job `98595`, failed before any fit because
the ordinary unified-balanced stream exposes authenticated `identity_keys`,
not precomputed `identity_digests`. The gate now derives the exact canonical
cache-compatible uint8 `[rows,32]` digests from those keys before exercising
the multi-teacher RAM mixture. A focused regression binds this ordinary stream
shape. No scientific output was produced, and the original pending DAG remains
dependency-blocked; recovery must be source-pinned and restart the preflight
from zero.

Source-pinned recovery commit `865fa09b4155eeee1974d91e631375728cb2a74f`
then passed production preflight as Tigris job `98738`. Its first U000 fit,
job `98739`, completed the 2,777,855-row train cache in 950.456 seconds and the
957,541-row validation cache in 479.990 seconds, then failed before pass one.
The cause was a non-timing `ram_teacher_count` entry incorrectly placed in the
shared trainer's timing-only `preparation_metrics` registry. MT20 now supplies
only `student_view_cache_seconds` and `pre_training_total_seconds`; teacher
cardinality remains graph- and mixture-registry-bound. The focused
source-pinned MT20 suite passes 7/7 after adding a regression for the exact
timing registry. No model checkpoint or scientific result was produced by the
failed fit, so the next exact-ledger recovery restarts U000 from zero.

## 2026-09-01: Phase-I persistent-HLT attention-reoptimized four spines

The Phase-I design in
`docs/plans/HCWDL_DISTILLATION_GUIDED_ATTENTION_REOPTIMIZATION_PLAN.md` is now
implemented as a separate `hcwsp4a_` full-four-spine campaign. It retrains the
corrected full-cardinality persistent-HLT `SP4P_U000` anchor, then executes all
29 downstream DIRECT, COARSE, DENSE, and ULTRADENSE fits. The existing
persistent-HLT campaign is read-only matched comparison lineage and does not
need to finish before this campaign is created.

Each downstream fit runs 60 ordinary fresh C25/P75 temperature-2 passes,
restores its Stage-0 selected checkpoint, runs 15 passes with only the
installed Weaver pair embedding and eight particle-attention modules
trainable, then runs 25 all-parameter joint-refinement passes. Stage A and B
add an all-eight-block, transported-token-aligned residual-delta Gram loss
from the immediate-parent selected model plus a trust region. The teacher is
live, frozen, evaluation-only, and batch-local. No dense attention target,
hidden-state bank, optimizer state, or rolling resume is durable; only compact
15-class probability banks are persisted.

The implementation adds versioned graph/recipe/specification, parameter-lock,
execution-acceptance, training/checkpoint, stage, aggregate/completion,
monitor, and recovery artifacts; a structural installed-Weaver parameter
compiler; training-only complete-block delta surfaces that leave ordinary
forward unchanged; exact freeze/unfreeze optimizer rebuilding; full reporting;
and restart-from-zero recovery. Fit jobs request one GH200, 72 CPUs, 500 GiB
RAM, and five days because the child train/validation caches and immediate-
parent train cache coexist in memory.

Submission is deliberately split. A three-task authenticate/support/preflight
gate must publish valid genuine-GH200 and exact-gradient acceptance locks
before the 58-task science plan can be released. Gate and science ledgers are
then merged into the exact 61-task monitoring/recovery ledger. Python syntax
compilation passes across the production surface. Local evidence is 12/12
new attention tests, 29/29 attention-plus-four-spine tests, and 122/122
TRI60/homotopy/full-cardinality neighbor tests. The full local suite is
817/817 passing (one unrelated existing PyTorch scalar-conversion warning).
Installed-Weaver parity and the genuine Tigris gate remain required before
science submission. No donor file was copied, so `docs/LEGACY_SOURCE_MAP.md`
is unchanged. No remote job or artifact was changed.

The first genuine-GH200 attention gate authenticated and completed its full
support audit as jobs `98852` and `98853`, then preflight job `98854` failed
before publishing either the parameter lock or execution-acceptance lock. The
ordinary unified-balanced parent and child streams expose authenticated
`identity_keys`, while the new attention acceptance boundary incorrectly
required an optional top-level `identity_digests` array. The boundary now
derives the exact canonical cache-compatible uint8 `[rows,32]` digests from
the keys and independently authenticates any supplied digest array. A focused
regression covers the ordinary missing-digest shape and corrupt supplied
digests. No fit, probability bank, checkpoint, or scientific result was
produced. Because the source pin changes before any science submission, the
failed root and ledger remain immutable and a fresh source-pinned three-task
gate reruns authentication, support audit, and preflight before science is
released. Local verification is 13/13 attention tests, 133/133 affected and
neighboring persistent-HLT tests, and 819/819 repository-wide tests; only
pre-existing third-party plotting deprecation warnings remain.
