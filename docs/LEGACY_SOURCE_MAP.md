# Legacy Donor-Source Map

## 2026-09-22: K2 deterministic parity and layout-preserving saved tensors

Repository-local donor/baseline: `49516092a646cfe21f7bb1377b07a142161eb3d2`,
the failed 21767292 execution. Modified `concat_k2_model.py` storage hooks and
parity harness; `concat_k2_runtime.py` composes a new early gate from existing
`acceptance.installed_parity`, `concat_k2_data.load_assignment`, `DatasetReader`,
`build_view`, `build_inputs` and `RamCache`. New `concat_k2_parity.py` reuses
those APIs without copying or editing their matching/data producer files.
The K2 worker sets preflight-only cuBLAS configuration; registration binds new
execution policy versions. Native model, optimizer, matching and science
recipe are unchanged. Original completed-map donor remains
`1f9306504dfd040c9c22e0a89829d277d1ff2194`, subject to the full reuse checks.

Tests reuse repository synthetic ROOT, completed-donor and model fixtures;
new tests cover stride/value preservation, deterministic-state restoration,
bounded authentic training-row sampling and early failure. Weaver 0.5.3 was
installed into an isolated scratch directory for additional local validation;
no Weaver/PyTorch source was vendored, no external weights imported, and no
third-party license/attribution changes were introduced. Results and the
remaining SPORC acceptance boundary are recorded in `docs/HANDOFF.md`.

## 2026-09-22: authenticated K2 compact-assignment reuse

Baseline: `df29abcce078b2be80bd465b22c28066ec32d8f6`. Original reusable K2
producer: `1f9306504dfd040c9c22e0a89829d277d1ff2194` (only after its completed
foundation is authenticated). New repository-local
`jetclass2_delphes/concat_k2_preparation_import.py` composes the existing
`concat_k2_data.load_assignment/foundation_lock/producer`,
`concat_k2_runtime.completed`, and `concat_k2_source.validate_import`
validators. Those matching/data producer files and their byte identities are
unchanged by this block. No external donor code or trained weights are copied.
Campaign/source/runtime and thin CLI/queue helper wire the optional import;
launch/campaign become v4, PREPARATION_IMPORT is v1, GPU acceptance remains v3.

## 2026-09-22: K2 pair saved-tensor storage and 128/256 probes

Repository-local donor/baseline: `b35fbda64d2d823a9eb9c5592017074db58d6ac8`.
The failed original K2 execution was `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
`concat_k2_model.py` adapts the saved-tensor CPU-storage pattern from committed
`jetclass2_delphes/dzfix_fusion_model.py` to the single native K2 encoder; it
does not import or modify that fusion adapter. Native base is `model.py`;
training/optimizer are reused from `salience_learned_training.py` unchanged.
`concat_k2_runtime.py` adds ordered probes and evidence validation;
`concat_k2_campaign.py` registers the storage/probe policies and v3 execution
artifacts. No external code or scientific model weights are copied.

`tests/test_jetclass2_concat_k2_memory.py` reuses committed `PairBN`/physical
input fixtures from `test_jetclass2_dzfix_fusion_offload.py`, and existing
`make_cache`/`fake_native` fixtures ultimately based on
`test_hcwdl_offline_hlt_fusion.py::_FakeWeaver`. They test a K2-only adapter,
not a fusion architecture. PyTorch public `save_on_cpu`/`saved_tensors_hooks`
are composed without vendoring. Existing partition-portability changes are
retained; no other campaign runtime is changed by this repair.

## 2026-09-22: dzfix fusion pair saved-tensor memory repair

Repository-local baseline/donor commit:
`6172f5be459f0a0f7b6c3ed99ee3c6f8ac7e542c`; the failed remote execution was
`2f0afe5c438cbab5d66e047c4d86534aab02032b` (preflight 21757056).
Modified `jetclass2_delphes/dzfix_fusion_{model,chain,runtime}.py` and added a
native-default pair-embedding hook to `salience_learned_model.py` at that
baseline. `tests/test_jetclass2_dzfix_fusion_offload.py` reuses the repository's
existing fusion-chain RAM fixtures and `_FakeWeaver` from
`tests/test_hcwdl_offline_hlt_fusion.py`, adding a trainable full-pair BN test
double. No external source is copied. PyTorch's public `save_on_cpu` and
`saved_tensors_hooks` APIs are composed, not vendored.

Scope: storage of tensors saved inside the three pair embeddings only; full
pair populations, BN computation, dtype, batch, loss, model keys and data stay
unchanged. Tests cover three optimizer updates in CPU/FP32 CUDA/BF16 CUDA,
actual pinned copies/retrieval, BN tampering, checkpoint transparency,
inference bypass and hook cleanup. LAUNCH_SPEC/CAMPAIGN_SPEC become v4,
ACCEPTANCE v2; SOURCE_IMPORT remains v3 so completed matching is reused via
the same authenticated read-only boundary. Real Weaver/A100 acceptance is
still required; local test doubles cannot certify it.

## 2026-09-22: K2 tier3/debug execution portability

Repository-local donor commit: `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
Adapted existing `jetclass2_delphes/concat_k2_{campaign,source,submit,runtime}.py`,
`scripts/jetclass2_concat_k2.py`, `scripts/queue_jetclass2_concat_k2.sh`,
`sbatch/run_jetclass2_concat_k2.sh` and their two test files. New K2-scoped
`concat_k2_execution.py` composes the existing `execution.execution_site`
profiles and strict `allocation`/`gpu_identity` validators without altering
those shared files or other campaigns. No external source or scientific
kernel is copied. New policy defaults to tier3 and accepts only partition-only
pending-job moves to/from debug with identical resources/hardware/software.
Launch/campaign/acceptance are versioned v2; view/matcher/seed identities are
unchanged. Tests add actual/requested-site and cross-partition gate coverage.

## 2026-09-21: isolated dzfix K2 concatenation ladder

Repository-local donor commit: `2f0afe5c438cbab5d66e047c4d86534aab02032b`.
No external repository, data, weights, generated campaign or Fresh_check
runtime import is copied. The K2 matcher/view semantics and graph are new.
Local donor-byte SHA-256 values below precede the LF normalization used while
adapting new files; the Git commit identifies the canonical source revision.

| Donor file | Local donor SHA-256 | New adapted surface |
|---|---|---|
| `jetclass2_delphes/dzfix_fusion_source.py` | `d3a888c251d45ffc053178c4d2124437b8ebfa63209ad88ff3ee6313a665ca94` | `concat_k2_source.py` |
| `jetclass2_delphes/dzfix_fusion_submit.py` | `da30bff454ce8914e3436d05d7b31001566e11a7c6973988747d11d2d3fc2fcf` | `concat_k2_submit.py` |
| `jetclass2_delphes/dzfix_fusion_runtime.py` | `5d9ccefc7faff6132e49e2b30689ad263a9e2cc9476419ef0584c87d0f4c8603` | `concat_k2_runtime.py` |
| `jetclass2_delphes/dzfix_fusion_data.py` | `dc00aad07a7a2d6a2b309caa9f13ce6c9bafa6bc1a5713fc172acb4921cca94c` | bounded-cache/partition patterns in `concat_k2_data.py` |
| `scripts/jetclass2_dzfix_fusion_chain.py` | `6966f734f2bd9f935680a9962d221eea5cf38a7dec9a46cea064a4b63809e9b2` | `scripts/jetclass2_concat_k2.py` |
| `scripts/queue_jetclass2_dzfix_fusion_chain.sh` | `92add0f6e4f241d38b93d73a8befd8592b32f34bdd7d19c5944cf23c33989107` | `scripts/queue_jetclass2_concat_k2.sh` |
| `sbatch/run_jetclass2_dzfix_fusion_chain.sh` | `c0a4a42a3b0342393d32cb43ac7d4913046c7a8ad5f1361ae135e47d5f8a0c8b` | `sbatch/run_jetclass2_concat_k2.sh` |
| `tests/test_jetclass2_dzfix_fusion_chain.py` | `718920b0fc25adcb1fdade60b5e9bf027465045d0d9f132eae15f643f3381bb1` | adapted source tests and reused synthetic fixtures |

Package paths in this table are under `src/hlt_classification/`. Retained:
read-only screen/ledger authentication, immutable task receipts, staged exact
submission, validation-role partitioning, ordinary train-bank KD and selected
checkpoint reporting. Changed: no dual-view model or weight/map reuse; new
capacity-two preparation, fixed 3*N_HLT support, K2-only schemas/namespaces,
ten-fit D-only graph and expanded-input GPU acceptance. All new jobs use debug.

Composition (not file migration) additionally uses the same revision's
`salience_views.pairing_matrices`, `scouting/hcwdl_fullcard_salience_matcher`
exact solver/reference, its versioned salience contracts, and the common
`reader`, `inputs`, `cache`, `model`, `banks`, `reporting`, `execution`,
`salience_learned_training`, `salience_learned_graph`, `salience_learned_data`,
`acceptance`, `submission`, provenance/cache-contract and exact-DAG helpers.
The full-native salience is frozen before new retention/duplication; the old
one-to-one artifact is never reused. Runtime source fingerprints include
actual preparation bytes, so unrelated dirty worktree edits are not falsely
claimed as donor-commit evidence or silently accepted as old artifacts.

Tests: `test_jetclass2_concat_k2.py` (exhaustive matching, typed views, real
synthetic ROOT preparation, CPU test-double DAG, deployment, gates) and
`test_jetclass2_concat_k2_source.py` (screen imports, dry/live plan isolation,
receipts, bad-parent failure injection). No new third-party dependency or
license obligation. Local CPU tests do not certify installed Weaver/SPORC.

## 2026-09-21: isolated JetClass2 dz-fix fusion-to-fusion debug chain

Internal code donor: `f1753bd1f26dd2bffe36d21fc937aeb663cab00d`.
New `jetclass2_delphes/dzfix_fusion_{chain,source,data,model,runtime,submit}.py`
uses the donor's `salience_learned_{graph,training,model,data,cache}.py`,
`cache.py`, `banks.py`, `execution.py`, `reporting.py`, `model.py`,
`salience_foundation.py`, `salience_screen.py`, `salience_production.py`,
`dzfix_salience_continuation.py`, `submission.py` and the shared exact-DAG
submission/authorization/cache-contract helpers. No shared graph or loss is
redefined. A default-no-op injection-bias hook in `salience_learned_model.py`
lets the new adapter apply `cms_salience_learned/model.py`'s temporary-mask
optimization while preserving historical adapters and full pair-BN semantics.
The native pure-OFFLINE adapter uses `cache._prepare_file` at its exact native
offline endpoint; persistent U000 remains a different view.

The scientific graph specializes the prior local CMS fusion-chain plan; that
prior uncommitted CMS extension is not falsely attributed to the donor commit.
No CMS model, acceptance, metric, split or matching artifact is imported.
Data/preparation selection donor: the source-pinned replacement v2 debug
screen at `0d25a4a53aafb1348c8279d86bac7dbac82c8841`. Its unchanged compact
foundations retain their original producer fingerprints. The updated consumer
reuses this commit's `salience_screen.py` v2 validator/command plan/task reports
and `salience_production.py::_screen_artifacts`; winner and exact foundation
paths/hashes are resolved from the screen, never named in advance. The canceled
46f10c7 continuation/production-preview route is no longer a parent.
The consumer orchestration baseline is `cde3ed8` (the committed initial
dz-fix fusion chain); only its source adapter, submission boundary, CLI/helper,
input-contract versions and tests are changed. Scientific kernels are not.

Thin CLI/worker/queue helper derive environment handling from
`sbatch/jetclass2_delphes_common.sh` and staged exact submission from
`salience_learned_autolaunch.py`, but all new jobs are explicitly debug.
Tests reuse only the synthetic `_FakeWeaver` from
`tests/test_hcwdl_offline_hlt_fusion.py` and particle fixtures from
`tests/test_jetclass2_delphes.py` (same code donor).
Capacity-fix baseline: `e3b02d6f6f0f3b6433230603a2438767b69e25fd`.
The consumer adapter now verifies the same inventory-maximum/round-up-to-16
capacity calculation as that commit's `jetclass2_delphes/foundation.py` and
`salience_foundation.py`, and uses `inputs.py::input_contract` unchanged.
Current dzfix capacity is 320, not the stale 240 inherited in the handoff.
Those shared producer files and all assignment payloads remain unmodified.
Tests explicitly preserve >240-particle native inputs, use 320 in pending and
completed source imports, and exercise capacity-driven cache/stress behavior.
Family: `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*`; LAUNCH_SPEC, SOURCE_IMPORT,
CAMPAIGN_SPEC advance to v3 for the inventory-derived no-truncation policy
(v2 introduced direct-screen provenance); other kinds stay v1.
No external source code copied.

## 2026-09-20: isolated CMS adjacent fusion-to-fusion KD chain

Internal implementation donor `f1753bd1f26dd2bffe36d21fc937aeb663cab00d`:
`src/hlt_classification/cms_salience_learned/{contracts,campaign,production,
preparation_import,shared_import,preflight_reuse,coarse_submission}.py` and
`scripts/cms_salience_learned.py` are extended for a new fusion-chain consumer.
New `fusion_chain.py` derives the shared-reference/first-acquisition nodes
from the donor's coarse graph, authenticates completed acquisition outputs,
and provides the new graph/import/report orchestration. Scientific model,
training, cache, publication/reduction and worker kernels remain unchanged.
New `scripts/queue_cms_fusion_chain.sh` derives its environment/staged workflow
from `scripts/queue_cms_direct_fusion.sh`, without cancellation. New tests
reuse `test_cms_direct_fusion.py`, `test_cms_salience_learned.py` and
`test_hcwdl_offline_hlt_fusion.py` fixtures from the same donor.
No external source code was copied.

Scientific execution donor remains accepted dense
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (genuine SPORC preflight 21720795),
with original preparation `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
First acquisition data donor is the explicit completed coarse-v5 campaign
at `7cf690301d66b1d02d98b450d20aababf296429a`; its spec, live ledger, exact
model/teacher identities, receipts and artifact bytes are verified at reuse.
CAMPAIGN_SPEC/v7, GRAPH/v4, PREPARATION_IMPORT/v4, SHARED_SOURCE/v3,
ACCEPTANCE_IMPORT/v3 and ACCEPTANCE_REUSE/v3 bind this separate consumer.
ACQUISITION_SOURCE/v1 and FUSION_CHAIN_AGGREGATE/v1 add explicit import and
two-endpoint reporting semantics. Old campaign graphs/artifacts stay unchanged.

## 2026-09-19: isolated CMS direct fusion/withdrawal comparison

Internal implementation donor `7cf690301d66b1d02d98b450d20aababf296429a`:
native CMS `contracts.py`, `campaign.py`, `preparation_import.py`,
`shared_import.py`, `preflight_reuse.py`, `coarse_submission.py` and
`scripts/cms_salience_learned.py` are extended for a separate direct consumer.
New `scripts/queue_cms_direct_fusion.sh` uses the environment setup from
`scripts/switch_cms_salience_coarse.sh` at that donor, but not its cancellation
workflow. New regression tests reuse synthetic fixtures from the native CMS,
coarse and fusion tests at the same donor. No external code is copied.

Scientific execution donor remains accepted dense commit
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (SPORC GPU job 21720795), with
original preparation `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
Native model/training/data/production kernels and the worker shell are reused
unchanged, not migrated to a new runtime. CAMPAIGN_SPEC/v6 and GRAPH/v3 declare
the one-arrow U000/D000 study. New preparation/shared/acceptance import
versions bind this consumer without changing coarse replacement artifacts.

## 2026-09-19: CMS reuse AST encoding compatibility fix

Internal baseline `4f862c11045943f1237d1cef166d32a85c341ed3`;
original preparation producer `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`,
accepted runtime donor `7bb171382b7206013bc5d9308a4c22b2929bc7f4` and coarse
registry donor `48ab8609ee87ba72ab9868dfc951a36a1c9d851e`.
Extended only the exact reviewed function-hash pairs in native
`preparation_import.py` and `preflight_reuse.py` to cover Python 3.10's
empty-list-emitting AST dump in addition to Python 3.13's original encoding.
Both are derived from the same immutable donor functions; no external source
copied, no stored fingerprints rewritten, no scientific contract version changed.
Regression changes are in the existing coarse and preflight-reuse test files.

## 2026-09-19: explicit dense-to-coarse execution evidence reuse

Internal baseline `48ab8609ee87ba72ab9868dfc951a36a1c9d851e`; execution donor
`7bb171382b7206013bc5d9308a4c22b2929bc7f4` (accepted SPORC preflight 21720795).
Added `cms_salience_learned/preflight_reuse.py` and focused tests; extended
native contracts/campaign/dispatch and the two operator CLIs. No external
code was migrated and no training/model/cache kernel changed.
New CAMPAIGN_SPEC/v5 plus ACCEPTANCE_IMPORT/v1 and ACCEPTANCE_REUSE/v1 record
explicit user-authorized reuse of the old measured envelope. Source evidence
is never rewritten as a new measurement. Git identities cover the same native
reference kernels, shared models/data, worker shell, allocation adapter and
preflight probes; the sole allowed driver AST migration is its schema-version
predicate, previously introduced in coarse commit 48ab8609. V4 fresh-gate and
all older scientific/operational meanings remain unchanged.

## 2026-09-19: native CMS coarse graph and authenticated shared references

Internal baseline: `7bb171382b7206013bc5d9308a4c22b2929bc7f4`, accepted dense
SPORC preflight 21720795. Extended native `cms_salience_learned/contracts.py`,
`campaign.py`, `production.py`, `preparation_import.py`, the native CLI and
tests. Added `shared_import.py`, `coarse_submission.py`, coarse tests and the
staged operator shell helper. No external code was copied. The exact journal
pattern and ledger/event schemas reuse internal
`scouting/hcwdl_exact_dag_submission.py` and `scouting/hcwdl_recovery.py` at the
same donor commit; those shared modules are not changed.

New native CAMPAIGN_SPEC/v4, GRAPH/v2, PREPARATION_IMPORT/v2 and
SHARED_SOURCE/v1 distinguish the coarse graph and reviewed imports. Retaining
the original family seed namespace is intentional. Preparation Git objects
from original producer `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea` are confirmed
identical to the accepted dense source; only the closed addition of coarse
coordinate names gets an explicit AST migration. Reference training/model/
cache modules and scientific worker/seed/schedule ASTs remain unchanged.
Neither imported source files nor the previous GPU gate are rewritten.

## 2026-09-18: native CMS-LFH compact shared cross-attention bias

Internal baseline: `77c9d2b1fe17a9fb321f1f85ebf9cd803c0f5ec1`, source
of SPORC debug preflight 21720511. Extended native
`cms_salience_learned/model.py` and shared
`models/hcwdl_offline_hlt_fusion_transformer.py`; added
`tests/test_cms_fusion_temporary_memory.py`. No external donor was copied.
The shared base exposes a legacy-preserving bias-preparation hook; only the
CMS subclass opts into merging padding once and sharing its compact rectangle
across the four injections. Weaver's full pair construction is not altered.
Other model adapters retain the old path. Scientific/state-dictionary and
execution contracts are unchanged; exact new source still requires its own
installed-Weaver checks and genuine A100 acceptance. Preparation-code Git
objects, graph, matching, loss, batch, schedules and seeds are untouched by
this patch, permitting authenticated read-only original preparation reuse.

## 2026-09-18: native CMS-LFH versioned 90% CUDA acceptance

Internal donor baseline: `9af094b08cdc112a8a5048374fee4f135b704087`, source
of SPORC debug preflight 21719837. Extended native
`cms_salience_learned/contracts.py`, `campaign.py`, `production.py` and
`tests/test_cms_salience_learned.py`; no external code was migrated.
Campaign execution v3 and execution acceptance v2 register the user-authorized
90% CUDA / unchanged 85% CPU policy and consecutive worst-length, batch-256
withdrawal probes. Legacy campaign v1/v2 keeps its 85% CUDA policy. Scientific
graph/artifacts remain v1. Training kernel, preparation modules, coordinate
semantics, matching, model, loss, batch size and schedule are unchanged.
Original completed preparation remains reusable read-only; no prior failed
gate or source campaign is modified or treated as newly accepted.

## 2026-09-18: native CMS-LFH preflight lifetime and memory diagnostics

Internal donor baseline: `85fd0214178227026625dd3a5d2b9c79a5fbe2a7`, source
of SPORC preflight 21719606. Refactored that revision's native
`cms_salience_learned/production.py` miniature loop into per-route function
scope and extended `tests/test_cms_salience_learned.py`. No external code
was migrated. The production training kernel, preparation code, matcher,
models, batch size, losses, schedules and registered 85% memory thresholds
are unchanged. Added console diagnostics, not a new scientific artifact
contract. A weak-reference regression reproduces the old acquisition
optimizer retaining parameters into the next route; failure injection checks
that CPU/GPU headroom failures remain fail-closed and publish no acceptance.

## 2026-09-18: native CMS-LFH paired-cache extraction inference repair

Internal baseline: `08c3647981ad6dac54d3cba30212a2515f839915`, the pinned
source of failed SPORC preflight 21719164. Extended native
`src/hlt_classification/cms_salience_learned/training.py` and
`tests/test_cms_salience_learned.py`; no external donor migration. Ordinary
inference now requests the primary-only batch even when a cache retains context.
The existing `data.Cache.batch_primary` API is reused unchanged. Models, loss,
matching, schedules, graph and byte-exact extraction semantics are unchanged;
no scientific contract/version change. Local tests cover the actual preflight
call with a paired cache and extend the optional installed-Weaver parity check.

## 2026-09-18: native CMS-LFH partition routing and preparation reuse

Internal donor checkout: `f966dd804ed9ca8ca93c9c3227d1a7f036ef4434`.
Extended `src/hlt_classification/cms_salience_learned/campaign.py`,
`contracts.py`, `production.py`, `scripts/cms_salience_learned.py`, and
`tests/test_cms_salience_learned.py`; the new `preparation_import.py` is local
implementation, not an external copied donor. Reused scheduler/environment
checks from `src/hlt_classification/jetclass2_delphes/execution.py` unchanged,
including its already registered `sporc_a100_debug` allocation profile; no
Delphes data/model semantics or profiling-to-production transfer are imported.
Publication continues to use native `storage.py` and `data/cache_contracts.py`.

The completed producer motivating reuse is native CMS commit
`f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`. Import validates the selected
producer's actual source commit, not this documentary example: native
`data.py`, `storage.py`, the full scouting/data Git trees, and the coordinate
function AST must match the consuming commit. Reusing preparation preserves
its original hashes and does not execute code from the old worktree.
`CAMPAIGN_SPEC` advances to v2 and `PREPARATION_IMPORT/v1` is added;
scientific graph, models, seeds, losses and remaining v1 artifacts are unchanged.

## 2026-09-17: CMS2JC2 staged production tooling and diagnostics

Same internal donor commit: `2b4c2531c39118edebc8c7890d312f2285ca8eab`;
the exact reused files remain enumerated below and in `provenance.DONORS`.
`campaign`, `submission`, `orchestration`, `measurement`, `tasks`,
`synthetic_acceptance`, `diagnostics` and `plots` are new repository code, not
copied donor implementations. Publication still uses internal cache contracts.
Vector example plots are generated directly, with no external assets. Installed
NumPy/SciPy/Awkward/Uproot/scikit-learn/threadpoolctl/matplotlib and native numeric
library bytes are recorded by acceptance; no environment was installed or changed.

## 2026-09-17: CMS2JC2 provisional integrated response development

Same internal donor baseline: `2b4c2531c39118edebc8c7890d312f2285ca8eab`.
The response continues using the internal sources listed below. The new storage
guard additionally reuses `data/cache_contracts.py::atomic_publish_bytes`;
held-out non-selecting class diagnostics reuse the existing native label mapping.
Assumption contracts, record sampling/parallel quotas, topology/generation,
conditional summaries, paired selection, support/transfer, graph and quota
logic are new code, not migrated sibling-worktree or external implementation.
No old matcher, classifier, active campaign, external environment or raw dataset
was changed. Portable fitted previews are local development evidence only;
this extension is not yet a full-science queue-ready campaign.

## 2026-09-17: CMS2JC2 response preparation and calibration primitives

Internal donor baseline: `2b4c2531c39118edebc8c7890d312f2285ca8eab`.
The new isolated namespace is `src/hlt_classification/cms2jc2_response/`.
No external source, sibling-worktree module, fitted classifier, or previous
matcher/calibration artifact is imported. Existing dirty scientific kernels
were preserved, not copied into this response.

- `data/cache_contracts.py`: canonical hashes, immutable JSON publication and
  validation, file checksums.
- `scouting/{schema,labels,splits}.py`: native branch names, exact historical
  baseline/mapped label selection, and historical manifest authentication.
  Labels authenticate frozen populations; they are not response predictors.
- `jetclass2_delphes/{inventory,contracts,schema,splits,split_registry}.py`:
  source/checksum validation, canonical row identities and frozen profile
  membership masks. The paired `reader.py` was inspected but deliberately not
  reused: the new offline-only reader must work without native HLT branches.

Association search, counter RNG, role allocation, conditional-model cores,
residual backend and guarded preparation submission are new implementation.
Numerical libraries are imported, not vendored. The optional `response` extra
declares scikit-learn/threadpoolctl without GPU/Weaver dependencies. Production
versions still need a recorded SPORC environment/resource acceptance. This is
partial implementation, not full-science readiness.

## 2026-09-16: bounded JetClass2 raw jet-pair audit

Internal donor snapshot: `9640bae8147c8e78ae14e165cb969b066a23baea`.
`jetclass2_delphes/pairing_audit.py` reuses `reader.py::_particles`,
`inputs.py::{eta_phi,wrap_phi}`, inventory checksum/schema validation,
selection policy, split-profile masks, canonical identities, source-byte
provenance and immutable JSON publication. It adds bounded diagnostic sampling
and class/source/kinematics-preserving shuffle controls. No external code is
copied, no constituent matcher or training kernel is changed, and no model
performance or physical-pairing certification is implied by completion.

## 2026-09-16: JetClass2 native concatenation, one-job oracle

Internal donor snapshot: `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
New adapters are `jetclass2_delphes/native_concat{,_data,_model}.py`.

- `jetclass2_delphes/{reader,inputs,cache}.py`: native paired reads, canonical
  per-reconstruction 17-feature transforms, packed blocks and bounded ordered
  process preprocessing. New batches add a discrete source transport channel
  and retain the sum of both native particle counts, without any matcher.
- `jetclass2_delphes/model.py` and installed Weaver: unchanged canonical
  backbone; only the numerical embedding is wrapped to add a learned source
  embedding. No Weaver source is copied. The earlier conceptual donor
  `models/hcwdl_tagged_concat_transformer.py` is not imported (wrong dataset).
- `jetclass2_delphes/salience_learned_{graph,training,data,production}.py`:
  unchanged CE schedule, matched control seeds, checkpoint selection, held-out
  validation partition and immutable completed-reference authentication.
- `jetclass2_delphes/{execution,submission}.py` and
  `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}.py`: explicit SPORC
  site, guarded exact-ID intent/receipt ledger and canonical dry-run behavior.

New scientific exception is scoped to `JETCLASS2_DELPHES_NATIVE_CONCAT_*/v1`:
native offline+HLT source tags are authorized only for this privileged oracle.
No existing campaign, matching strategy, input schema, or training kernel was
edited. Real installed-Weaver 0.5.3 CPU parity used an existing isolated Temp
installation; no Conda environment was modified. Actual SPORC/A100 acceptance
is required inside the new single job before its full scientific fit.

## 2026-09-15: CMS learned-dense synthetic four-vector parity repair

Test-only reuse of the positive-energy unit-mass construction already in
`models/scouting_particle_transformer.py::validate_scouting_weaver_fp32_parity`,
inspected at `f33b8bff611786129ff941aa09aeca2d9856af37`, for
`tests/test_cms_salience_learned.py::make_cache`. No production kernel or
scientific contract changed; no external Weaver source was copied into the
repository. The mock-independent fixture invariant and real-Weaver privileged
path checks are new tests.

## 2026-09-15: native-CMS learned dense ladder, 500k/250k/250k

New isolated family: `src/hlt_classification/cms_salience_learned/`.
In-repository donor baseline inspected at
`d878beec498bba1f089f4ea1041ed17794e8bf68`; no external donor checkout,
`Fresh_check`, sibling worktree, old checkpoint, or old execution artifact is
imported at runtime.

- Native inputs/labels/splits: `scouting/{schema,labels,identity,splits,
  selective_assignment,inputs,particles,highcov_matcher,highcov_data,matching}.py`.
  The last four already had user-owned uncommitted category-count additions;
  they were not changed or staged by this implementation.
- Exact matcher: `scouting/hcwdl_fullcard_salience_{matcher,contracts}.py`.
- View/coupling kernels: `scouting/hcwdl_{homotopy,upper_coupling,
  unified_balanced}.py` and `repair.py`. These are reused, not replaced with
  Delphes preprocessing. New preparation fits bounded train-only scales.
- Architecture/loss: `models/{scouting_particle_transformer,
  hcwdl_adjacent_fusion_transformer,hcwdl_offline_hlt_fusion_transformer}.py`
  and `scouting/hcwdl_offline_hlt_withdrawal.py`, unchanged. A native wrapper
  supplies the separate-view call interface and exact extraction.
- Training-loop design donor: `jetclass2_delphes/salience_learned_training.py`
  and `salience_learned_graph.py`. The new kernel has native CMS metrics,
  fifteen-class probabilities and its own recipe/artifact family; it does not
  import their eleven-class trainer, graph, models, data or artifacts.
- Shared operational helpers: `data/cache_contracts.py`,
  `scouting/hcwdl_{authorization,exact_dag_submission,recovery}.py`, and only
  site/allocation authentication from `jetclass2_delphes/execution.py`.

## 2026-09-13: auxiliary debug profile continuation

No external or sibling-worktree donor code was copied. Reused unchanged native
`jetclass2_delphes/execution.py` site/allocation helpers from
`bee8bc48a1e634d0858f689b6b733ee3b2232265` to authenticate debug and tier3
separately. The matching campaign's `profile_attempt.py` was inspected as an
operational reference, not imported as auxiliary acceptance or migrated source.
The new `offline_aux/preparation_import.py` binds original auxiliary CPU task
receipts and compares the registered preparation-kernel Git blobs between
producer/execution commits. Scientific targets/model/training kernels and the
existing source/Slurm journal primitives are reused, not rewritten.

## 2026-09-12: isolated Delphes offline auxiliary-supervision study

New code lives exclusively under `jetclass2_delphes/offline_aux/`, with its own
CLI, worker and tests. No external source or sibling-worktree code was copied.
Reusable in-repository donors inspected and unchanged relative to
`82032e35177f83436741d7fa1b9d38fbc4b3efc7`:

- `jetclass2_delphes/contracts.py`, `schema.py`, `selection.py`, `inventory.py`,
  `split_registry.py`, `reader.py`: identities, selection, file verification,
  exact Hamilton allocation and native particle decoding. The auxiliary reader
  adds explicit TRAIN/SELECT/REPORT capabilities; no generic validation bypass.
- `jetclass2_delphes/inputs.py`, `model.py`: unchanged native HLT transform and
  installed Weaver factory/configuration. The new wrapper taps the final class
  vector without copying or replacing Weaver's forward implementation.
- `jetclass2_delphes/cache.py`: RamBlock, the pure array batching operation and
  process-thread limiter only. New cache construction does not call constituent
  matching, view construction or require a matching-foundation artifact.
- `jetclass2_delphes/reporting.py`, `execution.py`: native eleven-class metrics
  and exact SPORC site/allocation checks. New weighted cluster-bootstrap code is
  tested against the existing metric evaluator on explicitly repeated rows.
- `data/cache_contracts.py`, `scouting/hcwdl_authorization.py`: canonical hashes,
  deterministic compact NPZ, immutable publication and exact clean/pushed source.
- `sbatch/jetclass2_delphes_common.sh` (and its unchanged `sbatch/common.sh`
  helper): isolated SPORC activation and thread/import-path hygiene.

New submission/recovery code follows the repository's exact-ID intent/receipt
pattern but owns a separate namespace and never calls cancellation/priority APIs.
New artifact families are all `JETCLASS2_DELPHES_OFFLINE_AUX_*/v1`; no existing
dataset, split, matching, scientific recipe or runtime-profile family was
silently redefined. The donor paths above have no dirty semantic-byte delta
against the cited HEAD. Unrelated dirty Scouting files were not modified.

Actual production artifacts require their own later clean pushed source pin;
this donor baseline is not an assertion that the working tree is clean.

## 2026-09-12: site-bound Delphes SPORC readiness

New execution/readiness modules and thin CLI/shell helper extend the preceding
uncommitted Delphes migration. No external donor was copied. Hash/immutable
publication and exact-DAG receipt helpers still come from repository baseline
`fd1ed1d01d54bf2ad4d42ffa6311432263a14770`, specifically
`data/cache_contracts.py`, `scouting/hcwdl_authorization.py`,
`scouting/hcwdl_exact_dag_submission.py` and `scouting/hcwdl_recovery.py`.
The new readiness submitter reuses Delphes `submission.py`'s guarded exact
submission, with an independent identity, roots and authorization phrase.
`sbatch/jetclass2_delphes_common.sh` explicitly sets the selected isolated
environment before calling the unchanged `sbatch/common.sh` activation helper.

RUNTIME_PROFILE and INSTALLED_ENVIRONMENT are v2; CAMPAIGN_SPEC is v3.
EXECUTION_SITE, READINESS_SPEC and RESOURCE_MEASUREMENTS are new v1 contracts.
The cache budget calculation is shared with the existing Delphes preparation
bound; science, membership, field policy and model/loss/schedule are unchanged.
No pre-existing dirty Scouting donor file was edited or staged by this task.

## 2026-09-11: reusable Delphes training-size split registry

New `jetclass2_delphes/split_registry.py` and
`scripts/create_jetclass2_delphes_split_registry.py` implement metadata-only,
class-stratified nested selection and compact per-file entry masks. No donor
split algorithm was copied. Runtime hash/immutable-JSON helpers are reused from
`data/cache_contracts.py` at repository donor baseline
`fd1ed1d01d54bf2ad4d42ffa6311432263a14770`; inventory/selection/schema/reader and
foundation/production integration extend the preceding uncommitted local
Delphes migration, not an invented published donor commit. Registry artifacts
record the actual implementation-file hashes separately from Git HEAD.
No raw ROOT files, legacy campaign source, checkpoints or remote jobs were
modified. New families are SPLIT_DESIGN/ROLE_MEMBERSHIP/SPLIT_REGISTRY/
SPLIT_PROFILE v1 and subset FOUNDATION_SPEC/CAMPAIGN_PLAN/CAMPAIGN_SPEC v2.

## 2026-09-11: isolated JetClass2 Delphes local migration foundation

New implementation lives under `src/hlt_classification/jetclass2_delphes/`.
Repository donor baseline: `fd1ed1d01d54bf2ad4d42ffa6311432263a14770`.
No `Fresh_check` imports, sibling-worktree imports, FullSim weights, old dataset
artifacts, or previous runtime evidence are reused.

| Donor file | Use and explicit adaptation |
| --- | --- |
| `data/cache_contracts.py` | Runtime reuse of hashing, deterministic NPZ and atomic publication |
| `data/part_inputs.py` | Adapted analytic 17-feature mathematics; new p4 reader, class map, capacity and contracts |
| `models/particle_transformer.py` | Runtime Weaver factory/config reuse; new 17/11 wrapper, sequence trimming disabled |
| `scouting/hcwdl_fullcard_bottleneck_matcher.py`, `hcwdl_fullcard_bottleneck_contracts.py` | Runtime matrix-level exact solver and quantization; no old schema projection |
| `scouting/hcwdl_homotopy.py`, `repair.py` | Adapted exact endpoint/support and atomic applicability principles into new raw-field layout |
| `scouting/hcwdl_upper_coupling.py`, `hcwdl_unified_balanced.py` | Adapted mass-normalized insertion/removal and balanced circular switch mathematics; no residual substitutions under full cardinality |
| `scouting/hcwdl_tri100_spine4_graph.py` | Copied branch/rational-coordinate/schedule definitions into independently versioned fresh-reference graph |
| `scouting/hcwdl_mhpe_tri60_training.py` | Adapted CE+forward-KL/T², schedule, patience/selection and no-resume principles; not a runtime import of its 15-class trainer |
| `scouting/hcwdl_authorization.py` | Runtime exact-clean-pushed-checkout validation |
| `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Runtime canonical dry-ledger validation, dependency resolution and submission receipts; new wrapper adds a campaign-wide submitter claim, pre-sbatch durable intent and cross-recovery active-job checks |
| `sbatch/common.sh` | Existing absolute-path Conda/Tigris activation helper; new worker scripts supply explicit new data/spec paths |

The matrix solver's inspected SHA256 was
`9c92616446e77c27a00e1871dfdbdb2d46b2634665b80a79c68fe25365d5f190`.
Its module-level transitive data imports include pre-existing dirty work:
`highcov_data.py` SHA256
`0803a1a813d043ebb30581ecaa622dd7d60bdb65cdbd09ae59895f15badd90c5`,
`particles.py` SHA256
`73d5890a2a5f4b35e0f2fd7e0fe7cd232a7b5afc9e2ea4dd4814e232435e4f38`,
and `highcov_features.py` SHA256
`1190f99e917e186a16e66dea5a865f41616cbc6255595dd623e3194015ef376b`.
Those files were not edited or migrated in this task. The new adapter passes
numeric matrices directly and does not use their old raw-particle adapters.

The provisional label table references upstream
`jet-universe/jetclass2_generation`, commit
`3a7a1355f4230b5790669286466080d7fa3b6794`,
`delphes_analyzers/FatJetMatching.h`. This is a code reference, **not** a claim
that it is Luka's exact producer revision. Source inventory records actual
implementation bytes as well as Git HEAD because new local code is uncommitted.

This block also implements source-pinned production orchestration and synthetic
end-to-end tests. It is not a claim of genuine Weaver/Tigris acceptance.

The new repository is standalone. Donor code may be migrated only through an
explicit entry here and may never become a runtime import from `Fresh_check`.

## Inspected donor snapshot

```text
repository:
C:\Users\22rya\ComputerScience\CERN\Fresh_check

inspected commit:
bfcda5bd6fd48037ab198bcca1eadf6f4d87e131

worktree:
dirty at Block-1 inspection
```

Because the donor worktree was dirty, the commit alone is insufficient for
uncommitted donor files. Each migration block must additionally record the
exact donor path and source content hash used.

## Historical full-cardinality U100 compatibility adapter

The adjacent output-handoff `/v2` family consumes the immutable
non-persistent full-cardinality `SP4_COARSE_U100_from_U050` artifact produced
by Tigris job `98318`. The exact completed source family was implemented at
clean same-repository commit
`175cbcd6bf749f0014486a59a8c103802708abd1`; the earlier `/v2` snapshot at
`965bdad6aea2fdf81356275dfcdca698717a471f` lacked the corrected offline
lost-track endpoint identity semantics. Its
module names were later reused by a separately versioned persistent-HLT
experiment. The adapter therefore preserves the historical contract and view
construction locally instead of importing the current persistent-HLT module.
The source checkpoint and report remain read-only artifact parents.

The following `/v3` donor blobs supersede the three `/v2` contract, graph, and
campaign rows retained below as historical implementation context:

| Historical path and Git blob | Current destination | Retained semantics |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_contracts.py` -- `dd0109ab40bf2b1f89717afab952aa7d129f927d` | `hcwdl_adjacent_output_handoff_source.py` | Exact non-persistent full-cardinality `/v3` source contract identities. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_graph.py` -- `a2ad38187006e43e6aa28c19386fd56f5b4115f9` | `hcwdl_adjacent_output_handoff_source.py` | Exact graph and recipe including both unclassified-particle endpoint policies. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py` -- `751d02b0b40f24358e86e3acd3704cac2bb90961` | `hcwdl_adjacent_output_handoff_source.py` | Immutable `/v3` campaign-parent, endpoint-policy, and population checks. |

| Historical path and Git blob | Current destination | Retained semantics |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_contracts.py` — `6e218fecb50c54b2a2f718e958e456eb206d55c7` | `hcwdl_adjacent_output_handoff_source.py` | Exact non-persistent full-cardinality `/v2` source contract identities. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_graph.py` — `058f0a4729137f9e0973b43e68e20769e921e297` | `hcwdl_adjacent_output_handoff_source.py` | Campaign, recipe, pairing, and exact coarse-U100 graph authentication. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py` — `0602fc5bfbfff0e3bbbc5d74bfc8390f53cdb598` | `hcwdl_adjacent_output_handoff_source.py` | Immutable campaign-parent and all-mapped-population checks. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_runner.py` — `4002c92c62e4d7ed02ad370db0dfa7fcb3124dac` | `hcwdl_adjacent_output_handoff_runner.py` | Exact `replace_source_with_target_v1` RAM-only student-view reconstruction and historical training authority. |
| `src/hlt_classification/scouting/hcwdl_tri100_spine4_graph.py` — `3de1bbe6886ac98df76fea9db6d6adc21856bc7e` | reused current module (byte-identical at the compatibility boundary) | Exact `SP4_COARSE_U100_from_U050` node payload and seed alias. |

No runtime import from a detached historical worktree is used.

## HCWDL representation-KD source integration

The full-data three-track 60-pass campaign integrates the reviewed
representation-KD implementation from clean commit
`acecf9f74dab3d4ac675d8160cfb5decf83ba680`. The development worktree was used
only to locate and review the files; it is not a runtime import, artifact
parent, or campaign path. The campaign integration lock additionally checks
the exact committed Git blob IDs of the twelve byte-preserved scientific
modules before live creation.

| Donor path and SHA-256 | New path | Retained semantics and integration |
|---|---|---|
| `src/hlt_classification/models/scouting_particle_transformer.py` — `c6128610d4d249984a8d7af874737e1dd2e00337466ebdd91a119424dbf765cc` | same path | Merged one-forward token/jet/relation surface capture into the current unified model while preserving ordinary-logit parity. |
| `src/hlt_classification/models/hcwdl_representation.py` — `49d754e3cae1abfbb1c332c90f2db15245303ffbe5ccf2a649e647a25fcd5219` | same path | Exact representation heads, student wrapper, and head-free deployable extraction. |
| `src/hlt_classification/models/hcwdl_surfaces.py` — `90d8449e862c10e8fe8a73c305d338b7411a4e74c734b353c9be1ccdfa7027ae` | same path | Exact typed one-forward surface container and validation. |
| `src/hlt_classification/scouting/hcwdl_representation_graph.py` — `5205b158aabc2f31587c5554f4bccadcdf3a878e2b1fa12423481a77522ccb0e` | same path | RSET/RREL strategy identities and paired seed semantics. |
| `src/hlt_classification/scouting/hcwdl_representation_artifacts.py` — `45cea8226a6e752857b407031d330453f10e999130a1daa9d4aeb33a2a7a5fb7` | same path | Exact representation artifact-envelope and lineage helpers. |
| `src/hlt_classification/scouting/hcwdl_representation_contracts.py` — `808ca40ed06ecf2ac0df22b2a51c1fc0854c8be1b870aa9beab733654e9c37c1` | same path | Exact versioned representation contract validators. |
| `src/hlt_classification/scouting/hcwdl_representation_data.py` — `0fa2f085e5646d2410135f61968ab139048b212cf1dacb740467c865689d9bbd` | same path | Exact representation-aware input and batch containers. |
| `src/hlt_classification/scouting/hcwdl_representation_recipe.py` — `0cc302ce6aa862506db32112038913cb4b9bd58bd64294d50c7193153188c1ed` | same path | Corrected representation recipe v5 schedules, kernels, and calibration. |
| `src/hlt_classification/scouting/hcwdl_representation_kernels.py` — `c236e668f481d6724f50eca59b693b376000e42788376d483206ed1f7668a173` | same path | Spectral resources, finite kernels, weighted means, and reference oracles. |
| `src/hlt_classification/scouting/hcwdl_representation_losses.py` — `413bcaebf645dce0811d23c1d3093b2255512f34c6c1b5b2c44560d0edf9d2e3` | same path | Jet, set, relation, topology, scheduling, and support-aware losses. |
| `src/hlt_classification/scouting/hcwdl_representation_calibration.py` — `c9a94361e09e27eca999e3e93fcf691f5e4d026c8f644b44294d800f311ae9c1` | same path | Train-only deterministic calibration with state/RNG restoration. |
| `src/hlt_classification/scouting/hcwdl_representation_targets.py` — `97324a3b492ab21785d0fe6be3184afbf91cc7521c93b1881de877965de65b0a` | same path | Target schemas, identity hashes, validation, and support metadata. |
| `src/hlt_classification/scouting/hcwdl_representation_target_runtime.py` — `450a7843293fe69a73306decc086385c650487c36b63cfc17e0c045d4e5a8b2b` | same path | One-forward in-memory carrier target construction and runtime audits. |
| `src/hlt_classification/scouting/hcwdl_representation_training.py` — `16f344be23a7ce40ed7c4a45222a8ca769953ffcfbfc06d82a08c82135817f1b` | same path | Representation model initialization, identity joins, normalized components, diagnostics, and extraction. |
| `src/hlt_classification/scouting/hcwdl_homotopy_stream.py` — `de49beb816ab4f80e83d0159e3143744d3b935cca8428eea1935382a4affcd04` | same path | Merged fixed-work-per-source-chunk endpoint streaming into the current homotopy stream. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_training.py` — `bbf468f1e9bcb199dc5ebce790391f47ec20fbcca79eaf440f56d73d538ae29d` | same path | Retained as the numerical wiring reference for carrier-view target generation and training. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_targets.py` — `e2e10e74c6009f5cecf1fb19fa1ff90045bd76f505caeed053b6821e45ff11d9` | same path | Retained only as a durable-bank numerical oracle; TRI60 never calls its publisher. |

The following exact same-commit support modules were also migrated so the
representation package and its regression tests remain standalone. They do
not authorize TRI60 to use a historical campaign graph, rolling-resume
publisher, or durable representation-target publisher.

| Donor path and SHA-256 | New path | Campaign role |
|---|---|---|
| `src/hlt_classification/scouting/hcwdl_direct_offline_kd_graph.py` — `16fd891eb2358f17af4e1a82a4628722ba96aad15304e73a13a0f5248c6bf312` | same path | Donor graph validation/reference only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_contracts.py` — `593c5ac9215490abab388f315fd1d81b226ca9d0a2bfee5d07bce874219e613a` | same path | Donor contract validation/reference only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_graph.py` — `d0841a608d1dccf71fa5d474f2ba1e6f19a489f31578079687903bf65d8670c6` | same path | Donor graph/reference fixtures only. |
| `src/hlt_classification/scouting/hcwdl_homotopy_representation_recipe.py` — `84ad20239df5b1b7c7ffab60caccaab36c8ac83e6265189742335e1ce18c4357` | same path | Donor recipe/reference fixtures only. |
| `src/hlt_classification/scouting/hcwdl_numerical_acceptance.py` — `3aa8f02919bbfb964fb7225486c929185744c7e7d128c703290f4295f4687596` | same path | Numerical acceptance helpers reused by regression tests. |
| `src/hlt_classification/scouting/hcwdl_paired_bootstrap.py` — `79d8f5f3e206b3137163208e0780226513e6cc08bb5a6fc954daf12fe049fb1e` | same path | Paired reporting/bootstrap reference. |
| `src/hlt_classification/scouting/hcwdl_parent_loss.py` — `879c8fc49263006375226398659ad8739cb729222f320f7c20fbff8c429bb2bf` | same path | Parent-loss reference and validation. |
| `src/hlt_classification/scouting/hcwdl_representation_graph_registry.py` — `63881ffc2e3554b15a9aefab709b12b6d697fb7ad6354066ebc63d2ca0163049` | same path | Donor strategy registry/reference. |
| `src/hlt_classification/scouting/hcwdl_representation_reporting.py` — `ea6bf42fed13384bdcca13381eb280f0b696dd913ac215640a34950bdf725b6d` | same path | Representation diagnostics/reference reporting. |
| `src/hlt_classification/scouting/hcwdl_representation_resources.py` — `511e17de3a88e6d9c5670af117ce2095680109383be31a07307ad124ab2a611e` | same path | Resource-envelope reference. |
| `src/hlt_classification/scouting/hcwdl_representation_resume.py` — `9333574c5edd3b8913df9f8f6039340817f8ec6fc037c9082434773ae213d223` | same path | Numerical no-resume equivalence oracle only; TRI60 never invokes its publisher. |

The donor regression references were
`tests/test_hcwdl_representation_math.py` (`83a1592346ef0d94f8cff7d313d1298cf98ae78e6f425508e55504a7cbe8ef25`),
`tests/test_hcwdl_representation_model.py` (`7ef207e1d916265a89f83d57c1709f6a5a8ea4969d8fa458c9a87dd4fe2e5d6b`),
and `tests/test_hcwdl_representation_training.py`
(`749f1d76c87c720200f4a0c18e829e2e2143ea24553755df893ff8995bdb7485`).
The math and training assertions were retained; the model fixture was adapted
to the current direct model-surface API. No runtime import from the donor
worktree remains.

## Migrated code

Transfer Block 1 contained new scaffolding only. Transfer Block 2 extracted and
rewrote the following semantics without donor runtime imports:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/jetclass_data.py` — `30bbb947f83aeab9b190af640842071af075a558e1e12f7018f6ae96edf248ea` | `src/hlt_classification/data/{schema,identity,root_reader,splits}.py` | class order, longest-prefix mapping, 14 channels, 128 cap, transformations, bounded retries, chunk ordering, per-role sampling RNG, global shuffle | canonical recursive parent root; one root-relative namespace; label-bound identity plus label-free leakage key; strict branches/lengths/finiteness/PID/one-hot checks; stronger capacity/balance/overlap audit; versioned self-hashing immutable atomic manifest; streaming iterator. Covered by `tests/test_data_foundation.py`. |
| `scripts/build_jetclass_splits.py` — `87293fff44a413c5e6e833a2bfcddf422fe3e2017a077b6bbaade9ecba99b222` | `scripts/build_splits.py` | explicit split sizes and report | rewritten standalone CLI; single recursive root; production fail-closed; authenticated atomic output. |
| `scripts/preflight_relational_part_data.py` — `c15cb8c30100809f6b6804ccaea425b545e3598c41e877c0b7d0d7682c8ea780` | `scripts/preflight_data.py` | class capacity preflight | generic standalone CLI; full branch schema validation; diagnostic skips can never be production-eligible. |

The donor test reference was
`tests/test_jetclass_data.py` at
`80095b50141e34249b4561c6293f710e87559fb3a13362bf030e6808042d2c7d`.
The new suite broadens its assertions and uses actual synthetic ROOT fixtures.
No third-party source was copied, so this block adds no new source-license
obligation beyond the declared Python dependencies.

Transfer Block 3 migrated the registered HLT-v3 v1 scientific kernel:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `teacher_logit_reco/relation_expert_token_bridge/hlt_v3.py` — `3f8509486818469fc63990eccb9d09182df460ae593efef474f4a6a3b9311037` | `src/hlt_classification/data/hlt_v3.py` | profile parameters, degradation profiles, type multipliers, PID transitions, operation order, substream IDs, threshold/loss/merge/response algorithms, mass preservation, stable sorting, validity states, and diagnostics | removed RETB/cache imports; strengthened invalid-type and finite validation; assigned serialized metadata a repository-specific contract name; retained donor random-domain bytes for registered-v1 output parity. Golden array and diagnostic hashes plus focused mechanism tests are in `tests/test_hlt_v3.py`. |
| `teacher_logit_reco/relation_expert_token_bridge/replicas.py` — `fe3013ac4543d048287527bb326c6ea943251b9448170c4f2fbc1dcc6181ffa8` | `src/hlt_classification/data/replicas.py` | policies, donor role seeds, random multipliers, identity cycle, event seed derivation, and evaluation replica zero | removed RETB contract dependency; added canonical `model_val=3054` and `stack_train=3055` domains required by the new five-role split while preserving every donor-existing mapping; added a repository-specific self-authenticating manifest contract, strict input validation, and detached manifest payloads. |
| `jetclass_fixed_hlt.py` — `c222a9233b1a5284a57bbcf2e253c83d3d7679ff26f5e825c854e8cece3bb2e4` | `src/hlt_classification/data/hlt_v3.py` | phi wrapping, local density, v2 efficiency base terms, and v2 kinematic base terms only | excluded legacy v1/v2 builders, CLI, caches, and unrelated profiles; inlined only the inherited formulas needed by registered HLT-v3. |

The focused donor test reference is
`tests/test_relation_expert_token_bridge_step2.py`. Block 3 does not copy
third-party source and introduces no new dependency or license obligation.

Transfer Block 4 inspected both legacy cache surfaces and rewrote them rather
than copying either monolithic format:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/hlt_cache.py` — `5631ac115a4f352739492e0db06c5e9c3111fa52d522d273243504e0731823c4` | `src/hlt_classification/data/{cache_contracts,offline_cache,dataset}.py` | bounded cached raw views, stable identity order, reusable HLT-facing dataset access | split offline and degraded caches into distinct versioned contracts; deterministic NPZ bytes; per-array and per-file hashes; atomic immutable sidecars and manifests; bounded cross-shard reads; exact source/split/schema/generator lineage. |
| `teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py` — `57d0dde4e4aad5aa622fc62c2020954ae69a0108cf89178d7304dabd2f6345e4` | `src/hlt_classification/data/{cache_contracts,hlt_cache,dataset}.py` | deterministic degradation from authenticated offline parents, replica/profile binding, resumable shards, aggregate diagnostics | removed RETB campaign coupling and construction-index exposure; added source-snapshot and exact parent-range hashes; made processing-batch and physical-shard layouts semantically invariant; added standalone build/audit CLIs. |

Block-4 failure injection and byte-level resume/layout evidence is in
`tests/test_cache_pipeline.py`. No donor runtime import or third-party source
was copied.

Transfer Block 5 migrated the canonical model-input and adapter semantics:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `jetclass_fresh/part_inputs.py` — `6e237ba0eec33ef9a2b78cde579896b02c5b739ccd8126de0512774eef30166a` | `src/hlt_classification/data/part_inputs.py` | two point coordinates, exact ordered 17-feature transform, four-vectors, supplied-view axis reconstruction, padding, and all-empty safety | removed donor package and candidate-weight coupling; accepts the standalone 14-token `JetView`; adds strict dtype, finite, padding, identity, label, and explicit source-view validation plus a repository-specific contract. Covered by `tests/test_part_inputs.py`. |
| `jetclass_fresh/hlt_baseline.py` — `982c1696967b809fd534add1766f9b780fb858297f334036a7d9aa84f078a418` | `src/hlt_classification/models/particle_transformer.py` | lazy Weaver construction, exact standard-four canonical architecture, input delegation, and class-token no-weight-decay name | excluded donor dataset/training code; mixed precision is disabled by the authoritative FP32 runtime rather than passed as a noncanonical model constructor argument; adds logits/input-gradient/parameter-gradient/mask/state-dictionary attestation. Covered locally with an interface-faithful test double; authoritative real-Weaver execution remains pending. |

Both donor files were inspected at commit
`bfcda5bd6fd48037ab198bcca1eadf6f4d87e131` and were clean at their recorded
paths. No Weaver source was copied. `weaver-core` and PyTorch are optional
model dependencies and retain their upstream licenses.

Transfer Block 8 copied and contract-rebased the reviewed HOSD scientific
plan:

| Donor path and SHA-256 | New path | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `teacher_logit_reco/HLT_OFFLINE_STRUCTURE_DISTILLATION_IMPLEMENTATION_PLAN.md` — `0d4eaab71c13dd64a4af326d085f35259490b90eca084c60d6c6c3a24325a937` | `docs/plans/HLT_OFFLINE_STRUCTURE_DISTILLATION_IMPLEMENTATION_PLAN.md` | all scientific questions, target definitions, controls, split counts, access rules, stages, selectors, metrics, statistics, locks, and acceptance criteria | replaced old repository/runtime paths with `src/hlt_classification/hosd` and local reusable contracts; retained optional donor comparators; required local ports plus donor-parity tests for missing RPT/RETB semantics; reset donor-only implementation statuses. No executable donor code was copied by this document migration. |

## PRAD implementation provenance

PRAD was implemented from the user-supplied repository-root documents
`prad_implementation_agent_prompt.md` and
`privileged_relational_attention_proposal(1).md`. No executable donor file,
checkpoint, cache, metric result, or historical campaign artifact was migrated
for PRAD. The implementation extends the already mapped canonical Weaver
wrapper and standalone data/campaign contracts; consequently there is no new
donor commit or donor-file hash to register for this campaign.

## PMARD implementation provenance

PMARD was implemented independently from the active repository plan and the
read-only operational guide
`C:\Users\22rya\ComputerScience\FCV\SCOUTING_AK8_USAGE_GUIDE.md` (SHA-256
`c80b3f3459848c4f634f8f3a861fc83da0bfa530f71e314826123d8dc6774ac9`).
The adjacent `scouting_ak8_tools` directory has no Git metadata, so no
executable file from it was copied or imported at runtime; without an exact
donor commit it is an operational reference only.

The repository vendors the data-only CMSSW V00 preprocessing transcription
under `configs/scouting/`. Its authoritative upstream source is
`cms-data/RecoBTag-Combined` commit
`37a104b51c8458bcda97a95ba52d304a5041e922`; the documented transcription hash
is `2e346c5da8491046205f29cba21b0f6e434e273eb89dae068aa5dd40e2ca91f7`.
The independently authored `TOFF/v1` JSON records the semantics and the
documented reference-YAML hash
`32da128ad0e82c6933a36cf409ae766523a116ae66df8d43153c671bc89f7c1d` plus
Weaver reference commit `6a084ec341f5af2e06f6d884e6a4ee4dab8bb4e4`. No Weaver source is copied.
The local JSON byte hash is also recorded by the config validator; it differs
from the guide's reference-file byte hash only by terminal newline encoding.
Scientific validation compares the parsed, ordered preprocessing semantics and
retains both hashes rather than falsely claiming byte identity.

### Fitted-strict matcher port

The matching study directory has no Git metadata, so this port is bound to
exact source hashes rather than inventing a donor commit:

| Donor path and SHA-256 | New paths | Retained semantics | Intentional changes and evidence |
|---|---|---|---|
| `HLTvsOfflineComparisons/matching/run_selective_matcher_campaign.py` — `68e2bbf1098506c37dbd5af7a1265db89bfbece19b7f7a6a00ab99605e2544d0` | `src/hlt_classification/scouting/fitted_strict.py` | fitted-strict edge features/signs, empirical LLR lookup, broad and strict gates, rectangular Hungarian private dummies, assignment diagnostics, confidence calibration, exact operating point | refactored inference only into a stable typed API; no implicit fitting; native offline indices are restored after lost-track exclusion; fail-closed artifact validation; numerical golden parity in `tests/test_fitted_strict_matcher.py`. |
| `HLTvsOfflineComparisons/matching/FITTED_STRICT_MATCHER_IMPLEMENTATION_GUIDE.md` — `d3b2adc32df67b0ee971a9383fadbae00540b4022be64a921b7d2d90c361733e` | `docs/contracts/FITTED_STRICT_MATCHER.md` | algorithm, population, selective-mask, and PMARD preservation contract | condensed into a repository-local versioned contract and explicitly blocks the incompatible legacy complete endpoint. |
| canonical `fitted_edge_model.json` — donor byte SHA-256 `2cfd51b209435164263247252a35cf83708fa71fc2580301c35bb5d905d41142`; `confidence_models.json` — `dc1ce2f3f889bef3e3c6cdc46da70bc14e79dff518d6d4b1d759fac51f093735`; `independent_validation.json` — `af1d2b009e7c9186b454acd461e4cf40335261ae4b491f1c82dd1957e941687e` | `src/hlt_classification/scouting/resources/fitted_strict_v1/` | exact parsed model/audit values from the intact 7,500-jet bundle | JSON semantic hashes are authenticated so Windows/Linux checkout line endings cannot change identity; original donor byte hashes remain in the vendored provenance file. No checkpoint, dataset, or race-corrupted result directory was copied. |

The independent donor validator
`validate_selective_matcher_results.py` was inspected at SHA-256
`f83b7e42bdd9d6b28292727fa31c341e22a7f1a4668b0eb0df52cad84a628ae3`.
No runtime import reaches the external FCV tree, and no new third-party source
or license obligation was introduced.

The subsequent selective campaign plumbing—`selective_assignment.py`, the two
builder CLIs, selective 21-field repair, assignment authorization, pilot DAG,
and oracle evaluation—is original repository-local work with no donor file or
commit. It composes the authenticated matcher above with existing Scouting
contracts; tests cover sparse joins, exact unmatched preservation, endpoint
replacement, and campaign registration.

## HCWDL high-coverage matcher port

The HCWDL completion-shell matcher was ported from the independent, clean
repository `high_coverage_matcher_research` at exact commit
`64be1a82f11f42949fdffa639a869ccea2528bfa`. The portable implementation
guide `docs/OUTSIDE_AGENT_IMPLEMENTATION_HANDOFF.md` was read in full. The new
runtime has no import, path lookup, or other dependency on that repository.

| Donor path and SHA-256 | New path | Retained semantics | Repository integration |
|---|---|---|---|
| `src/highcov/assignment.py` — `44dee1847dbc5287ee97c738468abf6be7a551831180130ace5ae8563166bba7` | `src/hlt_classification/scouting/highcov_assignment.py` | cardinality-first lexicographic assignment, private dustbins, consensus assignment, and 18 diagnostics | package-relative imports; immutable assignment shards and parity tests |
| `src/highcov/calibration.py` — `6dcbfefa0222ddd9a06a5cc1fdb7bfc5f137a00cdcef56c1f1c6878d034aa832` | `src/hlt_classification/scouting/highcov_calibration.py` | frozen isotonic confidence calibration | strict packaged-resource validation and uint16 persistence |
| `src/highcov/data.py` — `7e4b2ce99abab15b7abc06a1aae223ba58b5d81c5e16ad59c58f998cf18f8676` | `src/hlt_classification/scouting/highcov_data.py` | validated particle container, kinematics, categories, charge, measurements, and native indices | Scouting particle adapter; lost-track exclusion before matching |
| `src/highcov/features.py` — `db2a1206686697003bb6b8a2803b340424a0f56f667c08ac29286c559cffe597` | `src/hlt_classification/scouting/highcov_features.py` | frozen candidate gate, edge matrices, ranks, and feature order | donor-parity fixture and fail-closed schema validation |
| `src/highcov/scorers.py` — `77debf7a892c6e16149027563f91fb68433f58638ae965b3ed2a20e1c829d97c` | `src/hlt_classification/scouting/highcov_scorers.py` | empirical and independent consensus scores | role/fold scorer selection is enforced by `highcov_matcher.py` |
| `src/highcov/final_matcher.py` — `78a758ef5d3333d67d1e2aa193af2eed06edcb7d04b483f2f788e4722ddbc897` | `src/hlt_classification/scouting/highcov_matcher.py` | selected global matcher and post-assignment confidence | train cross-fitting, validation/audit model policy, native-index output |
| `src/highcov/hashing.py` — `fb602d4048bb8f8497dc899a219e55679272046b424dccd259a2e40c4bdec00a` | `src/hlt_classification/scouting/highcov_hashing.py` | canonical semantic hashing | composed with repository byte/content hashes and atomic publication |
| `configs/selected_matcher.json` — `eab6c945c058a5447063066eca79c931434d06d63e3729d1dfeeeab354100a87` | `src/hlt_classification/scouting/resources/highcov_v1/selected_matcher.json` | exact selected algorithm/configuration | packaged runtime resource; parsed semantic hash is also checked |
| `artifacts/models/empirical_models.json` — `6a9f8d594b0bfadae03a80187917fac2c8261cb0345965daa2f33af4e6836389` | `src/hlt_classification/scouting/resources/highcov_v1/empirical_models.json` | full and four holdout empirical scorers | exact fold selection and train-leakage rejection |
| `artifacts/models/final_confidence_calibration.json` — `40a9c4499244916e58a1f8c74a52d05aac0418a21220b27d03afcd97a8454c31` | `src/hlt_classification/scouting/resources/highcov_v1/final_confidence_calibration.json` | final confidence calibrator | exact semantic/resource validation and quantization tests |

The donor's research-only completion/contextual/inference/synthetic surfaces,
generated results, and datasets were deliberately excluded. The 21-field
Shell Exact/Soft/HC Exact integration is repository-local work in
`scouting/repair.py`; no donor `repair.py` runtime was copied. The port adds no
new third-party dependency or license obligation.

## Strategy-B adjacent learned-fusion composition

The `HCWDL_ADJACENT_LEARNED_FUSION_HANDOFF_*/v1` implementation is original
repository-local work; no external or legacy source file was copied. It
composes the already versioned adjacent-output source adapter and
full-cardinality cache construction at commit
`7acc62313c5cd47dc0902d025a31716448e7daf1` with the repository-local anchored
fusion/exact-withdrawal primitives. The latter primitives were developed in
this repository for the immediately preceding offline/HLT fusion campaign and
are imported as reusable package APIs, not migrated donor files. Strategy B
changes their content semantics from offline/HLT to adjacent richer/lower
views, freezes unused cross-pair parameters, adds the exact rational morph
schedule and physical primary extraction, and tests these changes in
`tests/test_hcwdl_adjacent_learned_handoff.py`. There is no new third-party
code, dependency, or attribution obligation.

## JetClass2 salience Strategy-B composition

The `JETCLASS2_DELPHES_SALIENCE_LEARNED_HANDOFF_*/v1` implementation is
repository-local composition rather than an external migration. It reuses the
versioned Strategy-B withdrawal loss from
`src/hlt_classification/scouting/hcwdl_offline_hlt_withdrawal.py` and the
JetClass2 salience foundation/view APIs already recorded by their own plans and
contracts. The new 17-input/11-output asymmetric fusion model, three-spine
54-fit graph, U000-to-D000 morph control, RAM cache adapters, staged SPORC DAG,
and tests were authored in this repository. No historical checkpoint, target
bank, dataset, generated campaign result, or external source file was copied,
and runtime imports do not reach an old worktree or repository.

The 2026-09-14 loss-interface repair adds read-only `hlt_states` / `hlt_mask`
aliases to the JetClass2 `FusionOutput` for compatibility with that shared
withdrawal objective. Both names refer to the original lower/primary tensors,
not necessarily native-HLT content on intermediate U/D coordinates. The
reference API is the unchanged repository-local
`scouting/hcwdl_offline_hlt_withdrawal.py` at failed-execution commit
`15094633f9aa3a0e3f9e418704ac3c0a46dac11d`. No donor file or loss implementation
was copied; the coefficients, masks, directed gradients and exact-zero route
are unchanged. Tensor-level regression tests now exercise the interface even
when installed Weaver is unavailable.

## Approved transfer surfaces

| Transfer block | Donor surface | Intended retained meaning | Migration policy |
|---|---|---|---|
| 2 | `jetclass_fresh/jetclass_data.py` | schema, identity, split, chunked ROOT logic | extract and test; do not copy package imports |
| 2 | `scripts/build_jetclass_splits.py` | CLI behavior | rewrite thin local CLI |
| 2 | `scripts/preflight_relational_part_data.py` | data-capacity preflight | rewrite generic local CLI |
| 3 | `teacher_logit_reco/relation_expert_token_bridge/hlt_v3.py` | HLT-v3 scientific kernel | migrate with semantic parity |
| 3 | `teacher_logit_reco/relation_expert_token_bridge/replicas.py` | deterministic replica seeding | migrate with parity |
| 3 | `jetclass_fixed_hlt.py` | v2 base-term helpers used by HLT-v3 | extract only required helpers |
| 4 | `jetclass_fresh/hlt_cache.py` | bounded cached view and dataset semantics | inspect and rewrite as authenticated shards |
| 4 | `teacher_logit_reco/relation_expert_token_bridge/hlt_cache.py` | degraded-cache lineage and resume | inspect and decouple from RETB |
| 5 | `jetclass_fresh/part_inputs.py` | canonical ParT input transforms | migrate with exact tests |
| 5 | `jetclass_fresh/hlt_baseline.py` | Weaver wrapper and baseline configuration | extract model surface; rewrite trainer |

For each actual migration add:

```text
donor commit
donor path
donor file SHA-256
new path
retained semantics
intentional changes
parity tests and results
third-party license/attribution impact
```

## Deliberately excluded

Do not migrate checkpoints, caches, logs, run registries, historical campaign
IDs, unrelated campaign packages, old handoffs, or broad script collections.
