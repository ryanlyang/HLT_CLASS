# Legacy Donor-Source Map

## 2026-10-07: CORR_MID deferred tier3 science

Internal donor commit `062713ead3e00721d9f3cb7073e88aef83ebd325`.
No external code or Fresh_check imports. Old gate worktrees/receipts stay immutable.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `src/hlt_classification/literature_ladder_followup.py` | Reuse read/hash, sanitized subprocess, lock and original-pin inspection bridge; new correlated controller binds v9, exact CORR_MID manifest, 11-job tier3 policy and a separate executor |
| `src/hlt_classification/cms_proxy_ladder/{correlated,production,submission,gate}.py` | Retain measured-science validation, kernel, command builder and durable claims. New `correlated_tier3.py` explicitly authenticates source transfer and v6 campaign; sole old-source edit is the three-line validation dispatch |
| `src/hlt_classification/jetclass2_delphes/execution.py` | Import existing same-A100 debug-to-tier3 policy/site definitions unchanged; actual worker allocation/GPU/software checks retained |
| `scripts/jetclass2_literature_ladder_followup.py`, `scripts/start_jetclass2_literature_ladder_followup.sh` | New thin correlated tier3 CLI and dry-first detached launcher |
| `tests/test_literature_ladder_followup.py`, `tests/test_correlated_tracking_ladder.py` | Fake accounting/claim/inspection failure tests, synthetic real gate/foundation integration reused for tier3 |

The [new contract](contracts/JETCLASS2_CORRELATED_TIER3_FOLLOWUP.md) records
versioned execution-only artifacts; science and dataset semantics are unchanged.

## 2026-10-07: frozen CORR_MID production and SPORC direct/coarse

Internal donor commit `0667355d8c4d72e48e32c03f2d1540595fe654b7`.
No external code copied or `Fresh_check` runtime import. Historical pilot kernel,
recipe, frontend and artifacts are unchanged; new versioned adapters select MID.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_context_production/{campaign,contracts,population,storage,submission,engine,output,worker,codec}.py` | New `correlated_tracking_production/`: 2.25M membership, bounded process output, reservations, exact array identity, full DAG and missing-only recovery; new namespace and saved CORR_MID replay |
| `correlated_tracking/{campaign,contracts,kernel,worker}.py` | Historical pilot evidence and unchanged physical generator; new evidence module authenticates known job 228972, source and all output hashes |
| `literature_proxy_v2/inputs.py`, `literature_proxy/population.py`, `cms2jc2_production/population.py` | Original completed-parent evidence, exact raw particle keys and existing registry/split membership; no CMS-fit generation |
| `literature_context/transform.py`; `literature_proxy_production/codec.py` | Import unchanged common asinh/log1p frontend and lossless physical codec, never apply context degradation |
| `cms_proxy_ladder/{literature,context,views}.py` | New `correlated.py` original-storage SPORC adapter and `correlated_views.py` verified identity matching/noise-amplitude bridge; six fits and three reducers |
| `cms_proxy_ladder/{release,data,cache,gate,production,submission}.py` | Explicit new-schema dispatch while preserving old campaign semantics; shared CE/KD/cache/report kernels retained |
| `scripts/jetclass2_literature_context_dataset.py`, `scripts/queue_jetclass2_literature_context_dataset.sh`, `sbatch/run_jetclass2_literature_context_dataset.sh` | Thin production CLI/environment/queue wrappers under `correlated_tracking_dataset` names |
| `scripts/jetclass2_context_ladder.py`, `scripts/queue_jetclass2_literature_proxy_ladder.sh` | Thin CORR_MID SPORC CLI and separately reviewed gate/science helper |
| `tests/test_literature_context_production.py`, `tests/test_context_ladder.py`, `tests/test_literature_proxy_v2_campaign.py`, `tests/test_jetclass2_delphes.py`, `tests/test_jetclass2_delphes_training.py` | Synthetic ROOT/parent/production/cache, exact-plan, teacher-bank and CPU CE/KD integration tests |

Package paths are under `src/hlt_classification/`. New production contracts use
`JC2_CORRELATED_TRACKING_PRODUCTION_*/v1`; shared ladder versions are recorded in
the [new contract](contracts/JETCLASS2_CORRELATED_TRACKING_LADDER.md). The active
plan supersedes the proposed CE-only strength screen through explicit user MID
selection, not through a claim that the desired classifier gap has been measured.

## 2026-10-07: shared-versus-independent tracking pilot

Internal donor commit `f889f359cdddca1beb3d6e817332d2b42a42ddf3`.
New `correlated_tracking/` package; no old generator, dataset or classifier
edited. No external code copied and no `Fresh_check` runtime import.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_proxy/campaign.py`, `contracts.py` | Clean/pushed source, dry/live authorization, durable ambiguous-submit lock, immutable artifacts; new namespace and complete Python source pin |
| `literature_proxy_v2/inputs.py` | Import unchanged completed-v1 pilot authentication and original OFFLINE reader; old NOMINAL endpoint is ignored, no v2 rates used |
| `literature_context/campaign.py`, `worker.py` | Per-file spawn pilot, physical banks, serial/process replay, receipt-last publication patterns |
| `literature_proxy/diagnostics.py`, `worker.py`; `data/cache_contracts.py` | Descriptive histograms/moments/CSV/PDF, physical arrays/RSS, deterministic NPZ serialization |
| `cms2jc2_response/bridge.py`; `jetclass2_delphes/contracts.py` | Physical schema/applicability validation and safe relative output paths |
| `scripts/jetclass2_literature_context.py`, `scripts/queue_jetclass2_literature_context.sh`, `sbatch/run_jetclass2_literature_context.sh` | Thin CLI and dry-first Tigris environment wrappers |
| `tests/test_literature_proxy.py`, `tests/test_literature_proxy_v2_campaign.py`, `tests/test_literature_context_campaign.py` | Particle/synthetic completed-parent fixtures and submission/replay failure checks |

Package paths in the table are under `src/hlt_classification/`. Reference-noise
kernel, conditional residual and analytic pair-covariance diagnostics are new
synthetic equations specified in the versioned plan, not a migrated CMS fit.

## 2026-10-06: CONTEXT_V1 relocated Oscar 100k/50k ladder

Internal donor commit `4f473c3e97ee3270cec520914477dfe9a38728ab`.
No external source or `Fresh_check` runtime imports. SHA-256 values below are
local donor bytes at inspection (before line-ending conversion in another checkout).

| Donor file | SHA-256 | New surface / retained meaning |
| --- | --- | --- |
| `src/hlt_classification/literature_proxy_consumer.py` | `1bd2152aff3febb94c04bf5d0473a0f8da11af8551bb0885391d7c68a1ac692d` | `literature_context_consumer.py`: external-root relocation, identity joins and sealed ordinary reader; new context manifest/recipe/input contract |
| `src/hlt_classification/cms_proxy_ladder/literature.py` | `316734ea5dd44d3fb738c00e81c3cfb88aeaa41c6456177bb7124ebc1a9d7fd0` | `cms_proxy_ladder/context.py`: source-pinned gate, exact dry/live DAG, fresh fits; Oscar 100k/50k and CE/KD preflight |
| `scripts/jetclass2_literature_proxy_ladder.py` | `464d417be8c482fe717c25b972a9745e2b12225c5e2f1b6fcb8e9d3e9230c579` | `scripts/jetclass2_context_ladder.py`: thin CLI and saved validation results |
| `scripts/queue_jetclass2_literature_proxy_ladder.sh` | `1fac627bcdb773cc8cd745c20a7745e59ae3dd9dafe085fa3ad1db1688014c92` | `scripts/queue_jetclass2_context_ladder.sh`: two explicit reviewed phases; Oscar copied roots and clean pushed source |
| `tests/test_literature_proxy_consumer.py` | `f106089c15dd1e34edff8e9ae1f41393b0bdd790ad6391d3ee37be48a30408cc` | Context relocation, corrupt bytes, swapped IDs, no native-HLT/test reads |
| `tests/test_literature_proxy_ladder.py` | `d6a717cd3c2afbb0ed0f65b4bf75c0c77a409b530ba62b9b04dd786d14dfaaeb` | Context fresh foundation/cache/DAG and durable submission tests |
| `src/hlt_classification/literature_context/transform.py` | `5917d13717776200681107adf8415d52828ff20e0d8320c95b8cb78aea26cc59` | Imported **unchanged** through `context_inputs.py`; no generator/inverse retuning |

Shared ladder release/data/cache/gate/production/submission/views dispatch through
explicit new versions, preserving existing versions. Matching, model, loss,
schedule and teacher publication code are reused without new scientific equations.
No third-party license change. Tests and remote-admission status are recorded in
`docs/HANDOFF.md`; local synthetic acceptance is not a real Oscar GPU run.

## 2026-10-06: frozen CONTEXT_V1 full dataset production

Internal donor checkout **`6cb5f32ab6e8d179f846f0cfeb4511ef1a0b6ab4`**.
Separate `literature_context_production/` package and wrappers; no old response
kernel, dataset, input transform or classifier modified. No external donor code
or `Fresh_check` runtime imports. New `JC2_CONTEXT_PRODUCTION_*/v1` contracts
retain the old pilot and NOISE_V3 namespaces without relabelling artifacts.

| Donor paths | Reuse/adaptation |
| --- | --- |
| `literature_proxy_production/campaign.py`, `contracts.py`, `storage.py` | Frozen evidence, source, quota, immutable attempts and reservations; new namespace and exact approved context pilot binding |
| `literature_proxy_production/population.py` | Same metadata population, file-local shards, offline ROOT reader and original identity/RNG keys |
| `literature_proxy_production/engine.py`, `worker.py` | Bounded ordered processes, actual production-reader/writer preflight; explicitly compose unchanged low-noise and context kernels and check inverse integrity |
| `literature_proxy_production/output.py`, `submission.py` | Role manifests, exact array IDs, durable submit journals, full DAG and same-source missing-shard recovery; require context recipe/input encoding |
| `literature_proxy_production/codec.py` | Imported unchanged lossless physical-bank codec; no old scientific artifact identity reused |
| `literature_context/kernel.py`, `transform.py`, `campaign.py`, `worker.py` | Import frozen equations, input encoding, approved pilot authentication and saved CONTEXT endpoints without edits |
| `literature_proxy_v3/inputs.py`, `contracts.py`; v1/v2 parent modules | Completed pilot ancestry, original full-precision COUNT38 rates and physical evidence |
| `cms2jc2_production/population.py`, `contracts.py`; `literature_proxy/population.py` | Existing TRAIN_1M/250k-validation/1M-test metadata selection and raw offline conversion, not CMS-fitted response generation |
| `cms2jc2_response/bridge.py`, `readers.py`, `measurement.py`; `jetclass2_delphes/inventory.py`, `split_registry.py`; `data/cache_contracts.py` | Physical validation, authenticated source reads, measured resources, tree/entry identities and artifact hashing |

Package paths above are under `src/hlt_classification/`. Thin wrappers adapt
`scripts/jetclass2_literature_proxy_dataset.py`,
`scripts/queue_jetclass2_literature_proxy_dataset.sh`, and
`sbatch/run_jetclass2_literature_proxy_dataset.sh` into the corresponding
`literature_context_dataset` files. `tests/test_literature_context_production.py`
adapts `tests/test_literature_proxy_production.py`, reuses the original completed
v1/v2 synthetic ROOT pilot fixtures, and adds context-specific approval,
composition, representation, inverse and namespace checks.

## 2026-10-06: lower noise and context coupled tracking pilot

Internal donor checkout **`0a1013e9fa0571333242871b7cc34b2dfe0ba464`**.
New `literature_context/`, CLI/queue helper and CPU worker are additive.
No third-party source or `Fresh_check` import. Old NOISE_V3, generator, input
transforms and datasets remain unchanged; no fitted CMS response is reused.

| Donor paths under `src/hlt_classification/` | Use or adaptation |
| --- | --- |
| `literature_proxy_v3/kernel.py` | Adapt independent smearing implementation with tracking amplitude 1 instead of 4 and kinematic amplitude 1 instead of 2; original function remains the noisy control |
| `literature_proxy_v2/kernel.py`; `literature_proxy/kernel.py` | Exact frozen topology, PID/drops/merges, full-precision calibration validation and original keyed random streams |
| `literature_proxy_v3/inputs.py`; v1/v2 ancestor input and campaign modules | Authenticate completed training-only parent, physical NPZ endpoints and separate diagnostic ancestry, without reading ROOT |
| `literature_proxy_v3/campaign.py`, `contracts.py`, `worker.py`, `diagnostics.py` | Adapt single-job dry/live safeguards, source closure, atomic artifact patterns and per-file spawn execution; reuse count reduction and descriptive statistics |
| `literature_proxy/diagnostics.py`, `worker.py`, `contracts.py`; `data/cache_contracts.py` | Metrics, PDF/CSV, array serialization, sampled process RSS, file/content hashes and immutable publication |
| `cms_proxy_ladder/inputs.py`, `contracts.py`; `jetclass2_delphes/inputs.py` | Reuse unchanged 17-input physical frontend for saturation audit; optional new four-channel tracking encoding is separately versioned and never installed in an old campaign |
| `cms2jc2_response/bridge.py`; `jetclass2_delphes/contracts.py` | Physical validity, p4/wrapped angles and safe relative file paths |

Adapted thin wrappers from `scripts/jetclass2_literature_proxy_noise.py`,
`scripts/queue_jetclass2_literature_proxy_noise.sh` and
`sbatch/run_jetclass2_literature_proxy_noise.sh`. Tests reuse original synthetic
particle fixtures and completed v1/v2 pilot fixtures. Context equations and
inverse are new synthetic benchmark choices, not fitted/literature-derived
detector parameters. New `JC2_LITERATURE_CONTEXT_*/v1` artifacts preserve all
original donor schemas and calibration parents.

## 2026-10-05: restore original-memory K2 continuation without another native gate

Scientific donor: **`c891da0d45dd3251dea9ea72df975bb96bae3570`**, unchanged.
Accepted execution donor: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
Implementation base and unused retirement target:
**`c640942f8a4a91086016fa4bd151ae5d3bc9c31a`**. No third-party code copied.

| Donor paths | Use/adaptation |
| --- | --- |
| `k2_segmented/memory_migration.py` (c640942f) | Existing exact checkpoint/prefix authentication and independent byte-copy machinery; narrow v3 dispatch to original-memory restoration |
| `k2_segmented/campaign.py`, `runtime.py` (c640942f) | V3 omits native gate/launcher and completed D025 segments, restores v1 resources, retains exact-ID retirement/submission and hardware/software checks |
| `k2_segmented/training.py` (71c1bde2) | Unchanged scientific full-state kernel; byte equality required against the accepted donor |
| `scripts/jetclass2_k2_segmented.py`, `queue_jetclass2_k2_128g.sh` (c640942f) | Add explicit restoration/CPU preparation modes and separate dry-first restoration helper; worker and original scientific bootstrap unchanged |
| `tests/test_k2_segmented_memory.py` (c640942f) | Reuse tiny real CPU state/resume fixtures for original-vs-restored final state and downstream teacher tests |

Package paths above are under `src/hlt_classification/`. Added
`k2_segmented/restoration.py`, `scripts/queue_jetclass2_k2_restored.sh`,
`tests/test_k2_segmented_restoration.py`, restoration contract and plan amendment.
New spec v3 and `RESTORE_RESUME_IMPORT`, `ABANDONED_EXECUTION`,
`REUSED_NATIVE_GATE` artifacts under `K2_SEGMENTED_*/v1`. The actual original
native acceptance retains its original identity; no new measurement is claimed.
Other campaigns, prior specs, completed outputs and final-test seal stay intact.

## 2026-10-05: 128-GiB SPORC K2 full-state continuation

Execution donor: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**. Scientific donor
remains **`c891da0d45dd3251dea9ea72df975bb96bae3570`**. Developed against local
HEAD `984ec4560cce1c3b6610f69e928dd248218e67cf`. No third-party files copied.

| Donor paths | Use/adaptation |
| --- | --- |
| `src/hlt_classification/k2_segmented/campaign.py`, `runtime.py` (71c1bde2) | Backward-compatible v1 plus v2 graph/resource routing; authenticate the existing segmented donor and import its part2 endpoint into a fresh root |
| `src/hlt_classification/k2_segmented/training.py` (71c1bde2) | Unchanged full-state kernel/serialization; require byte parity with the donor at registration and runtime |
| `scripts/jetclass2_k2_segmented.py`, `queue_jetclass2_k2_segmented.sh`; `sbatch/run_jetclass2_k2_segmented.sh` (71c1bde2) | New `create-128g` CLI mode and separate dry-first helper; existing scientific bootstrap and Slurm worker unchanged |
| `tests/test_k2_segmented.py` (71c1bde2) | Reuse tiny seeded CPU model/cache fixtures; test exact old-vs-migrated continuation and downstream banks |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_campaign.py`, `salience_learned_training.py`, `banks.py`, `production.py`, `execution.py`; `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` (under `src/hlt_classification/`, c891da0d) | Existing original scientific semantics, source/ledger authentication, byte hashes, immutable publication and exact-ID retirement/submission; no scientific donor modules edited |

Added `k2_segmented/memory_migration.py`, `scripts/queue_jetclass2_k2_128g.sh`,
`tests/test_k2_segmented_memory.py`, the 128-GiB contract and active-plan
amendment. New spec `K2_SEGMENTED_CAMPAIGN_SPEC/v2`; new import/copy artifacts
`K2_SEGMENTED_MEMORY_RESUME_IMPORT/v1` and `K2_SEGMENTED_RESUME_COPY/v1`.
Existing checkpoint/payload/task schemas retain their identities. The D025
payload keeps its donor binding; v2 execution receipts explicitly record the
migration. No `Fresh_check`, other campaign, final-test, or environment changes.

## 2026-10-04: additive OSCAR NOISE-K2 targeted recovery

Donor commit: **`0ffb4ba54b02cea6558db5ea8a44995c76ef22d7`**.
New `noise_k2/recovery.py` and `recovery_runtime.py` add an isolated recovery
namespace and output resolver; all original scientific Python modules and
environment helpers must remain byte-identical before native acceptance reuse.

| Donor | Recovery reuse/adaptation |
| --- | --- |
| `src/hlt_classification/noise_k2/campaign.py`, `contracts.py` | Original immutable graph, completed receipts, resource commands, safe paths and byte checks; adapt exact-ID replacement planning into a sibling root |
| `src/hlt_classification/noise_k2/runtime.py`, `data.py` | Same caches, seeded model factory, kernels, inference, selected state and preparation; fit/reduce I/O adapted to resolve original or replacement teachers without changing global source state |
| `src/hlt_classification/jetclass2_delphes/salience_learned_training.py`, `banks.py`, `reporting.py`, `model.py`, `execution.py` | Unmodified scientific kernels, ordered T2 banks, recovery metrics, environment and allocation checks |
| `src/hlt_classification/data/cache_contracts.py`; `src/hlt_classification/scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Atomic immutable publication, content/file hashes, exact submission journals and ledgers |
| `scripts/noise_k2.py`, `scripts/queue_noise_k2.sh`, `sbatch/run_noise_k2.sh`, `sbatch/jetclass2_delphes_common.sh` | Separate recovery CLI/dry helper/worker; unchanged absolute OSCAR environment helper |
| `tests/test_noise_k2.py`, `tests/test_jetclass2_dzfix_fusion_chain.py` | Local graph/cache fixtures and seeded tiny model tests; synthetic adapter parity is not native OSCAR evidence |

New schema family: `NOISE_V3_K2_RECOVERY_*/v1`. Existing scientific/report
artifacts retain their original contracts, hashes and source/job identities.
No donor files modified, external copies, `Fresh_check` imports, producer
changes, or final-test capabilities introduced.

## 2026-10-04: NOISE_V3 K2 100k/50k consumer on OSCAR

Internal donor checkout: **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
Additive package `src/hlt_classification/noise_k2/`, CLI, queue helper and worker;
old native Delphes/SPORC K2 and producer source are unchanged. No third-party
copy, `Fresh_check`, detector generator, old assignment or trained weight import.

| Donor under `src/hlt_classification/` | Reuse/adaptation |
| --- | --- |
| `jetclass2_delphes/concat_k2_views.py`; `scouting/hcwdl_fullcard_salience_matcher.py`, `hcwdl_fullcard_salience_contracts.py`, `hcwdl_fullcard_bottleneck_matcher.py` | Same pre-retention endpoint salience, exact capacity-two integer objective and independent exhaustive solver; physical endpoints replace native Delphes arrays |
| `cms2jc2_response/bridge.py`, `readers.py`; `cms_proxy_ladder/inputs.py`, `views.py` | Physical GeV/mm/validity schema, authenticated ROOT context, 17-feature adapter and salience projection; adapt interpolation with atomic validity groups |
| `literature_proxy_production/population.py`, `output.py`; `literature_proxy/population.py`; `jetclass2_delphes/selection.py`, `split_registry.py` | Frozen shard/raw-entry identity joins, physical bank schema, offline-only ROOT fields, scalar labels and natural Hamilton quotas |
| `jetclass2_delphes/concat_k2_campaign.py`, `concat_k2_data.py`, `cache.py`, `salience_learned_data.py` | Full ten-node recipe/seeds, stratified validation partition and ragged RAM-only/indexed caches |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_model.py`, `salience_learned_training.py`, `salience_learned_graph.py`, `model.py`, `acceptance.py` | Reuse native model/lossless pair storage, scoped FP32/BF16 parity, full original CE/KD kernel, longest-real-batch stress and representative miniature selection; adapt fresh OSCAR admission |
| `jetclass2_delphes/banks.py`, `reporting.py`, `production.py`, `execution.py` | Ordered T2 banks, metrics/recovery, clean pushed source, existing OSCAR site/allocation |
| `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Immutable publication, canonical/file hashes, exact-ID submission journal and dry/live ledgers |

Also imports the independently developed, previously untracked
`src/hlt_classification/literature_proxy_consumer.py` without editing it.
That working-tree donor has no published commit asserted here; its SHA256 at
integration is `1bd2152aff3febb94c04bf5d0473a0f8da11af8551bb0885391d7c68a1ac692d`.
Include that reader and its handoff/contract/tests in the eventual pushed source.
The source dataset's immutable manifest and the original K2 formula's copied
JSON/raw SHA256 are independent provenance anchors, not this development HEAD.

The new Slurm worker sources the existing absolute-path
`sbatch/jetclass2_delphes_common.sh` OSCAR environment. Tests reuse physical
fixtures and real synthetic relocated ROOT/NPZ fixtures from
`test_literature_proxy_consumer.py`/`test_literature_proxy_production.py` and the
ragged cache fixture in `test_jetclass2_dzfix_fusion_chain.py`.
New scientific/execution namespace: `NOISE_V3_K2_*/v1`.

## 2026-10-03: relocated literature NOISE_V3 consumer for Oscar

Internal donor checkout **`71c1bde2d759b11cddb1f5cfbc1d199847bd6eef`**.
New root module `src/hlt_classification/literature_proxy_consumer.py` and thin
inspection/smoke CLI reuse these implementations without modifying original
producer packages or their source-bound science. No third-party code copied,
no `Fresh_check` import, no generator rerun, and no producer metadata rewrite.

| Donor path under `src/hlt_classification/` | Reuse/adaptation |
| --- | --- |
| `data/cache_contracts.py` | Canonical content hashes, file SHA256 and JSON reads |
| `literature_proxy_production/contracts.py`, `output.py`, `codec.py` | Original schema/parent validation, safe local relative paths, physical bank schema and `Particles` decoding |
| `literature_proxy_production/population.py`; `cms2jc2_production/population.py`, `contracts.py` | Exact population/shard metadata, original-entry mask decoding, canonical jet IDs; no split rebuilding |
| `literature_proxy/population.py` | Offline-only branch set and raw native-mm physical conversion; adapt bounded ROOT read loop for paired consumption |
| `literature_proxy_v2/kernel.py`; `literature_proxy_v3/kernel.py`, `contracts.py` | Validate frozen count calibration, NOISE_V3 recipe and direct pilot evidence; no fitting/generation |
| `cms2jc2_response/bridge.py`, `readers.py` | Physical `Particles` validity and ROOT byte authentication before/after access, including early close |
| `jetclass2_delphes/inventory.py`, `split_registry.py`, `selection.py`, `contracts.py` | Original inventory/profile contracts, tree-cycle selection, mapped labels and canonical identity semantics |

Tests reuse existing synthetic production fixtures from
`tests/test_literature_proxy_production.py` (including its v1/v2/v3 pilot fixture
chain), copy their actual ROOT/banks to a new layout and forbid original paths.
New contract `JC2_LITERATURE_RELOCATED_READER/v1` describes read-only runtime
inspection; scientific production/pilot/split contracts remain unchanged.
Dataset producer lineage continues to come from the original, independently
anchored manifest/study, not from this reader-development donor commit.

## 2026-10-03: K2 epoch-boundary segmented continuation

Scientific donor commit **`c891da0d45dd3251dea9ea72df975bb96bae3570`**;
executor developed against local HEAD `98095b680cb968a06bd766cf114007883ce053cc`.
No third-party code copied and no `Fresh_check` import. New package
`src/hlt_classification/k2_segmented/`, CLI/queue wrapper and Slurm worker are
additive; they do not modify the donor checkout or any old scientific module.
The CLI binds every scientific import to the original donor checkout and only
the new executor package to the new pushed checkout.

| Donor path (under `src/hlt_classification/`) | Use |
| --- | --- |
| `jetclass2_delphes/salience_learned_training.py` | Adapt original epoch loop; reuse original AdamW, CE/KD loss, BF16, inference, metrics and selection key; add full-state save/resume without changing schedule or patience |
| `jetclass2_delphes/salience_learned_graph.py` | Frozen training constants and absolute-epoch LR function |
| `jetclass2_delphes/concat_k2_runtime.py`, `concat_k2_campaign.py` | Read-only source registration, completed-prefix acceptance, native models/caches, source teacher bank, metrics/reporting/export primitives |
| `jetclass2_delphes/concat_k2_submit.py` | Original canonical science plan and exact walltime parsing |
| `jetclass2_delphes/concat_k2_model.py`, `model.py` | Original lossless pair storage and scoped deterministic parity backend; installed environment identity and model contract |
| `jetclass2_delphes/salience_learned_data.py`, `banks.py`, `contracts.py` | Indexed miniature cache, identity-joined T2 banks and safe relative paths |
| `jetclass2_delphes/production.py`, `execution.py`, `submission.py` | Clean pushed source, SPORC allocation checks, submission-intent protection |
| `data/cache_contracts.py`; `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Immutable publication, content hashes, exact-ID journals and ledger validation |
| `sbatch/jetclass2_delphes_common.sh` | Existing absolute-path SPORC environment setup; inherited from the new pinned executor checkout |

New contracts are `K2_SEGMENTED_*/v1`, `K2_SEGMENT_CHECKPOINT/v1`, and
`K2_FULL_TRAINING_STATE/v1`. Authority and evidence limits are documented in
the K2 plan amendment, segmented contract and HANDOFF. Existing full-size and
pilot schemas are not rewritten.

## 2026-10-03: genuine CMS common-interface direct/coarse comparison

Internal donor checkout: **`b7fd9c63d5f747983b4ab603b86a315d68bcc24a`**.
New implementation under `src/hlt_classification/cms_feature_ladder/`; no
third-party source copied and no runtime `Fresh_check` import. Older scientific
modules/specs are unchanged. New authority is CMS_FULLSIM_FEATURE_LADDER/v1.

| Donor (relative to `src/hlt_classification/`) | Reuse/adaptation |
| --- | --- |
| `scouting/schema.py`, `splits.py`, `selective_assignment.py`, `identity.py` | Authenticated native file roles, 15 labels, seed1337 proportional identity-rank subsets; new200k/50k budget |
| `scouting/hcwdl_homotopy.py`, `repair.py`, `inputs.py` | Raw endpoint projection with validity, retaining lost tracks; native21 transforms for opt-in control, no inversion of clipped inputs |
| `cms_proxy_ladder/views.py`, `inputs.py`, `campaign.py` | PT_LINEAR matching, persistent support, rational U/D coordinates, keyed interpolation, shared17 normalization, paired coordinate seeds; raw binary PID flags explicitly preserved for CMS |
| `cms2jc2_response/bridge.py` | Validated physical `Particles` container only; **not** `from_cms`, whose lost-track filtering would change this population |
| `cms_salience_learned/data.py`, `storage.py`, `training.py` | Bounded raw ROOT projection/process map, RAM batch interface, immutable byte IO; tensor/autocast/AdamW utilities, no learned-fusion scientific graph |
| `jetclass2_delphes/campaign.py`, `runner.py`, `acceptance.py`, `model.py`, `execution.py` | Exact literature schedule, adapted15-class kernel/parity, installed environment hashing and SPORC allocation checks |
| `models/particle_transformer.py`, `scouting/evaluation.py` | Installed Weaver factory/backbone; CMS metrics with explicit censored R50 representation |
| `scouting/hcwdl_authorization.py`, `hcwdl_exact_dag_submission.py`; `data/cache_contracts.py` | Pushed clean source validation, exact DAG/journal, content hashes/immutable publication |
| `cms2jc2_response/measurement.py`; `sbatch/jetclass2_delphes_common.sh` | Process-tree RSS measurement and absolute SPORC environment setup |

The new runner is single-view, cold-start, immediate-parent C25/P75, not the
native learned-fusion acquisition/withdrawal donor. The opt-in CMS21 control
shares support/p4/identity changes with SHARED17 and holds trim=False. This
compares representations, not an assertion of an exact old21-feature result.
Tests/evidence and the remaining real-GPU gate are recorded in HANDOFF.

## 2026-10-03: additive literature gate-to-science follow-up

In-repository donor commit **`99a1e231e38fe5a07ce5a31a4f4a74bb9ec40539`**.
No external code copied, license change or `Fresh_check` import. New reusable
execution module is `src/hlt_classification/literature_ladder_followup.py`.

| Donor | SHA-256 at inspection | Retained use |
| --- | --- | --- |
| `cms_proxy_ladder/literature.py` | `316734ea5dd44d3fb738c00e81c3cfb88aeaa41c6456177bb7124ebc1a9d7fd0` | Original pinned gate/profile/science validation and exclusive submit claim; no changes |
| `cms_proxy_ladder/submission.py` | `a6e969c62c049cb2f3721afee06447fe6ba53109e7e86a81a2266eb1f9a47d02` | Exact recomputed plans, canonical dry ledger and site checks, delegated to old CLI |
| `scouting/hcwdl_exact_dag_submission.py` | `65c031db95630d034d19112d8d0ff76af5189fc9a43e7ad7e9da1430b5df4a6b` | Authenticate old gate journal; original idempotent science DAG submission |
| `cms2jc2_production/recovery_controller.py` | `eb0cb1d862e3903ef9799408f97e4b6f23adda7726778ed7ae8c0c915cd785d5` | Pattern only: kernel controller lock and bounded exact-job polling; no imports |

Donor paths above are relative to `src/hlt_classification/`. The embedded
read-only inspection runs in a fresh interpreter importing the original gate
checkout, not the controller checkout. This preserves already-running source
locks. Execution amendment and operational receipt v1 are documented in
`docs/contracts/JETCLASS2_LITERATURE_LADDER_FOLLOWUP.md`; scientific schema
versions remain unchanged. Tests: `tests/test_literature_ladder_followup.py`;
local/remote evidence and remaining launch step are recorded in `HANDOFF.md`.

## 2026-10-03: literature V3 200k/50k SPORC debug DIRECT + COARSE

Repository-local donor commit **`ebd5bc1ae4eb6d6c11bfba5e1a6cdc5fd487653a`**.
No third-party donor or runtime `Fresh_check` import. The dataset generator
and its recipe/calibration are reused unchanged; new classifier versions
explicitly identify the literature rather than CMS-fitted endpoint.

| Donor files (under `src/hlt_classification/` unless noted) | Reuse/adaptation |
| --- | --- |
| `cms_proxy_ladder/release.py`, `data.py`, `views.py`, `inputs.py`, `cache.py` | Versioned 200k/50k literature release, native-unit paired reader, fresh salience matching, persistent skeleton/offline tail, exact D000 and existing RAM cache |
| `cms_proxy_ladder/campaign.py`, `gate.py`, `production.py`, `submission.py`, `contracts.py` | Existing direct/coarse coordinates, seeds, C25/P75 training, reports and exact DAG; new literature source lock, full-population debug-only gate/profile and nine-fit science variant |
| `literature_proxy_production/campaign.py`, `population.py`, `output.py`, `contracts.py`; `literature_proxy_v3/contracts.py`; `literature_proxy/population.py` | Validate frozen source/bundle/population, ordinary role manifests and physical blocks; convert original offline fields without CMS unit/sign conversion; no generator execution |
| `cms2jc2_production/output.py`; `cms2jc2_response/bridge.py`, `generation_benchmark_data.py` | Reuse physical-bank decoder/schema and offline branch allowlist only, not learned response or CMS bridge conversion |
| `jetclass2_delphes/{campaign,model,runner,acceptance,execution,reporting,inventory,selection,contracts}.py` | Reuse model/recipe, installed-Weaver parity, single-GPU kernels, per-class R50, SPORC site and inventory-bound identities |
| `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py` | Existing exact-ID journal/ledger; optional explicit environment for sanitized literature submissions, plus adapter-owned exclusive submission claim |
| `scripts/jetclass2_cms_proxy_ladder.py`, `run_jetclass2_cms_proxy_ladder_task.py`; `sbatch/jetclass2_delphes_common.sh` | Thin new CLI/helper; retain existing worker and absolute environment setup |
| `tests/test_literature_proxy_production.py`, `test_cms_proxy_ladder.py` | Synthetic producer fixtures and classifier contract patterns; new reader/gate/DAG/claim tests |

New authority: `docs/plans/JETCLASS2_LITERATURE_V3_200K_DIRECT_COARSE_PLAN.md`
and `docs/contracts/JETCLASS2_LITERATURE_V3_LADDER.md`. No historical artifact
is relabeled or overwritten; no final-test particle capability is added.

## 2026-10-03: frozen literature v3 full production dataset

Repository-local donor commit **`d505e807500b7e0f4149f940413c542b63b308d2`**.
New code: `src/hlt_classification/literature_proxy_production/`; old scientific
modules remain unchanged. No third-party donor or `Fresh_check` dependency.

| Donor files (under `src/hlt_classification/`) | Production reuse/adaptation |
| --- | --- |
| `literature_proxy_v3/kernel.py`, `inputs.py`, `worker.py`, `campaign.py`, `contracts.py` | Reuse frozen generator and recipe verbatim, completed v3/v2/v1 authentication, saved-v3 replay, exact source/hash chain |
| `literature_proxy_v2/kernel.py` | Validate inherited full-precision calibration and original parent lineage; never recalibrate |
| `literature_proxy/population.py`, `campaign.py` | Reuse exact raw OFFLINE conversion, padded particle keys and branch allowlist, clean/pushed source and exclusive failed-attempt locks |
| `cms2jc2_production/population.py` | Reuse metadata-only TRAIN_1M population builder including unchanged validation subset domain; new sharding and raw reader do not call its CMS-compatible physical converter |
| `cms2jc2_production/contracts.py`, `storage.py` | Adapt truthful flags/atomic references and path-safe quota reservations into a separate versioned namespace |
| `cms2jc2_production/campaign.py` | Adapt fresh persistent-root non-overlap checks only; no learned campaign/gate dependency |
| `cms2jc2_production/submission.py` | Adapt exact array-element identity, scheduler/environment checks, journals and terminal-only recovery; replace staged pilot/bulk submissions with one frozen preflight/array/finalizer DAG |
| `cms2jc2_production/output.py` | Adapt physical-bank validation, receipt binding and manifest checks; add train/validation-only independent releases |
| `cms2jc2_response/generation_benchmark_engine.py` | Copy pure physical digest, pack, lossless codec/readback; adapt bounded ordered process/writer design with ONLY literature v3 generation |
| `cms2jc2_response/readers.py`, `measurement.py`, `bridge.py` | Reuse byte-authenticated ROOT open, Linux process-tree measurement and physical schema; no CMS reading/learning or units conversion |
| `data/cache_contracts.py`, `jetclass2_delphes/inventory.py`, `contracts.py`, `split_registry.py` | Atomic bytes, identities, tree/inventory authentication and canonical split masks |

The new CLI/queue/Slurm wrapper adapt
`scripts/jetclass2_literature_proxy_noise.py`,
`scripts/queue_jetclass2_literature_proxy_noise.sh` and
`sbatch/run_jetclass2_literature_proxy_noise.sh`. Tests reuse synthetic ROOT
and sealed-pilot fixtures from `tests/test_literature_proxy_v3_campaign.py`,
`tests/test_literature_proxy_v2_campaign.py`, `tests/test_jetclass2_delphes.py`
and `tests/test_jetclass2_delphes_split_registry.py`, with scheduler/codec
test patterns from `tests/test_cms2jc2_proxy_production.py`. Production role
counts are not configurable; only isolated tests patch miniature counts.

## 2026-10-02: final noise-only literature proxy v3

Repository-local donor commit `d4b5d5fda19a6401442cb6b605ad22c2b7013e5e`.
New package: `src/hlt_classification/literature_proxy_v3/`. No external source,
learned CMS response or `Fresh_check` runtime dependency.

| Donor files (relative to `src/hlt_classification/`) | v3 reuse/adaptation |
| --- | --- |
| `literature_proxy_v2/kernel.py` | Reuse topology and probability validation unchanged; copy generation equations with only tracking amplitude 3 to 4 and kinematic amplitude 1.5 to 2 |
| `literature_proxy_v2/inputs.py`, `worker.py` | Authenticate original OFFLINE and saved count38 blocks and complete receipts; replace calibration passes with exact inherited calibration, add every-jet structure/count equality |
| `literature_proxy_v2/campaign.py`, `contracts.py` | One-job source pinning, immutable publication, submission protections; new v3 namespace and pinned v2 ancestors |
| `literature_proxy/kernel.py`, `worker.py`, `diagnostics.py` | Unchanged random streams, physical arrays/RSS, inclusive moments/conditional statistics/exports; add separate v3 paired-response diagnostics |
| `literature_proxy/contracts.py`, `cms2jc2_response/bridge.py`, `data/cache_contracts.py`, `jetclass2_delphes/contracts.py` | Hash/reference primitives, validated particles, p4/wrapped phi, deterministic NPZ and confined artifact paths |

The new noise CLI, queue helper and Slurm wrapper adapt the donor's
`scripts/jetclass2_literature_proxy_count38.py`,
`scripts/queue_jetclass2_literature_proxy_count38.sh` and
`sbatch/run_jetclass2_literature_proxy_count38.sh`. Tests reuse the donor's
`tests/test_literature_proxy.py` and `tests/test_literature_proxy_v2_campaign.py`
fixtures (including their JetClass2 synthetic ROOT/split fixtures). All v1/v2
source remains unchanged. No third-party source/license changes. Verification
and genuine Tigris status are recorded in `HANDOFF.md`.

## 2026-10-02: count-targeted literature proxy v2

Repository-local donor commit `08cbf06309ff7bb2bdeccf6da2874e21de43da87`.
New code lives under `src/hlt_classification/literature_proxy_v2/`. No external
donor, learned CMS response, or `Fresh_check` dependency was introduced.

| Donor files (under `src/hlt_classification/`) | Reuse/adaptation |
| --- | --- |
| `literature_proxy/kernel.py` | Physical response equations, crowding, common per-operation random streams and Response container; v2 separately scales PID/tracking/kinematics and changes loss topology |
| `literature_proxy/campaign.py`, `contracts.py` | Clean/pushed source guard, exclusive lock and immutable/hash primitives; new v2 campaign and artifact namespace |
| `literature_proxy/population.py` | Recompute exactly the same authenticated training selection; no new ROOT reader |
| `literature_proxy/worker.py` | Authenticate completed parent receipt, physical-array extraction, per-process RSS conventions; new two-pass calibration/generation worker |
| `literature_proxy/diagnostics.py` | Existing moments, masks, fixed-bin/conditional statistics and PDF/CSV; optional bins/sides preserve v1 defaults |
| `cms2jc2_response/bridge.py`, `data/cache_contracts.py`, `jetclass2_delphes/contracts.py` | Physical particle constraints, deterministic NPZ, byte/content hashes and confined artifact paths |

New CLI/helper/Slurm wrapper adapt the donor's
`scripts/jetclass2_literature_proxy.py`,
`scripts/queue_jetclass2_literature_proxy_pilot.sh`, and
`sbatch/run_jetclass2_literature_proxy_pilot.sh`. Synthetic fixtures reuse
`tests/test_literature_proxy.py`, `tests/test_jetclass2_delphes.py`, and
`tests/test_jetclass2_delphes_split_registry.py` from the same commit. No
original scientific v1 kernel, population or saved result was replaced.
The new count calibration is explicitly a training-only benchmark choice,
not a CMS/literature-measured rate or a classification-performance fit.

## 2026-09-30: proxy dataset consumer documentation

The consumer handoff references existing production, recovery, bridge and
JetClass2 label/identity APIs at repository commit
`3655955f6ef7184cc6fdb3b0e68bdc6d089e9fe2`; it migrates no implementation or
external donor files. Its example test reuses synthetic ROOT/split fixtures
from `tests/test_jetclass2_delphes.py` and
`tests/test_jetclass2_delphes_split_registry.py` at that commit. The separately
tested production reader remains authoritative for physical bank validation.
No new schema, response semantics or final-test permissions are introduced.

## 2026-09-30: committed proxy training audit

Repository-local donor commit `314e9faa680cad677aca9c9cd158d83cf973ea0b`:

- `cms2jc2_production/{campaign,contracts,population,output,recovery}.py`:
  import immutable provenance, safe paths, exact offline-only source membership,
  physical bank schema/identity checks and repaired receipt validation unchanged.
- `cms2jc2_response/{dev_diagnostics,metrics,features,bridge}.py`: reuse canonical
  observables, validity masks, offline-only cohorts, transforms and bridge units.
  The independent audit compares saved populations; it does not reuse a paired
  same-population score for unpaired CMS-vs-JetClass2 samples.
- `cms2jc2_response/{dev_data,provenance,generation_benchmark_engine}.py`:
  reuse authenticated metadata references, environment/source evidence and
  physical digests. The audit never calls the generation kernel.
- `tests/test_cms2jc2_proxy_production.py`,
  `test_cms2jc2_generation_benchmark.py`, and JetClass2 split-registry fixtures:
  reuse synthetic physical schemas and ROOT data for the new audit tests.

New logic lives in `cms2jc2_proxy_audit/`; existing frozen producer/response
modules are unchanged. New plan and `CMS2JC2_PROXY_AUDIT_* /v1` contracts prevent
diagnostic completion from masquerading as dataset completion or qualification.
No external donor files, Fresh_check imports or remote changes.

## 2026-09-30: source-pinned production execution recovery

Repository-local donor commit `b6f88defce6f357969b2e6ded80585db439540a3`,
plus the local array-element authentication fix documented immediately below:

- `cms2jc2_production/{campaign,submission,worker,output}.py`: retain original
  admission, exact submission intents, resource guards, worker generation
  and bank validation. Enumerated operational entry points add explicit
  repair lineage; the generation kernel is unchanged.
- `cms2jc2_production/{contracts,storage,population}.py`: reuse immutable
  content/parent hashes, safe paths, claims, reservations and frozen identities
  without changes. New recovery modules use these rather than rebuilding data.
- Existing production CLI/queue helpers: reuse Tigris environment activation,
  absolute source paths, single-thread numerical libraries and detached log
  handling in new `cms2jc2_proxy_recovery.py` and its queue shell helper.
- `tests/test_cms2jc2_proxy_production.py`: reuse synthetic bank, scheduler and
  source fixtures for `test_cms2jc2_proxy_recovery.py`; local mocked timings
  do not replace the required real retained/retried pilot measurements.

New execution-repair plan/contract version the repaired attempt, shard and
manifest kinds instead of changing legacy artifact meaning. No external
donor migration, Fresh_check runtime import, scientific refit or remote action.
Exact local verification and remaining remote step are recorded in HANDOFF.

## 2026-09-30: exact production Slurm array-element lookup

Repository-local donor commit `b6f88defce6f357969b2e6ded80585db439540a3`:
`src/hlt_classification/cms2jc2_production/submission.py` and
`tests/test_cms2jc2_proxy_production.py`. Correct raw-parent ambiguity in
`worker_identity`/`fields`, retaining original submission and resource guards.
No donor file migration or Fresh_check import; no response kernel, physical
writer, population, random-key or scientific-schema change. Production plan
and contract clarify the exact selector and unchanged source-pinning boundary.
Test and user-supplied remote evidence are recorded in HANDOFF.

## 2026-09-29: explicit 36-of-57 frozen confirmation amendment

Repository-local donor commit `f3db2ce8ff0f8e5716aab19df4cba9677f72c06c`:

- `cms2jc2_response/frozen_joint_campaign.py`, `frozen_joint_metrics.py`,
  `bdz_worker.py`, `bdz_audit_metrics.py`: unchanged frozen protocol, additive
  diagnostic merging, file-bootstrap checks and significance diagnostics.
  Called by new `cms2jc2_response/reduced_confirmation.py`; original complete
  membership/report validation remains untouched. Only the new contract
  authorizes the fixed prefix, with explicit missing-source/file coverage.
- `cms2jc2_response/dev_submission.py`, `dev_campaign.py`, `dev_worker.py`,
  `dev_data.py`: reuse exact-intent/ledger scheduler identity, immutable outputs,
  source/environment authentication and publication; new dispatch/authorization
  for a single report task. Retirement is separately authorized and preserves
  completed excluded artifacts and all included artifacts.
- `cms2jc2_production/{campaign,output}.py`: new explicit reduced-evidence
  study/manifest kinds. Original full-evidence route and physical generation
  semantics remain unchanged. No legacy donor/Fresh_check runtime import,
  license change, data migration, raw-particle read or remote action here.
- Existing synthetic frozen diagnostics and production bank fixtures support
  new retirement/refusal, report aggregation, opt-in, provenance and sealed
  reader tests. Test evidence is recorded in HANDOFF, not remote qualification.

## 2026-09-29: 2.25M frozen proxy dataset production

Repository-local donor commit `b88a35123e6f7658a4b1679cd797b7f157c4602b`:

- `cms2jc2_response/generation_benchmark_engine.py`: unchanged JOINT replica-0
  generation, ordered spawn pool, physical/key digests, float64 bank packing,
  deterministic lossless encoding and full schema/readback checks. Called
  directly; old benchmark writer/caps are not weakened or reused for bulk.
- `cms2jc2_response/generation_benchmark_data.py`, `readers.py`, `bridge.py`:
  original offline-only branch capability, bounded windows, authenticated ROOT
  open, stable row identity, unit/validity conversion and physical schema.
  Production adapts the window reader to its new versioned population; original
  reader semantics remain unchanged.
- `cms2jc2_response/generation_direct_campaign.py`,
  `frozen_joint_campaign.py`, `dev_campaign.py`, `dev_data.py`, `provenance.py`:
  completed gate/confirmation and immutable donor authentication, existing
  scientific source matching and native-library/environment pinning.
- `cms2jc2_response/measurement.py`: real Linux process-tree resource sampling.
- `jetclass2_delphes/{inventory,split_registry,contracts}.py` and
  `data/cache_contracts.py`: input/profile validation, packed masks, existing
  original identities, canonical hashes and atomic no-overwrite publication.
- Existing direct/benchmark tests, `test_cms2jc2_bounded.LocalMeasurement`, and
  JetClass2 registered synthetic ROOT fixtures: fixture inputs and local-only
  measurements. Synthetic timing is explicitly forbidden as bulk admission.
- Existing response queue/wrapper scripts: activation, absolute project paths,
  single-thread child environments and disconnect-safe heartbeat patterns.

New package `cms2jc2_production`, separate CLI/worker/helper, plan and
`CMS2JC2_PROXY_* /v1` contracts add durable production storage, new authorized
test materialization semantics, pilot admission, arrays, recovery, manifests
and the train/validation physical-bank reader. No fitted response, tracking
map, random key algorithm or old test-access boundary is modified.

## 2026-09-29: direct Tigris frozen-JOINT generation

Repository-local donor commit `9215fdcf940c116d5d89675a42bd8c82bdec05ea`:

- `cms2jc2_response/generation_benchmark_{campaign,data,engine,worker}.py`:
  authenticated GEN lineage/bundle/TRAIN membership, unchanged bounded
  offline-only ROOT reader, bounded once-per-process generation, lossless
  NPZ/readback, physical/key/identity digests and filesystem measurements.
- `generation_portable_{campaign,data,worker}.py`: registered Tigris CPU site,
  resource projection, exact discrete/tolerant floating comparison, source
  and environment separation. Reused through calls; PORT semantics unchanged.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, `dev_restart.py`,
  `storage.py`, `measurement.py`, `provenance.py`: claims, immutable submission
  and output receipts, pre-existing execution-only source allowlist, native
  environment pinning, cap/free-space checks. Dispatcher/CLI/source-list and
  scheduler-worker recognition additions are execution-only.
- `scripts/queue_cms2jc2_portable_benchmark.sh` and
  `sbatch/run_cms2jc2_response_portable_cpu.sh`: heartbeat/stdin/exit and site
  activation patterns. New direct helper adapts them; original files unchanged.
- GEN/PORT response fixtures and `LocalMeasurement` tests: deterministic
  synthetic data, real Generator/spawn replay and mocked scheduler lifecycle.

New `generation_direct_{campaign,worker}.py` provide separate TG_* v1
contracts, reuse the existing 64-jet reference and read the same 10k TRAIN
ROOT population on Tigris. No new external donor, fitting, scientific kernel,
key-domain, unit/sign convention or Fresh_check dependency. Full 10k SPORC
parity is not claimed; only the existing 64-row reference is available.

## 2026-09-29: explicit dzfix partial-snapshot benchmark location

Repository-local donor commit `292e38896d7ae8bf9e973e9fd1c1efed8dc71548`:
`cms2jc2_response/generation_benchmark_{data,campaign}.py` supplies the existing
root guard and authenticated TRAIN membership/reader; the corresponding
benchmark tests supply synthetic ROOT, inventory/profile and campaign fixtures.
The guard now also accepts the exact documented `_partial_v1` directory name.
`docs/JETCLASS2_DZFIX_SALIENCE_MATCHING_HANDOFF.md` and
`docs/plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md` at that commit
record the actual SPORC snapshot and explain its noninterchangeable inventory.
No new external donor, particle transform, response fit, key-domain change or
Fresh_check dependency. GEN v1 membership and PORT replay semantics unchanged.

## 2026-09-29: separate Tigris portable JOINT benchmark

Repository-local committed donors remain
`455557980168a131e22b6990c902e1c52815a52f`: `cms2jc2_response/{dev_campaign,
dev_submission,dev_worker,storage,measurement,provenance,bridge,readers,
bdz_joint_maps,response}.py`. Science kernels, fitted maps and key domains are
unchanged. Exact scheduler identity is extended only for PORT_STAGE's new worker
and Tigris's site-assigned (not guessed SPORC) QoS; old contracts retain their
checks. New CLI, helper and source-snapshot registrations are execution-only.

The preceding, still-uncommitted 2026-09-29 GEN benchmark supplies
`generation_benchmark_{campaign,data,engine,worker}.py`: membership, frozen
bundle, bounded offline reader, once-per-process initialization, deterministic
NPZ encoding/readback and measurements. These are co-developed in this worktree,
NOT falsely attributed to the earlier committed donor. The engine additionally
admits 36/72/144 workers for new PORT task registries; old GEN tasks stay frozen.
The new `generation_portable_{campaign,data,worker}.py` adds source-pinned packet
export/import and explicitly separate cross-architecture tolerance contracts;
it never weakens the GEN donor-environment validator. The native CPU wrapper and
dry-default heartbeat helper adapt existing queue patterns. No external donor or
Fresh_check import is added. New tests reuse existing synthetic physical/model
fixtures and supplement them with relocatable packet and lifecycle checks.

## 2026-09-29: generation-only JOINT engineering benchmark

Repository-local donors at `455557980168a131e22b6990c902e1c52815a52f`:

- `cms2jc2_response/bdz_joint_campaign.py`, `bdz_joint_worker.py`,
  `bdz_joint_maps.py`, `response.py`: authenticate and reuse the selected JOINT
  runtime unchanged, one label-independent realization per jet. No numerical
  implementation, fitted parameters, or key domains are edited.
- `frozen_joint_campaign.py`, `frozen_joint_worker.py`, `dev_parallel.py`:
  separate gate/next-stage pattern, bounded spawn queue, native-thread limiting,
  conservative resource estimates and frozen-replay checks, adapted for a
  distinct TRAIN generation capability with no confirmation/test access.
- `bridge.py`, `readers.py`, `dev_data.py` plus
  `jetclass2_delphes/{inventory,splits,split_registry,contracts}.py`: authenticated
  metadata, stable row identities, offline-only physical bridge and file-byte
  verification. A new bounded-window reader is separate from old readers.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, `storage.py`,
  `measurement.py`: immutable records, receipts, exact CPU submission journals,
  allocation claims and storage accounting. Changes to existing modules are
  additive execution routing/source snapshots/CLI only; no old scientific
  semantics or no-JC2 boundary is relaxed.
- `scripts/queue_cms2jc2_frozen_joint.sh`: adapted heartbeat/stdin/exit handling
  for the new dry-default queue helper, with an explicitly supplied reviewed
  plan hash. `frozen_joint_queue.py`: read-only resource-probe pattern, without
  interpreting a single-job estimate as multi-job completion evidence.
- Existing response-science, JOINT map, synthetic ROOT/split-registry and
  LocalMeasurement test fixtures are reused; mocked allocation evidence cannot
  authorize real production. New NPZ encoding/readback/digest implementation
  stays in `generation_benchmark_engine.py`; no external donor or Fresh_check
  runtime dependency is added.

## 2026-09-28: frozen JOINT collection-response confirmation

Repository-local donors at `7b57a32de74e4bcff4fa83e1ffcde67b5fcd0872`:

- `cms2jc2_response/bdz_joint_campaign.py`, `bdz_joint_worker.py`,
  `bdz_joint_maps.py`: authenticate the completed selected JOINT response and
  replay unchanged maps/random keys. The new `frozen_joint_worker.py` adapts
  ordered, bounded spawn execution to exactly JOINT and B_DZ. Acceptance compares
  both against the original five-candidate kernel; there is no new fitting API.
- `bounded_data.py`, `dev_data.py`, `readers.py`, `splits.py`: authenticated
  metadata, identities, streaming and access-lock pattern. New
  `frozen_joint_data.py` admits ALL outer-confirmation rows only through its
  separately claimed capability; the old fit-only reader is unchanged.
- `dev_diagnostics.py`, `bdz_audit_metrics.py`, `bdz_worker.py`: unchanged bounded
  additive histograms, tracking/PID diagnostics and canonical merges. New
  `frozen_joint_metrics.py` adds predeclared thresholds and paired file-bootstrap
  supported/rejected/inconclusive evidence, not a new development selection.
- `dev_campaign.py`, `dev_submission.py`, `storage.py`, `measurement.py`: existing
  source/receipt/submission/allocation/storage checks. Only execution dispatch,
  source extras and CLI are extended. `frozen_joint_queue.py` probes actual CPU
  shapes with test-only requests; it does not mutate existing jobs.
- `scripts/queue_cms2jc2_bdz_joint.sh`: heartbeat/stdin/exit-status pattern adapted
  into `queue_cms2jc2_frozen_joint.sh`, with separate gate/confirmation and no
  cancellation or automatic follow-up.
- JOINT/B_DZ/audit/bounded synthetic ROOT test ancestry reused. The
  `test_cms2jc2_frozen_c.py` fixture adds its synthetic authenticated `cms_root`
  field; no donor scientific implementation or real dataset is modified.

Intentional new semantics are under additive FROZEN_* v1 contracts and the
frozen JOINT confirmation plan. No external code, model, particle bank or
Fresh_check runtime import was copied. This finite collection-response study
does not replace the original production qualification or authorize JC2 transfer.

## 2026-09-27: bounded PID-safe / joint tracking repair

Repository-local donors at `b1d59811609020b40c65f9f4063128be1faf526b`:

- `cms2jc2_response/bdz_maps.py`: unchanged historical controls/quantile knots;
  `bdz_joint_maps.py` adds coordinate-local PID identity fallback and joint
  log-error / error-rank-conditional asinh-significance transport. No old map
  or model is rewritten; new semantics use new BDZ_JOINT_* contracts.
- `bdz_worker.py`, `bdz_audit_worker.py`: bounded ordered spawn orchestration
  adapted into `bdz_joint_worker.py`; original aggregation called directly for
  exact historical replay. New calibration reads residual jets only.
- `bdz_audit_metrics.py`, `bdz_metrics.py`, `dev_diagnostics.py`: unchanged
  diagnostic accumulators reused; new PID-aware score, guard and rendering
  live separately in `bdz_joint_metrics.py` and `bdz_joint_campaign.py`.
- `bdz_campaign.py`, `bdz_audit_campaign.py`, `bdz_audit_debug.py`: completed
  donor/receipt/source checks reused, including the completed debug audit.
- `scripts/queue_cms2jc2_bdz_audit_debug.sh`: phase heartbeat/stdin/exit-status
  pattern adapted into the new joint helper WITHOUT retirement or cancellation.
- `tests/test_cms2jc2_bdz_tuning.py`, `test_cms2jc2_bdz_audit.py` and their
  bounded/B/C synthetic ROOT fixture ancestry supply regression patterns.

Only the existing execution-only `dev_campaign.py`, `dev_worker.py` and CLI
are extended for source/stage/worker dispatch. No donor scientific file is
edited; no external code or Fresh_check runtime dependency is introduced.

## 2026-09-26: significance audit pending-job debug replacement

Repository-local donors at `a61e4ea1d09d0cca80251c7b8fc7e70e72577fb5`:

- `cms2jc2_response/c_topology_debug.py`: exact-ledger pending-only retirement,
  scheduler identity/account checks, admission-before-cancel, immutable evidence
  and fresh debug study adapted to `bdz_audit_debug.py`. Audit migration also
  refuses original claims and refreshes accounting after slow authentication.
- `bdz_audit_campaign.py`, `bdz_audit_worker.py`, `bdz_audit_metrics.py`:
  unchanged frozen protocol, acceptance, generation, replay and reporting
  primitives. `bdz_audit_debug_worker.py` is the external-acceptance adapter;
  it publishes new-stage results and references the original acceptance hash.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py`, and the development
  CLI: additive contract/source/retirement/dispatch routing on the existing
  execution-only allowlist. No enlargement of that scientific-source exception.
- `scripts/queue_cms2jc2_bdz_audit.sh` and `queue_cms2jc2_bdz_tier3.sh`:
  dry/live source-pinned helper pattern, with phase/heartbeat feedback added in
  the new debug helper only. Neither donor helper is modified.
- Audit and C-debug tests plus their B_DZ/bounded/B/C synthetic ROOT ancestry:
  reused fixtures and lifecycle tests; no external donor or Fresh_check import.

## 2026-09-26: frozen B_DZ significance replay

Repository-local donors at `6b9a4bc1b3a9172d8b3ca63deb8c2c9e49c387d8`:

- `cms2jc2_response/bdz_worker.py`: unchanged historical accumulator invoked
  directly; bounded ordered spawn-worker orchestration adapted into
  `bdz_audit_worker.py`. No response/quantile-map refit is introduced.
- `bdz_campaign.py`, `bdz_tier3.py`, `bdz_maps.py`, `bdz_metrics.py`: completed
  donor/calibration lineage, numerical maps and historical score validation
  reused unchanged. New PID/value-error diagnostics live in separate modules.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py`:
  additive stage/source/worker/CLI dispatch; immutable submission/claim and
  receipt machinery reused without changing old scientific semantics.
- `scripts/queue_cms2jc2_bdz_tier3.sh`: source/environment and dry/live wrapper
  pattern adapted to `queue_cms2jc2_bdz_audit.sh`, without any retirement or
  cancellation path.
- `tests/test_cms2jc2_bdz_tuning.py` and its bounded/B/C synthetic ROOT fixture
  ancestry: donor replay, immutable receipts and CPU lifecycle test patterns.

The new plan/contracts explicitly permit bounded hashed forensic examples,
not durable full particle banks. No external donor or Fresh_check import.

## 2026-09-25: B_DZ tier3 execution replacement

Repository-local donors at `1cdf6b890ce0b88de174948916593c106ddc8a06`:

- `src/hlt_classification/cms2jc2_response/c_topology_debug.py`: exact-ID,
  pending-only retirement, fresh-root reuse, immutable retirement evidence and
  scheduler race checks adapted into `bdz_tier3.py` (now also refuses claims).
- `bdz_campaign.py`, `bdz_worker.py`, `bdz_maps.py`, `bdz_metrics.py`: reused
  byte-for-byte for protocol, gate/map ancestry and all numerical evaluation;
  selection-display validation adapted in the new module, not the donor.
- `dev_campaign.py`, `dev_submission.py`, `dev_worker.py` and
  `scripts/cms2jc2_response_dev.py`: additive execution routing only.
- `tests/test_cms2jc2_c_topology_debug.py`, `test_cms2jc2_bdz_tuning.py` and
  their synthetic ROOT ancestors: fixture/lifecycle test patterns reused.

New plan/contracts and queue wrapper change only execution placement. No
external model or Fresh_check runtime dependency is introduced.

## 2026-09-25: B_DZ tracking-calibration screen

Repository-local donors at `b79e76be2ad7fb8884f6bb5128ae5b22e7aaa0a1`:

- `src/hlt_classification/cms2jc2_response/bounded_campaign.py`,
  `bounded_models.py`, `bounded_metrics.py`: authenticated original B_DZ
  selection/recipe and conditional TV semantics reused without modifying donors.
  New `bdz_campaign.py` handles only the known guard-list order normalization.
- `bounded_worker.py`, `dev_parallel.py`, `c_diagnostic_worker.py`: bounded
  spawn-process orchestration, identity digest and tight historical replay checks
  adapted/reused by `bdz_worker.py`.
- `dev_diagnostics.py`, `bridge.py`: physical/conditional histograms, plotting,
  immutable particles and validity semantics reused directly.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py`: additive
  routing/source-plan registration; existing submission and claim machinery reused.
- `tests/test_cms2jc2_bounded.py`, `test_cms2jc2_b_tracking.py`,
  `test_cms2jc2_frozen_c.py`: real synthetic ROOT/receipt fixture ancestry reused
  in `tests/test_cms2jc2_bdz_tuning.py`.

The marginal tracking maps, tail-balanced metrics and BDZ_* contracts are new;
no Fresh_check runtime dependency or external model donor is introduced.

## 2026-09-25: bounded final B / B_DZ / BC comparison

Repository-local donors at `7d56425f3bb88b6ea36bbc9bd80e81fb166e5bb2`,
under `src/hlt_classification/cms2jc2_response/`:

- `b_tracking.py`, `c_diagnostic.py`: completed B/C fit/report/receipt import
  reused directly by `bounded_campaign.py`; scientific source equality remains.
- `b_tracking_worker.py`, `dev_parallel.py`: bounded spawned-process chunking,
  identity checking, histogram aggregation and progress patterns adapted in
  `bounded_worker.py`. New worker merges in bounded deterministic order.
- `dev_data.py:iter_sample`, `_sample`: reader/hashed membership pattern adapted
  into a separately claimed, 20k outer-confirmation capability in
  `bounded_data.py`; the original development reader is not broadened.
- `response.py`, `residuals.py`: unchanged runtime generator and joint residual
  backend reused by new portable recipe wrappers in `bounded_models.py`.
- `dev_diagnostics.py`, `c_diagnostic_worker.py`: unchanged fit-only histograms,
  plotting and historical replay helpers reused; new explicitly development-only
  selection/uncertainty rules live in `bounded_metrics.py`.
- `dev_campaign.py`, `dev_worker.py`, `scripts/cms2jc2_response_dev.py` gain
  additive source/stage/CLI dispatch; no existing scientific worker changes.
- `tests/test_cms2jc2_frozen_c.py` and `tests/test_cms2jc2_b_tracking.py` supply
  the tiny ROOT and actual B/C fitting fixtures for the new staged tests.

No Fresh_check imports, external donor code, model weights or raw-data copies.
No old execution is edited or adopted as a newly fitted model. New artifacts
use BOUNDED_* v1, with an explicit recipe and old-model parents. No new license
obligation. Independent confirmation is separately authorized, not production
qualification or automatic JC2 transfer.

## 2026-09-25: frozen C topology debug replacement

Repository-local donor: `0bed4cbb598f0476b1c00d08f43859c7e46120fa`.
New `cms2jc2_response/c_topology_debug.py` adapts `b_tracking_debug.py`'s
authenticated acceptance reuse, fresh five-job registration and exact pending
retirement lifecycle. C-specific acceptance fields, original six-view worker,
evaluation association recomputation and DEV_C_REUSE lineage are retained.
CLI/source/stage/submission/worker dispatch is additive; original B and C
scientific files remain byte-identical. New tests adapt the B migration fixture
to real tiny ROOT C acceptance and evaluation. No external code, dataset,
weights or output artifacts are migrated; no new license obligation.

Donor checkout-byte SHA-256 (Git may normalize line endings), under
`src/hlt_classification/cms2jc2_response/`:

| Reused donor | SHA-256 |
| --- | --- |
| `b_tracking_debug.py` | `aced8c3dab0430015e65d9bfd281f2de7001fbd6d57bfa0dcd977425072d45ad` |
| `c_topology.py` | `462e5b3bee90b9da0adc6be679e2f692b4f2be782ab63ce6e7c722ba87bd22d8` |
| `c_topology_worker.py` | `65103052a33ab245981de253f8e2cf9ab54b2912ddbfc596edd677984cff0d74` |

Ten focused migration tests pass, including exact worker replay/population,
old-file and unrelated-B preservation, corrupt donor rejection, admission before
cancellation, state races and accounting lag. Full evidence is in HANDOFF.

## 2026-09-25: frozen B debug replacement

Repository-local donor: `56ab24cba04b81456021f10e45caa5379e244256`.
New `cms2jc2_response/b_tracking_debug.py` composes the existing frozen-B
authentication, execution-only source whitelist and acknowledged scheduler
journals. It registers a distinct five-job debug stage, reuses completed B
acceptance, and adds explicitly authorized pending-only retirement evidence.
The CLI and existing campaign/submission/worker dispatch change only for this
new stage. Original B generation, metrics, worker and result reader are reused
unchanged. No external code, data, learned weights or original output artifact
was migrated; no license/attribution changes.

Donor checkout-byte SHA-256 (Git may normalize line endings), under
`src/hlt_classification/cms2jc2_response/`:

| Reused donor | SHA-256 |
| --- | --- |
| `b_tracking.py` | `46f6cf6a896c77f15bb9d9e708daa9462fd2452a2dd919947d281eb797ac7cf3` |
| `dev_restart.py` | `0798dbd1887d4fcf081b27cb03bca221443390ac127602482164826b7ed9d39b` |
| `b_tracking_worker.py` | `6a7deb61ccccdd4f5243a4f0ce576d215a5909b76571d509caa9d7729bfc7adb` |
| `b_tracking_results.py` | `3878d5cfa244c9b7410566378e1e67566c0e897aa9de4d180ea2ab416e052cdb` |

Regression evidence: `tests/test_cms2jc2_b_tracking_debug.py` exercises the
unchanged worker on tiny ROOT inputs, donor byte preservation, exact mocked
five-job submission, cancellation races and authenticated acceptance reuse.
Complete response regression results are recorded in HANDOFF.

## 2026-09-25: frozen B tracking audit

Repository-local donor: `5c18760f06fb5cb727668ce5482c2b6c35a73223`.
New `b_tracking.py` adapts the C frozen-reuse registration to B_L with a distinct
DEV_B_REUSE/v1 contract. `b_tracking_generation.py` copies the diagnostic
traversal and reuses the unchanged residual draw and physical emission codec;
only selected tracking residuals are zeroed. New event fields capture central
predictions, increments, limits and selected cells. `b_tracking_metrics.py`
extends original counters with validity-filtered tracking accounting.
`b_tracking_worker.py` adapts the bounded process/evaluation lifecycle from
`c_topology_worker.py`, removing association/oracle work. New result display
reuses authenticated products and historical replay checks. No old scientific
module, external code, dataset, trained weights or donor artifact is changed.

Donor checkout-byte SHA-256 (Git may normalize line endings):

| Donor under `src/hlt_classification/cms2jc2_response/` | SHA-256 |
| --- | --- |
| `c_diagnostic.py` | `7f02b36c4dbb2692c08a82b9d544f93c22cd4651bde97969c030f67a594fbb29` |
| `c_diagnostic_generation.py` | `59a8ca4069e6e03511421778bb724b077fdc1b9f38297632907352d0754c0050` |
| `c_diagnostic_metrics.py` | `aa467d44b02c977c70d000ddb08654e7290826cecafc0966b59838a662a46b3f` |
| `c_topology_worker.py` | `65103052a33ab245981de253f8e2cf9ab54b2912ddbfc596edd677984cff0d74` |

Fourteen focused tests pass, including actual B fitting, exact FULL replay,
unchanged kinematics/state, selective masks, validity/clipping accounting,
process parity and tiny ROOT -> immutable reports/plots. No new licensing
impact. Combined response regression: 181 passed, one unchanged long 33-fit
production integration deselected. The first registered real SPORC acceptance
is still required.

## 2026-09-25: observed-topology C diagnostic

Repository-local donor: `36a01900a73447680cdc3e28bb0b9170c265a544`.
New `cms2jc2_response/c_topology_generation.py` copies the frozen continuous
emission equations from `c_diagnostic_generation.DiagnosticGenerator.emit`;
learned topology/state draws are intentionally replaced with observed-association
state for a privileged diagnostic. `topology.calibration_records` and its codec
are reused unchanged to authenticate the registry/round trip. The new worker
reuses frozen-C inputs, metrics/accounting, plotting and CPU bounded scheduling.
Only execution dispatch/source registration and the CLI are extended. No donor
scientific kernel, raw dataset, external code, model or weights are copied.

Donor byte SHA-256 values (local checkout bytes; Git may normalize line endings):

| Donor under `src/hlt_classification/cms2jc2_response/` | SHA-256 |
| --- | --- |
| `c_diagnostic_generation.py` | `59a8ca4069e6e03511421778bb724b077fdc1b9f38297632907352d0754c0050` |
| `topology.py` | `f7c9f19a74830e7321068ccd6dad8af1c948c3fa0832a48c5a4924e483fa578a` |
| `c_diagnostic_worker.py` | `2e3e70722a42b21a7e302fd01efe74e673e9b7ee159577497202de2d308dde00` |
| `dev_parallel.py` | `7185b56220a78491c6339924ff22ac50e21dadd5f66390407a19d75de9c9c099` |
| `c_accounting_audit.py` | `891b3b7b038ed60a59e741cd07f61530ac086708299854391966b141dff23f0f` |

Parity tests cover frozen continuous generation given identical free states,
no observed-continuous-value leakage into fixed prediction, canonical codec
closure, real process execution, ordinary full-population historical replay,
and tiny ROOT data through immutable reports. The new observed-topology plan
and contract declare all intentional differences; no new licensing impact.
Local validation: 13 new focused tests passed; combined response regression
167 passed, with the unrelated long 33-fit production integration deselected.
Real SPORC acceptance remains a separately queued requirement.

## 2026-09-25: read-only frozen-C momentum accounting

Repository-local donor: `0c3682bbb9a4c57b0f632574f878c43e863faa5c`.
New `cms2jc2_response/c_accounting_audit.py` consumes the existing
`dev_diagnostics.Histograms` payload and `c_diagnostic_metrics.Counters`
schemas, reuses development receipt/registration authentication and the
historical floating-moment replay comparison. No generator traversal or
scientific kernel is copied or modified. The dev CLI adds `c-audit` only;
existing frozen-C integration fixtures exercise read-only accounting.
New pure accounting tests cover ratios, empty jets, unknown charge bounds,
split/fallback denominators, inventory-not-occupancy and corrupt sums.
Additive stdout-only DEV_C_ACCOUNTING_AUDIT/v1 does not modify the donor reports.
No external data, models, weights or new training artifacts are copied.

## 2026-09-24: frozen CMS2JC2 C diagnostic

Repository-local donor: `5f8fecc8013a2439fb725198487dc5899246ab94`.
`cms2jc2_response/c_diagnostic_generation.py` copies the traversal from
`response.Generator` and reuses `residuals.sample`, RNG, features and topology.
Production `response.py` and `residuals.py` are unchanged. FULL replay is checked
against the production generator on every evaluated jet-replica. Other variants
are explicitly diagnostic interventions, not version-compatible fitted models.
`c_diagnostic.py` reuses development campaign/restart authentication patterns;
`c_diagnostic_worker.py` reuses frozen development readers, histograms, plotting,
publication and receipts. `c_diagnostic_metrics.py` and `c_diagnostic_results.py`
add bounded counters/display. The existing dev CLI, worker and campaign dispatch
are extended without changing old task definitions. Existing submission and
Slurm worker files are reused unchanged. Tests reuse science particles and tiny
ROOT fixtures. No external donor, weights or raw files are copied into Git.
DEV_C_DIAGNOSTIC/REUSE/ACCEPTANCE/SHARD/REPORT/SUMMARY v1 contracts bind the
fresh tier3 study; completed original de9890b2 C_L remains a read-only donor.

## 2026-09-24: CMS2JC2 CPU36 tier3 execution profile

Repository-local donor: `0560c6db42a43fc5ddca22ef7bbf5407b64fa933`.
Reuse `cms2jc2_response/dev_parallel.py`, `dev_restart.py`, `dev_campaign.py`,
`dev_worker.py` and `scripts/cms2jc2_response_dev.py` from the CPU64 implementation.
Generalize only registration/dispatch to a separate 36-CPU tier3 profile;
retain original CPU64 artifact definitions and the legacy 16-CPU path.
Tests extend `test_cms2jc2_response_cpu64.py`, including its original development
and science fixtures. No external donor, raw data, models or weights copied.
New DEV_STAGE36/DEV_CPU36_EXECUTION/DEV_CPU36_REUSE v1 contracts bind fresh
execution. Exact read-only confirmation reuse remains from the original
`de9890b22c7506c58172f20a4ab71e1b85ba5a18` development study; no scientific
matching, sampling, response or diagnostic definitions changed.

## 2026-09-24: CMS2JC2 CPU64 chunked preparation

Repository-local implementation baseline:
`a43c6ef7a00a3d0cc470a4f2be66ea1f5ce3f6f9`. Original development donor pin:
`de9890b22c7506c58172f20a4ab71e1b85ba5a18`.
New `cms2jc2_response/dev_parallel.py` reuses `dev_data.sample_stream`,
`response.collect`, `records.Reservoir`, and the process-pool pattern from
`dev_worker.py`; `combine_records` is extracted from that worker without
changing record semantics. New `dev_restart.py` reuses `dev_campaign.py`
source/import/receipt validation and its existing task/submission machinery.
No external donor code, models, or weights are copied. The matcher, native
readers, features, sampling rules, models and diagnostic definitions remain
unchanged; all non-execution source hashes are checked on confirmation reuse.

Tests reuse the development population fixture and science particle fixtures,
plus the existing real-ROOT end-to-end fit/evaluation test. Additive
DEV_STAGE64, DEV_CPU64_EXECUTION and DEV_CPU64_REUSE v1 contracts distinguish
the opt-in 64-CPU comparison; legacy DEV_STAGE v1 remains unchanged. The old
SEARCH confirmation and preparation products are read-only metadata donors,
not imported fit results or authority to modify their Slurm jobs.

## 2026-09-23: separate K2 100k/50k short-ladder debug pilot

Repository-local baseline/donor: `c891da0d45dd3251dea9ea72df975bb96bae3570`.
Reuse K2 campaign/source/submission/execution/runtime/parity and thin queue
interfaces. New `concat_k2_pilot.py` owns the opt-in recipe; new
`concat_k2_pilot_data.py` adapts `concat_k2_data.py` RAM construction and
`split_registry.py` quota/mask containers to authenticated parent-row subsets.
Matching/data/view producer files themselves are unchanged. Original compact
matching donor remains `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
`salience_learned_training.py` gains an exact opt-in pilot schedule with no
changes to default callers. Fixtures reuse `test_jetclass2_delphes.py`,
`test_jetclass2_concat_k2.py`, `test_jetclass2_concat_k2_source.py`,
`test_jetclass2_concat_k2_memory.py`, `test_jetclass2_concat_k2_reuse.py`,
`test_jetclass2_dzfix_fusion_chain.py`.
No external donor code or checkpoints copied. Pilot launch/campaign v8,
training report v3, GPU acceptance v7, population v1 isolate the smaller
study. Existing v7 full-size registration and other campaigns remain unchanged.

## 2026-09-23: CMS2JC2 isolated CPU development study

Repository-local baseline: `c891da0d45dd3251dea9ea72df975bb96bae3570`.
No external donor code copied. New `cms2jc2_response/dev_{data,campaign,
submission,worker,diagnostics}.py` reuse the namespace's `audit.py`, `splits.py`,
`readers.py`, `bridge.py`, `association.py`, `features.py`, `records.py`,
`parallel.py`, `families.py`, `response.py`, `metrics.py`, `evaluation.py`,
`measurement.py`, `provenance.py`, `storage.py`, `contracts.py`, `assumptions.py`
and underlying keyed RNG, topology and residual modules. Submission journaling
adapts `submission.py`; the CPU worker adapts
`sbatch/run_cms2jc2_response_cpu.sh`. Tiny ROOT test fixtures are reused from
`tests/test_cms2jc2_response_end_to_end.py` and particle fixtures from
`tests/test_cms2jc2_response_science.py`.

Read-only RC metadata donor is the completed preparation at
`f966dd804ed9ca8ca93c9c3227d1a7f036ef4434`, not its models or old JC2 inputs.
CMS audit/split/schema/label semantic hashes must still match before reuse.
New DEV_* v1 contracts isolate smaller exploratory samples/diagnostics from
production gates. No classifier, matching, response-family or old queue code
is changed. Existing jobs remain source-pinned to their original worktrees.

## 2026-09-23: K2 tier3 long-walltime execution policy

Repository-local baseline: `20358f5568a2a5d42c8849ca99439bc435122f26`.
Execution evidence: batch-128 source `11ff05dff72224ff365e3873c2731d79a802780d`,
SPORC preflight 21770762 (memory probes passed; 72.91h projection exceeded
23h guard). Reuse `concat_k2_campaign.py`, `concat_k2_execution.py`,
`concat_k2_runtime.py`, `concat_k2_submit.py`, and the existing CLI/queue helper;
no external source is copied. Adapt existing K2 registration/source/reuse tests
and add `tests/test_jetclass2_concat_k2_walltime.py` for 96h-tier3 admission,
95h acceptance, unchanged short-job portability and exact TimeLimit rejection.

Original preparation donor remains
`1f9306504dfd040c9c22e0a89829d277d1ff2194`; the importer additionally recognizes
fresh v7 donors, without accepting nested imports or historical GPU gates.
Matching/data/view producers and their source fingerprints are not changed.
Launch/campaign v7, GPU acceptance v6 and execution policy v2 distinguish the
new resource envelope. Model/training code, batch 128, scientific seeds and
other campaigns are unchanged. New real SPORC acceptance remains required.

## 2026-09-23: dzfix fusion tier3 long-fit registration

Repository-local baseline/donor: `20358f5568a2a5d42c8849ca99439bc435122f26`,
the 21770778 debug preflight. Adapted `dzfix_fusion_chain.py`,
`dzfix_fusion_runtime.py`, `dzfix_fusion_submit.py`, the fusion CLI/queue/worker
and existing fusion contract tests. Added tier3 scheduler/runtime regression
tests. Reused `execution.execution_site("sporc_a100")` without modifying shared
site code. No model, training kernel, external code or weights copied.
Launch/campaign v6 and acceptance v4 separate the tier3 72-hour registration
from debug; source-import v3 and all matching producer hashes are unchanged.
K2 and salience campaigns are not edited. See HANDOFF for test/remote evidence.

## 2026-09-23: dzfix fusion strict parity and layout-preserving offload

Failed execution baseline: `b35fbda64d2d823a9eb9c5592017074db58d6ac8`
(SPORC preflight 21765886). Repository-local donor:
`91be01940f814e1ea7a8a460b9694f0148f0d024`, specifically
`src/hlt_classification/jetclass2_delphes/concat_k2_model.py` (scoped strict
backend and physical-span saved-tensor storage), `concat_k2_parity.py`
(bounded early real-train diagnostic pattern), and
`tests/test_jetclass2_concat_k2_parity.py` (failure-before-cache tests).
Adapted into `dzfix_fusion_model.py`, new `dzfix_fusion_parity.py`, and fusion
tests. The sampler uses salience `load_assignments` / `build_view`, not K2
maps, views or batch policy. No K2/shared preparation files are changed.
No external source or scientific checkpoints were copied.

Execution contracts: fusion launch/campaign v5, acceptance v3, early parity
v1; matching source-import stays v3. Production remains batch 256. Local
installed Weaver 0.5.3 is an isolated test dependency, not vendored code.

## 2026-09-23: explicitly registered K2 physical batch 128

Repository-local baseline/donor: `91be01940f814e1ea7a8a460b9694f0148f0d024`,
the user-reported 21768860 preflight. Adapt `concat_k2_campaign.py`,
`concat_k2_model.py` probe policy, `concat_k2_runtime.py` and the K2 queue
helper; add opt-in batch overrides to `salience_learned_training.py` while
preserving legacy defaults/report layout. No training loop, Weaver source,
model weight or external code is copied. K2-only versions distinguish the
new recipe/acceptance; preparation import additionally admits fresh v6 donors.

Original matching donor remains `1f9306504dfd040c9c22e0a89829d277d1ff2194`.
Matching/data/view producer files and hashes are untouched by this block.
Tests adapt existing campaign/memory/reuse fixtures and add 300-row physical
CE/KD batch-boundary, teacher-join, final-partial-batch, legacy-default and
batch-evidence rejection tests in `test_jetclass2_concat_k2_batch128.py`.
Real SPORC full batch-128 acceptance remains required before science release.

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

## 2026-10-05: exact final-direct node-failure recovery adapter

Scientific donor: `fd1c05786287c57b080b2d229dd64664f830a571`.
New adapter: `src/hlt_classification/jetclass2_delphes/dzfix_fusion_recovery.py`,
with thin CLI/worker and queue helper. No scientific donor files are copied or
edited. The new CLI imports these original checkout modules at execution:

- `jetclass2_delphes/dzfix_fusion_chain.py`, `dzfix_fusion_runtime.py`, and
  `dzfix_fusion_submit.py`: original campaign, gate, training/publication and
  exact allocation checks;
- `jetclass2_delphes/contracts.py`, `jetclass2_delphes/submission.py`,
  `data/cache_contracts.py`, `scouting/hcwdl_exact_dag_submission.py`, and
  `scouting/hcwdl_recovery.py`: original hashing, immutable publication and
  guarded exact-DAG submission;
- `sbatch/jetclass2_delphes_common.sh`: original SPORC environment setup;
- `tests/test_jetclass2_dzfix_fusion_chain.py`: reusable local cache fixtures
  for the recovery tests (not a production import).

The new versioned recovery artifacts govern only a three-task restart-zero
execution exception; scientific contracts and original pinned source remain
unchanged. See `docs/contracts/JETCLASS2_DZFIX_FUSION_RECOVERY.md`.

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

## 2026-10-02: literature-inspired offline-only pilot

No external donor or `Fresh_check` runtime dependency. New implementation lives
under `src/hlt_classification/literature_proxy/`; it does not reuse any learned
CMS response, calibration map, or generated proxy particles. In-repository
interfaces reused from donor commit `41a59b644f1fa4398fc022e06ff88ab16f5fb5d9`:

| Donor file | Local SHA-256 at inspection | Retained semantics |
| --- | --- | --- |
| `cms2jc2_response/bridge.py` | `c3a3e9e810996899894390f8733c5c6ac547599fecaf323a3415c137d7db9dae` | Physical `Particles`, wrapped phi, physical p4 reconstruction; no fitted response |
| `data/cache_contracts.py` | `004aef8e8bbdf6ce513479024d4a119f92277fca3e359f977a5f4022d8e0d7dc` | Canonical hashes, deterministic NPZ, immutable atomic publication |
| `jetclass2_delphes/split_registry.py` | `fd2246185b042de3e255c5073ccd75c47d687a9428139d22cb6fac32aeace911` | Existing authenticated train membership, without resplitting |
| `jetclass2_delphes/inventory.py` | `998323e064d10916268f0ab2e4ea833d2cd7a568eafde259611d92ce6b98e605` | Per-source ROOT byte/schema/cycle authentication |

Paths above are relative to `src/hlt_classification/`. The new offline adapter
retains the bridge's explicit unknown PID and unavailable-error policy, but
uses native JetClass2 GeV/mm/sign/PV semantics directly instead of importing a
CMS/JC2 fitted compatibility review. The new reader deliberately does not use
`DatasetReader`, which also reads HLT arrays. No third-party source was copied;
literature citations motivate mechanisms, not measured parameter values.
Focused replay, mask, physics-invariant, native ROOT, and artifact tests reside
in `tests/test_literature_proxy*.py`; execution evidence is in `docs/HANDOFF.md`.
