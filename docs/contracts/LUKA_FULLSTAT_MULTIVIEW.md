# FullStat multiview contracts v1

Authority: [active scientific plan](../plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md).
Implementation: `hlt_classification.luka_fullstat_multiview`. These contracts
are distinct from small-data SPORC multiview and Oscar retained-predictor runs.

Every `LUKA_FULLSTAT_MULTIVIEW_*/v1` object has canonical content hash, parents,
schema version and FullStat development-validation flags. Test access remains
false and sealed; event independence and old-trial overlap remain unverified.

- RECIPE fixes 20 epochs, batch128, original FullStat optimizer/schedule,
  CE1, total embedding10 split exactly 10/3 each. No inference-time predictors.
- REUSE authenticates corrected readout db9afe8b in its own source, completed
  U100-w5 and original COARSE_U100 TRAIN bank, original OFFLINE node, references,
  prepared-cache identity and original Oscar environment. Original/current source
  byte inventories are checked independently; four later SPORC compatibility
  fixes are pinned to baseline 74cba41f, never silently treated as identical.
- SPEC fixes exact pushed additive source, immutable root, counts, all nodes,
  teacher definitions, resources and available output quota. Existing roots
  cannot be repurposed. Three jobs, no setup or separate preflight jobs.
- PLAN binds exact three commands. FUSION is ordered afterany CONCAT; student
  requires afterok CONCAT and FUSION. Claims and journals prohibit silent retry
  after partial submission. Workers verify exact job/allocation/environment.
- STARTUP binds actual full-support populations, Weaver/native deployment
  parity, finite gradients, stress memory and measured training, validation and
  teacher export throughput. Probes are discarded before scientific training.
  Historical acceptance does not certify these new graphs.
- EPOCH binds the report row to full model, optimizer and RNG state, atomically
  saved after every validation pass. REPORT records all 20 epochs and selects
  by full-validation AUC with existing tie-breaks. No performance gate exists.
- BANK is TRAIN-only, selected checkpoint/report/prepared/spec bound, width128,
  FP32 export at batch128. Payload contains ordered identities/labels, targets,
  independent FP64 TRAIN moments and scale with .001 floor. No class logits or
  validation embeddings are supervision. New banks must exactly join native
  HLT TRAIN rows. Original U100 bank is immutable and uses its existing contract.
- TASK authenticates all published outputs, epoch states and selected-state
  readback; teacher completion includes its bank. A path alone proves nothing.
- RESULTS uses the same registered validation rows and AUC-selected checkpoints
  for accuracy/AUC/R50/R30; fixed M0HLT/OFFLINE anchors define recovery. Teacher
  rows are privileged. Historical U100-w5 is not a fresh matched control.

CONCAT retains the entire physical HLT/OFFLINE multiset with shared conversion
and source-blind ordering, including duplicates. It has no source identifiers,
positional tags or per-source normalization. Natural feature differences can
still reveal origin. FUSION preserves the existing native two-stream model,
HLT-query/OFFLINE-context direction, pair geometry and padding semantics.
Deployable student uses only native HLT and its native classifier.
