# FullStat three-teacher auxiliary embedding study on Oscar

Active plan, revised 2026-11-03. User authorized two fresh privileged teachers and one
fresh HLT-only student. This adapts the SPORC multiview scientific graph to
3,000,000 TRAIN / 500,000 validation, 20 epochs, batch128 and the FullStat20
optimizer/schedule. The final 1,500,000 test and 105,325 unused rows stay sealed.
The approved paired, stratified jet-row split is unchanged; event independence
and overlap with the older trial remain unverified.

## Scientific graph

- Reuse the original regular COARSE_U100 behind completed FullStat U100-w5,
  authenticated through corrected readout source db9afe8b. Do not use hybrid
  UE100, a small-data teacher, or the retained-predictor checkpoint.
- CONCAT: CE-only fresh native ParT on the full physical multiset union of HLT
  and OFFLINE particles. Apply the existing common physical conversion,
  source-blind ordering and 17-feature transform once to the union. Keep both
  measurements, without matching, deduplication, source tags, positional/source
  embeddings or source-specific normalization. This is an oracle multiset,
  not a physical reconstructed jet. Different measurements can reveal origin
  statistically. Never crop the union to the old single-view capacity.
- FUSION: reuse the existing two-stream asymmetric ParT: native HLT queries
  attend to native OFFLINE keys/values after blocks 2,4,6,8. Preserve the cross
  geometry, zero residual projections, learned gates, full pair population and
  padding semantics. Both active streams train from scratch with CE only. Its
  target is the 128-dimensional final enriched HLT representation.
- Student: exact native HLT inputs, fresh original U100-w5 backbone/sampler
  seeds; three independent 128-GELU-128 prediction heads and native classifier.
  CE coefficient 1, total embedding coefficient 10, each teacher 10/3:
  `L = CE + (10/3)*(E_U100 + E_CONCAT + E_FUSION)`. Each E is SmoothL1(beta1),
  mean over rows and coordinates. Each teacher uses its own TRAIN-only FP64
  moments, scale floor .001 and frozen FP32 targets. Predictions are never
  classifier inputs. All auxiliary heads are removed at deployment.

No retained predictor, logit KD, readout, relational loss, warm start, extra
student/control, teacher-coordinate averaging or automatic hyperparameter
selection. Total10 is an experimental choice, not an established optimum.
Historical U100-w5/M0HLT/OFFLINE are references, not a new matched weight5 arm.

## Execution and artifacts

Exactly three scientific jobs: teacher_CONCAT, teacher_FUSION, train_multi3.
Teacher FUSION follows CONCAT with afterany for resource ordering only; the
student requires afterok on both complete teacher jobs, including TRAIN exports.
Poor finite metrics do not fail a teacher or suppress the student. Technical
input, integrity, resource and nonfinite failures fail closed.

No separate preflight. New-graph technical checks, installed-Weaver parity,
full-support batch128 stress, real bounded training/validation/export timing,
memory admission and strict deployment readback occur inside each job before
fresh scientific initialization. Historical acceptance is not claimed to cover
the new privileged graphs. Keep 20 full scientific epochs, including partial
batches; no silent smaller batch, crop, accumulation, schedule or model fallback.

Oscar L40S, 6 CPUs, 192000 MiB, 48h for each teacher and 24h for the student.
The exact resource shapes must pass Oscar's scheduler dry review before submission.
Reserve 20 GiB outputs plus 2 GiB
filesystem headroom and require declared available personal quota >=22 GiB.
CONCAT is reconstructed in bounded RAM from authenticated TRAIN/validation raw
rows using four workers. No persistent new particle cache or historical writes.
U100 targets are reused read-only; new teacher banks are published inside their
new task roots. Save every epoch's complete model/optimizer/RNG state before
publication so a late failure does not discard the fit. No automatic retry.

SPEC and STARTUP v2 distinguish the resource repair from the failed v1
execution at 2b1f0370. The scientific RECIPE remains v1: inputs, losses, seeds,
20 epochs and batch128 are unchanged. FUSION reuses the existing synchronous,
stride-preserving pair saved-tensor host storage and compact cross-mask adapter.
Full pair populations and BatchNorm statistics remain intact; there is no
recomputation or microbatching. Inline checks verify native-mask equivalence
and three-update FP32/BF16 storage parity, including gradients, buffers, updated
weights and optimizer states. Keep the existing 90% GPU and 85% host-memory
limits (including 8 GiB host reserve). Persist timing and memory measurements
before admission, and stage/failure diagnostics even when a guard rejects them.

Other `LUKA_FULLSTAT_MULTIVIEW_*/v1` contracts and exact pushed-source closure
distinguish this from small-data SPORC and the retained experiment. A detached
launcher authenticates the completed parent, saves a full dry review, and then
submits only the authorized three jobs. It never cancels current jobs. Partial
submission retains claims/journals for inspection.

Each fit selects on its own full-validation AUC with the existing tie-breaks.
Report AUC/accuracy/R50/R30 using the same selected checkpoints and fixed
M0HLT/OFFLINE recovery anchors; label both new teachers privileged. Local
synthetic/installed-Weaver checks do not constitute new real Oscar acceptance.

The original CONCAT job8148495 spent 4200.94 seconds loading the union and then
failed the 24h timing guard. FUSION job8148496 reached 43,110,401,536 peak CUDA
allocated bytes and failed the memory-margin guard. Neither saved a scientific
epoch. Preserve their root and source; launch corrected code into a fresh root.
The failed jobs did not persist the union or fitted states, so their RAM-only
preparation must run again. No new standalone preflight is introduced.

## Recover the completed-CONCAT execution

User-supplied accounting and diagnostics for source `8ad0b381` show CONCAT
8177360 COMPLETED/0:0, including twenty epochs and its published TRAIN bank.
FUSION 8177361 passed native/storage parity, full-support training, validation
and export stress at 10.59 GiB CUDA peak, then rejected the timing envelope:
4164 conservative minutes versus 2880 allocated. No scientific FUSION epoch
or accepted startup exists. Student 8177362 was cancelled without running.

The specific recovery reuses completed CONCAT in its original root, validated
in original source including all epoch/checkpoint/prediction/bank payloads.
Queue only a fresh FUSION fit requesting 4320 minutes (72h), followed by the
unchanged 1440-minute student afterok on FUSION. The exact new request must pass
Oscar scheduler dry review. The 1.5 timing factor, 120-minute allowance, memory
limits, integrated checks and all scientific semantics remain unchanged. The
69h24m estimate is extrapolated admission evidence, not measured full-fit time.

Use a fresh root and RECOVERY_SPEC/RECOVERY_POLICY/RECOVERY_REUSE v1; preserve
all original spec, startup, bank and receipt hashes. Source admission freezes
scientific kernels and external dependencies, allowing only declared operational
recovery adapters relative to `810433f8`. Never reparent or copy the old bank
as a new export. A single atomic claim in the original root prevents competing
recoveries. The detached launcher authorizes exactly two jobs, retains failures
and partial submission journals, and performs no automatic retry or cancellation.
No separate preflight, new control, altered batch, shortened fit or relaxed
resource margin is introduced. Local tests do not certify Oscar's 72h policy
or a completed full-size FUSION run.

## Recover the finished FUSION fit after export crash

Job8220911 at `a1eb1a746875a2697216d45a7c8ea4b6acdabc8a` ran all twenty
scientific epochs and selected epoch14 (validation AUC0.9705942285340523).
Its selected deployment readback completed before signal7 terminated the job
at the start of TRAIN embedding export. Elapsed42h15m40s was below the72h
allocation. The log does not establish the SIGBUS cause. Job8220912 was
cancelled without starting; no FUSION bank or completed task receipt exists.

The authorized recovery is **one fresh-process export and one student fit**.
No teacher training, validation inference or new checkpoint selection occurs
in the export. Authenticate the original spec/source, scheduler identities,
all twenty epoch receipts/states, selected full/deployed weights, report and
saved validation predictions. Preserve these files, the failed task status,
completed CONCAT and all historical sources. Do not fabricate an original
completion receipt. A new export receipt records the failed fit's provenance
and the successful recovery job separately.

Use the original a1eb1a74 Python scientific dependencies, with only the additive
`luka_fullstat_multiview_export` operations loaded from the new pinned source.
Load TRAIN HLT/OFFLINE views only, verify their fingerprint against original
startup, reload selected FUSION weights, and run bounded first/max-support
FP32 inference and timing checks within the export allocation. The original
batch128 FP32 exporter and TRAIN normalization remain unchanged. Phase records
and Python fault-handler traces make an export failure observable; a successful
fresh process is not assumed to prove the underlying SIGBUS cause was fixed.

Request6h for export and24h for the unchanged student, each L40S/6CPU/192000MiB.
Use the unchanged1.5 timing factor,120-minute fixed allowance and memory limits.
The student's original inline checks remain; there is no standalone preflight.
The exact allocations pass scheduler dry review before submission. Student
starts afterok on export and verifies original CONCAT and U100 plus the new
FUSION bank before fitting. Science remains CE1 + total embedding10,20epochs,
batch128, original seeds and HLT-only deployment. These allocations are upper
limits, not promised runtimes.

Use fresh `LUKA_FULLSTAT_MULTIVIEW_EXPORT_*/v1` execution artifacts and a fresh
output directory, with the original BANK/RECIPE formats unchanged. Reserve
20GiB outputs plus2GiB headroom. A new original-root exclusive claim prevents
competing recoveries; other original artifacts are read-only. Retain partial
outputs and submission journals on failure, with no automatic retry. The
[export recovery runbook](../LUKA_FULLSTAT_MULTIVIEW_EXPORT_RUNBOOK.md) governs
this incident; the older timing-only recovery must not be relaunched.
