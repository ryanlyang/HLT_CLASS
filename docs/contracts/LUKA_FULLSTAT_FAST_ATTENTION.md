# FullStat fast-attention execution contracts v1

Authority: [operational plan](../plans/LUKA_FULLSTAT_FAST_ATTENTION_PLAN.md).
All artifacts carry canonical content hashes, explicit parents and unchanged
development-validation/sealed-test flags. Contract prefix:
`LUKA_FULLSTAT_FAST_ATTENTION_`, version1.

| Artifact | Binding and validation |
| --- | --- |
| REUSE | Original62c9 science spec; original-source gate/preflight/export replay; original bank files authenticated; no requirement for the cancelled duplicate logit task |
| GATE | Fresh root; exact clean additive source snapshot; reuse hash; immutable optimization policy and declared personally available result quota |
| PREFLIGHT | Gate hash; exact registered Slurm job; installed environment/GPU; real-bank timings and parity; host/GPU envelope; conservative train walltime |
| SPEC | Accepted gate and preflight; attention-only fit plus dependent aggregate; no export/logit/test task |
| EPOCH | Same gate, bank and source; immutable `.pt` hash; epoch/update/history consistency; receipt published after payload |
| RESUME | Same science spec; explicit checksum-bound epoch receipt within that execution's attempts; fresh recovery ledger; prior registered jobs terminal |
| FIT_EXECUTION | Original scientific report's execution provenance, native deployment parity, epoch-store path, explicit optional resume receipt |
| TASK | Exact science spec/source/task; selected checkpoint/report/predictions/execution artifacts; validation population and metrics reverified |
| RESULTS | All required completed outputs; original references labeled; ATTN minus original COARSE_D000; no test-selected winner |
| PLAN / CLAIM | Exact reviewed Slurm argv, durable journal, explicit authorization, no ambiguous duplicate retry |

The **scientific** recipe, exported bank and training-report schema remain
`LUKA_FULLSTAT_LAST_ATTENTION_*/v1`, since their meaning is unchanged. New
execution receipts distinguish implementations; fast execution is not passed
off as a completion of the old job/spec.

## Input and memory contract

ResidentBank first runs all original BANK validations. For each shard it then
hashes the exact bytes copied into the FP32 buffer. Counts, source rows,
probabilities and attention surface do not change. Bank arrays are read-only.
The hot path performs no target filesystem open, mmap, hash or shard eviction.
Batch queries retain bounds, width and mask checks. Final-test/validation
teacher banks remain forbidden.

Resident target cap112GiB, additionally bounded by85% of180000MiB minus current
RSS minus24GiB reserve. Training uses the same allocation as measurement.
Two CPU futures plus the consumer's current padded batch are bounded; no CUDA
prefetch thread or buffer reuse. Old payloads are never rewritten.

## Checkpoint contract

Current model, AdamW, Torch CPU/CUDA RNG, Python/NumPy RNG, global update count,
history, selected-best state and selection index are stored after validation.
Resume restores state before the next epoch and reconstructs the same epoch
permutation from the unchanged sampler seed. Checkpoints from a different
source, gate or bank are forbidden. Orphan payloads without committed receipts
are never resumed. There is no partial-epoch continuation or automatic retry.

Rolling attempts use fresh job-specific paths and an OS-released exclusive
training lock. Completed selected outputs are never overwritten. A partial
final publication requires inspection, not automatic cleanup.

## Admission is technical, not scientific

Invalid inputs/lineage, nonfinite required quantities, parity failure and
insufficient resources fail closed. Accuracy/AUC/rejection values do not gate
submission or skip registered science. Timings are sample-based; full epoch
throughput has not been measured by the preflight. No epoch count or validation
population is silently shortened to meet a walltime.
