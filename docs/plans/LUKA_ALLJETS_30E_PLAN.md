# Matched trial ALLJETS, fixed thirty epochs

Authorized 2026-10-16. This is a NEW exploratory experiment, not admission of
the old 60--100-pass experiment. It supersedes that schedule only for execution
v6. Keep 299,234 TRAIN / 74,895 validation, original50k secondary reporting,
all seven views, seeds, true batch128 / accumulation1 / inference128, AdamW,
peak3e-4 / floor1.5e-5, C25/P75 T2, model and synchronous offload unchanged.
All seventeen controls/direct/regular-coarse/fusion/both-ending fits are fresh.
Eleven TRAIN banks plus aggregate/complete give thirty science jobs.

Every fit runs exactly30 full epochs, three warmup epochs then cosine decay
through30, no plateau or early stopping. Select the best checkpoint by the
existing full-validation ordering. Never select on original50k or test. Former
test is already retired into TRAIN; there is NO independent test set. Equal
per-fit budgets do not imply equal total direct/ladder compute or convergence.

Reuse only authenticated technical resource evidence from job21898881/source
922bd972ef9ec52079e9e750e6e7896e857aba8b. Original-source read-only replay still
authenticates all twelve probes, exact failure, accounting, source, payloads and
cache lineage. The old candidate retains its original recipe and failed time
admission. A separate technical validator checks ALL parity/support/measurement/
memory/payload requirements, including the correctness of the old time estimate,
without requiring the OLD schedule to fit. No candidate fields are rewritten.

New budget: max-family ceil(1.75*(cache + 30*full-pass runtime + selected-model
inference/readback)/60), minimum30min. Reducer unchanged: max-family
ceil(2*(cache+inference)/60), minimum15min. Runtime already includes validation.
Keep85% host /90%GPU headroom and1440min debug ceiling. Reject excessive estimates
during dry creation, before any job. Reuse measures operations and memory, not
thirty-epoch convergence or a new GPU run; disclose this distinction.

One new CPU-only admission job (1CPU/32GiB/4h/debug) publishes the v6 receipt.
Science requires that job COMPLETED/0:0, exact pushed source and a separately
reviewed thirty-job plan. No auto retry/followup, old checkpoint/bank import,
raw-data preparation, source/artifact overwrite or real submission by this task.

Source compatibility explicitly permits the named v6 recipe/dispatch/admission
changes. Concurrent FullStat modules must be byte-identical to commit
592a6dace8adc662903d3da0d409bcab5fefb9a7; they are not trial inputs. Original
models, data/cache/split, probe producers and batch128 recipe remain identical.
The FullStat20 and legacy schedules remain unchanged. Local tests must cover
actual thirty-pass execution, both paths, all17 fits/11 banks, strict old/new
admission, corruption and version isolation. Existing real A100 evidence is
not new thirty-epoch acceptance; local small Weaver checks cannot establish
full-support A100 resources.
