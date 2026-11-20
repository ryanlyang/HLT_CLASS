# Six-class Hbb reselection contracts v1

Authority: [analysis plan](../plans/LUKA_SIXCLASS_HBB_RESELECTION_PLAN.md).
Implementation: `operations/sixclass_hbb_reselection.py`, loaded by the thin
`scripts/reselect_luka_sixclass_hbb.py` adapter against the original source.

`LUKA_SIXCLASS_HBB_RESELECTION_SPEC/v1` binds the original execution hash,
original and evaluator clean source snapshots, installed runtime fingerprint,
canonical validation identity digest, exact class counts, and original results.
Policy is explicit full-validation binary Hbb/QCD AUC, 20 epochs/model,
maximum AUC then earliest exact tie, exploratory selection, final test sealed.

`EPOCH/v1` binds the analysis hash and original immutable epoch-commit hash.
It records stage/epoch, binary AUC, QCD/Hbb counts, R50/R30 thresholds and passing
counts, runtime job and duration, and original in-range metric reproduction.
The original selected epoch must additionally reproduce published probability
payloads for both validation regions exactly. Other signal classes are omitted
only from binary metrics, not from original inference batches. No threshold or
epoch is chosen using final test. Undefined pair scores/nonfinite probabilities
fail; zero passing QCD is recorded as censored rejection, not infinity.

`REPORT/v1` binds the analysis and contains all 60 epoch receipts, original and
new selections, and their separately recomputed Hbb-AUC recoveries. Selection
requires all 20 epochs for each model. A nonpositive OFFLINE-minus-M0HLT gap is
unresolved. Report publication is immutable and completion-last.

On resume, source/runtime/population/original artifacts are authenticated again.
Existing epoch receipt hashes/parents are checked; original-selection prediction
replay is repeated. A Linux advisory lock prevents concurrent evaluation and is
released by process exit. The queue entrypoint refuses duplicate initial
submission; explicit `resume` requires the prior exact job to be terminal and
unsuccessful and the report to be absent. Ambiguous submissions stop for operator
inspection. No existing science claims, jobs, states, or reports are changed.

The requested 48 hours is an allocation ceiling, not a measured completion
estimate. Local installed-Weaver checks do not establish Oscar GPU replay;
production parity happens inside the validation job, without a preflight fit.
