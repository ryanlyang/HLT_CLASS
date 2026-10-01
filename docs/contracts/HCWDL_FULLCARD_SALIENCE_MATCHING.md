# HCWDL full-cardinality salience matching contracts

The `HCWDL_FULLCARD_SALIENCE_* /v1` and
`HCWDL_TRI100_FOUR_SPINE_SALIENCE_PERSISTENT_HLT_* /v1` contract families
define an isolated forced-alignment study. They do not change or authorize
reuse of any high-coverage or full-cardinality bottleneck artifact.

## Matching boundary

For every nonempty jet, an assignment contains exactly
`min(n_hlt, n_offline)` one-to-one pairs over the complete bipartite graph of
valid HLT and offline particles. Thus every particle on the smaller side is
paired. The durable orientation is one native offline index per HLT particle,
or `-1` only for the unavoidable excess HLT particles. A pair is a forced
training coordinate, not a detector-level correspondence claim.

The primary objective maximizes total integer pair utility: the sum of the
two endpoint saliences multiplied by bounded angular closeness. Delta-R is
canonicalized in float64 with half-open phi wrapping, quantized at `1e-7`, and
clipped at `0.8` only in the primary closeness term. The solver then applies
the registered integer-exact secondary objectives in order: uncapped delta-R,
larger-side selected salience, absolute log-pT response, raw category
mismatch, valid-charge mismatch, and the native-index deterministic tie-break.
The production Hungarian result must equal exhaustive enumeration for every
registered reference case with side length at most eight.

Exactly three salience definitions are screen-eligible:

- `SALIENCE_PT_LINEAR`;
- `SALIENCE_PT_QUADRATIC`;
- `SALIENCE_PT_QUADRATIC_CORE25`.

All use positive integer pT-share salience with a nonzero floor. The third may
add only the registered at-most-25% own-axis core bonus. Labels, gradients,
attention, impact parameters, and learned correspondence confidence cannot
enter the matching score. The old bottleneck matcher is a contextual control,
not a fourth selectable candidate.

## Candidate foundations and selection firewall

Each candidate has its own immutable matcher specification, compact assignment
shards/manifests, all-row train/validation diagnostic reports, assignment and
coupling locks, and rebuilt assignment-dependent descendants. Dense cost
matrices, particle views, ROOT copies, optimizer state, and rolling-resume
state are RAM-only or absent. The inherited pure-offline U000 checkpoint and
probability bank are read-only equivalence references and are not retrained by
candidate-foundation construction.

The screen owns a deterministic class-and-identity validation partition:
`V_checkpoint` is visible for checkpoint selection and `V_select` is evaluated
exactly once after restoring each checkpoint. Four seed-, data-, view-,
schedule-, and architecture-matched CE-only U100 fits run: the bottleneck
reference and all three candidates. All three candidate rows are reported.
Selection first maximizes V_select macro one-vs-rest AUC, treats candidates
within `5e-5` as tied, then uses macro mean log R50, accuracy, pT-weighted
delta-R, and frozen registry order. Bootstrap intervals are descriptive and
cannot control selection. Poor metrics never fail the screen.

The immutable selection lock binds the screen report, selected candidate,
selected foundation, selected matcher specification, frozen three-candidate
multiplicity, and all four fit reports. A production source lock reopens and
authenticates that report and lock; path existence or a candidate name alone
is insufficient authority.

## Winner-only four-spine campaign

The selected foundation feeds a new persistent-HLT-support campaign with one
fresh hybrid U000 anchor and the same DIRECT, COARSE, DENSE, and ULTRADENSE
path geometry used by the controlled four-spine comparison: 30 fresh fits and
26 single-model probability reducers. Every downstream edge uses 25% CE and
75% temperature-2 probability KD from its immediate parent. There are no
ensembles, M1 compression, weight continuation, DDP, or cross-spine teachers.

Training is single-node, single-task, single-GH200 with effective batch size
256 and the registered 100-pass floor-tail schedule, minimum 60 passes,
patience 15, and restoration of the best checkpoint. The selected candidate,
three-candidate multiplicity, and selection-lock lineage remain explicit in
campaign, support, execution, reducer, aggregate, and completion artifacts;
fit reports bind the same selection lock directly in their parent registry.

Only compact 15-class train/validation probability banks and ordinary model
artifacts are durable. No particle-view, dense matching-matrix, hidden-state,
optimizer, or rolling-resume cache is durable. Recovery is exact-ledger and
restart-from-zero. Existing campaigns have no scheduler dependency on this
study and no output path that it may mutate.

## Access and failure semantics

Ordinary execution can access authenticated train and validation roles only.
Final-test assignment, view construction, inference, and metrics remain
sealed. Nonfinite required values, incomplete smaller-side coverage, solver
disagreement, corrupt bytes, stale source, unregistered candidates, lineage
mismatch, forbidden storage, or invalid production topology fail closed.
Scientifically weak accuracy, AUC, rejection, calibration, or matching-tail
quality is a reportable result and cannot cancel later registered work.

Live production remains gated on the exact pushed commit, clean detached
worktrees, candidate matcher/resource acceptance, installed-Weaver parity,
genuine Tigris preflight, immutable selection, and explicit submission
authorization.
