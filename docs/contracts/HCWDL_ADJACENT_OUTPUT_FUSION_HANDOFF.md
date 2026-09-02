# HCWDL Adjacent Output-Fusion Handoff Contract

This contract governs Strategy A of the implementation-authoritative
adjacent-view handoff plan. Its artifact family is
`HCWDL_ADJACENT_OUTPUT_FUSION_HANDOFF_*/v2`.

## Scientific graph

The read-only source carrier is the explicitly selected, authenticated
non-persistent full-cardinality `SP4_COARSE_U100_from_U050` model (production
job `98318`). It uses the established `replace_source_with_target_v1` support
semantics; the later persistent-HLT `SP4P` lineage is a distinct experiment
and is forbidden as a substitute. The scheduler ID is an operational locator;
the source spec, report, checkpoint, graph, recipe, foundation, and assignment
content hashes form the scientific identity. The sequential path is
`U100 -> D080 -> D060 -> D040 -> D020 -> D000`. At each transition a cold
lower-view direct model is trained with C25/P75, T=2 LOGIT KD. Two output
mixture families then combine the richer carrier and lower model over the
fixed alpha grid `0/40 ... 40/40`. A cold lower-view compression model learns
from the selected mixture. Only the compression model carries science to the
next transition.

The terminal transition has five paired direct/compression seeds and five
matched CE-only seeds. Fixed lexical prefix ensembles E1 through E5 are
published for all three families. No favorable subset search is allowed.
Three identically initialized final D000 models use C10/P90, T=1 KD from the
CE E5, direct E5, and handoff E5 distributions. The campaign therefore owns
exactly 26 fresh fits, 9 selections, and 15 prefix ensembles.

## Selection boundary

Validation is deterministically class-stratified and disjointly partitioned
into `V_checkpoint`, `V_blend`, and untouched `V_report`. Checkpoint selection
and early stopping use only `V_checkpoint`. Temperature calibration, alpha
selection, and 2,000 paired class-stratified Gaussian-multiplier bootstrap
replicates over exact paired AUC influences use only `V_blend`. Every candidate
uses the same registered multiplier stream. Primary model, mixture, and
ensemble metrics use only `V_report`.
The imported `M0CE60` and pure-offline `U000` checkpoints are also reevaluated
on those exact `V_report` identities; full-validation metrics copied from their
source reports never serve as recovery denominators.

A candidate is feasible only when its macro-AUC point difference from the
richer carrier is at least -0.0001, its one-sided 95% bootstrap lower bound is
at least -0.0003, and its Xbb/Xcc/Xqq QCD rejection ratios at 50% signal
efficiency are each at least 0.95. Nonfinite required quantities fail closed.
The selector chooses maximum poorer-view alpha, then macro AUC, macro R50,
calibrated-logit family, and lexical identity. Alpha zero is always evaluated,
recorded, and provides the scientific fallback. Poor performance never prunes
the registered graph.

## Training and execution

Every fit is cold-started, uses batch size 256 on one GH200, and has the
registered 100-pass ceiling: three-pass warmup, 3e-4 hold through pass 45,
cosine decay through pass 60 to 1.5e-5, then constant floor refinement.
Minimum passes are 60, patience is 15, and the exact best checkpoint is
restored. Direct and compression models within a rung share stochastic seed
domains; different rungs do not. Terminal CE/direct/compression triplets are
paired per S1 through S5. The final three distillers share one seed.

The source campaign has no scheduler edge and cannot be mutated, cancelled,
held, or reprioritized. Overall source completion is unnecessary when the
explicit U100 report and selected checkpoint authenticate independently.
Deployable endpoints consume D000 HLT inputs only. Final test is unavailable.

## Storage and recovery

Particle views exist only in process RAM. Hidden states, optimizer states,
and rolling-resume checkpoints are not durable. Reusable probability banks
contain uint8 identity digests and one canonical float32 T=1 15-class
probability table per role. The train manifest registers the consumer
temperature; T=2 or T=1 KD targets are derived exactly once in RAM at identity
join time. This halves train-bank storage and prevents mixing already softened
targets as if they were model outputs.

The gate preflight reads a genuine validation row through the authenticated
non-persistent full-cardinality U100 stream and executes the installed production Scouting
Particle Transformer through a C25/P75, T=2 forward/backward update on one
Tigris GH200. Full science submission requires that immutable acceptance lock.

Tasks publish atomic content-hashed artifacts and exact output attestations.
Submission is journaled and exact-ID bound. Recovery may be created only from
a terminal monitor, inherits only content-valid completed tasks, retries the
failed downstream closure from update zero, and cannot change scientific
semantics. Final-test access remains false in every artifact.
