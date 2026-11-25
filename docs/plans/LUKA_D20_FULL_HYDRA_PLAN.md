# Full-population D20 RAW campaign on four GTX1080Ti GPUs

Authority: the user's 2026-11-24 full-campaign implementation request. This
supersedes the 600k/150k population and undecided hardware in the earlier
miniature follow-up proposal. Historical execution
specifications and the separate scaling benchmark remain unchanged.

## Science

Use all 2,488,148 eligible TRAIN jets and 497,954 natural validation jets from
the authenticated new Run-2 QCD/Higgs data, six classes QCD/X_bb/X_cc/X_ss/X_qq/
X_gg, 200 <= offline jet pT < 2600 GeV. Replay the original unused-only role
extension; preserve test/unused roles and quarantine label206. Preserve the
audited population hash57eb7febd6ee6ae2c9729bdcc5f58a4f34dea232d7b1eaf12b7901ca405025d1.
No historical model, bank or reconstruction normalizer is a scientific parent.

All fits draw 2,488,148 times per epoch with replacement: uniform among144
class/pT cells, then uniform among that cell's TRAIN jets. Twenty epochs,
global batch256 (64 per GPU; last global84, 21 per GPU), 9720 updates/epoch.
The worst expected exposure is21.46435 appearances/jet/epoch. Validation is
natural/unweighted; selection is macro OVR AUC, then CE, macro logR50, earliest
update. Final-test inference remains sealed. Also report the600--2600 validation
subset at the same selected checkpoint, without reselection.

AdamW and original decay/epsilon settings, peakLR1e-4, two-epoch linear warmup,
cosine decay to5e-6 at epoch20. Use the actual consumed-draw clock, no automatic
LR scaling, early stopping, accumulation, precision changes or input cropping.

Three fresh matched seed indices20/21/22 each train OFFLINE, M0HLT, U100,
U100_W5 and D20: fifteen fits. OFFLINE is a privileged CE control; M0HLT is
HLT-only CE. U100 uses its privileged matched-particle view and the same-seed
selected OFFLINE EMA05 teacher, main objective0.25CE+0.75*T^2*KL with T2.
HLT-only U100_W5 adds weight5 standardized U100 EMA05 embedding prediction.
D20 adds the original complete OFFLINE mixed22 set reconstruction: decoder
coefficient0.3 from the beginning; encoder delay3, cosine ramp6 to0.3; native
jet and block8 particle-token conditioning; exact assignment, chunk8 backward.

Eight additional first-seed D20 fits each change exactly one parameter:

| Arm | Difference |
| --- | --- |
| D20_E01 | Encoder maximum0.1; decoder remains0.3 |
| D20_E06 | Encoder maximum0.6; decoder remains0.3 |
| D20_D1 | Encoder delay1 |
| D20_D5 | Encoder delay5 |
| D20_R3 | Encoder cosine ramp3 |
| D20_R9 | Encoder cosine ramp9 |
| D20_NO_PREP | Decoder follows encoder schedule, removing early decoder preparation |
| D20_EMB25 | Embedding coefficient2.5 |

All same-seed fits/variants use identical untrained native/predictor/decoder
tensors, TRAIN sampling and dropout domains. Shape-inapplicable components
are unused. Main HLT classifier has CE only; no additional main-head KD,
contrastive objective, subclasses or auxiliary logit heads are introduced.

RAW is the primary selected/readout for every model. Only outgoing OFFLINE
and U100 teachers additionally retain EMA05 (half-life0.5 draw epoch, start3,
fixed8192 balanced TRAIN BN recalibration) to preserve successful target
choices. Student EMA and RAW_BN are omitted; these were copy-only diagnostics
and never update the live training weights. RAW selected probabilities,
accuracy/AUC, per-class R50/R30, passing QCD counts, per-pT and600--2600 metrics,
matched RAW anchor recoveries and paired three-seed summaries are published.
The eight single-seed variants remain exploratory; censored tails/unresolved
anchor gaps remain unresolved, not job failures.

## Execution and continuation

New namespace `luka_d20_full_hydra`, owned versioned artifacts, thin workers.
Exact pushed clean source, immutable donor byte inventory, installed Weaver
runtime, verified transfer/producer, population, starts, policies and all
teacher/normalizer/checkpoint parents must authenticate before reuse.

One CPU prepare (4CPUs/96,000MiB/4h), four-GPU qualification
(12CPUs/96,000MiB/48h), then CPU release. Before any science submission require
installed-Weaver parity, known GPU SUM/control checks with no-P2P transport,
independent full-graph/four-rank gradient/AdamW parity, both schedule branches,
discarded real-input miniatures including save/restore equivalence, complete
particle support stress and measured full-validation/resource envelope.
Actual Hydra worker acceptance governs this Hydra-specific campaign; local
single-GPU/Tigris evidence cannot establish four-Pascal execution.

Each scientific fit has four afterok five-epoch segments (1--5/6--10/11--15/
16--20), each four GTX1080Ti/12CPUs/96,000MiB/48h:92 science allocations,
23 scientific fits. Measure the five-epoch envelope before release. Segments
restore full model/decoder, AdamW, teacher EMA where applicable, selected
native states/history and every rank's CPU/CUDA RNG at a completion-last epoch
commit. Sampling and schedules reconstruct from the immutable seed/epoch.
Previous state is retired only after the next commit is durable. Failed jobs
may explicitly resume at their last complete epoch; no partial epoch is used.

Every fit starts fresh; a segment continuation is the same fit. Canonical
starts are always reloaded after discarded acceptance probes. Only final
segment publication exports teachers/results; downstream students depend on
their teacher's final segment. No unrelated fit is serialized, cancelled or
mutated. Actual script dry review precedes live submission. Intent/response/
accepted-ID journaling and exclusive locks reject ambiguous duplicate submits.
Explicit recovery requires prior accepted jobs terminal and preserves completed
fits/segments. Poor scientific metrics never fail a row or block variants.

RAM-only allocation-local particle views avoid a persistent full input cache.
Fit allocations construct only their declared view and TRAIN-only targets.
Qualification constructs/reuses three full views sequentially; measured host
memory must fit the request. New TRAIN reconstruction moments are frozen once
and reused exactly across all reconstruction fits, with recomputation evidence.
Banks are unique TRAIN only, exported by rank-partitioned streaming writes and
fixed two-pass FP64 standardization. No giant object-gather or durable raw
particle cache is needed. Native forward cannot receive pT/indices/targets.

HOME admission is checked only at registration/explicit recovery, using a
24GiB output allowance plus2GiB spare. The initialized full-support output
inventory is14,748,932,625bytes (13.74GiB), including all simultaneously retained
restart/staging states; the allowance adds space for interrupted exports.
No automatic HOME rechecks at release/startup/epochs
and no runtime cap. Real filesystem errors propagate. Cleanup targets only
authenticated obsolete restart states or exact owned job scratch. No earlier
campaign data/model/environment is removed. All datasets remain read-only.

Queue readiness means local semantic, failure-injection, resume, CUDA/Weaver,
shell and dry-plan checks pass and the launch path gates science on real Hydra
qualification. It does not mean remote qualification was already performed.
