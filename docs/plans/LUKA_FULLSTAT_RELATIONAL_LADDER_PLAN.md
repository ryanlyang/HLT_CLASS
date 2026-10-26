# FullStat K2.5/E10/relational ladder (Oscar, fixed20)

Active scientific authority, user approved 2026-10-26. Additive experiment;
do not change/cancel the current split-head recovery ladder. This supersedes
the older single-U100 relational proposal for THIS campaign only. The pasted
trial recipe contributes losses, not its population, teachers or 30 epochs.
User-approved revision 2026-10-26: choose `k2p5_e10_rel` instead of `k5_e10_rel`.
Change ONLY total auxiliary-logit weight from 5 to 2.5; embedding weight 10 and
relational supervision remain unchanged at every stage. The reported screening
AUC 0.963040 is effectively tied numerically with K5/E20/plain at 0.963044. These
single-seed screening results motivate the choice; they do not establish which
recipe generalizes better or predict this FullStat ladder's outcome.

| Fresh stage | Input | Frozen teachers for KD, embedding AND relation |
| --- | --- | --- |
| UE100 | U100 | OFFLINE, M0HLT |
| DE066 | D066 | OFFLINE, NEW UE100, M0HLT |
| DE033 | D033 | OFFLINE, NEW UE100, NEW DE066, M0HLT |
| DE000 | D000 | OFFLINE, NEW UE100, NEW DE066, NEW DE033 |

All four start from scratch. No old UE100 checkpoint/target substitution.
Authenticate completed finalization/retry DE000 and its ancestry in that
execution's original pinned source BEFORE create/submission. The old0c4a71a5
root contains failed DE033/cancelled DE000, not a completed parent.
Submission requires the current ladder to finish; it does not poll for it.
Use explicit runtime_repair_v1 with the original environment fingerprint,
without changing the legacy site profile or original workers.
Historical models are comparisons, not additional fresh control fits.

## Objective and population

For N teachers: 0.5 CE(main) + 2.5/N sum T^2 KL(q_teacher||p_aux)
+ 10/N sum SmoothL1(predicted_embedding, target) + beta/N sum relation_KL.
T=2, forward KD batchmean, T-squared exactly once. SmoothL1 beta1, mean over
queries/dimensions. Main native classifier receives CE only; separate Linear128->11
logit heads and Linear128->GELU->Linear128 embedding heads read shared native
features independently. All auxiliary objectives update the shared encoder.
Neither actual nor predicted targets enter the main head or feed auxiliary logits.
Per-teacher KD coefficients are 1.25 for UE100, 2.5/3 for DE066, and 0.625 for
DE033/DE000; embedding coefficients remain 5, 10/3, 2.5, 2.5 respectively.

Targets: frozen selected native 128-D pre-classifier representations, separately
TRAIN-standardized (FP64 moments, floor1e-3, stored FP32), and native T2 probabilities.
Never export predicted vectors/auxiliary probabilities as targets. Original
OFFLINE/M0HLT banks retain their contracts; each new intermediate exports its own
native banks. Final DE000 has no outgoing bank. No averaging teacher latent axes.

Relation: independent PCG64 SeedSequence([314159,epoch,batch_index]), 2048 distinct
TRAIN reference rows/update shared across teachers:186 per class and two extras
rotating as (2*zero_based_update)%11 and next. Require >=187 rows/class. Same-jet
mask on both softmaxes after authenticated unique row joins. No class mask.
Per teacher, L2 normalize predicted/true query and that teacher's frozen references
(eps1e-8), cosine softmax tau0.2, forward KL summed over references, mean queries,
mean teachers; no tau-squared. beta=min(1,one_based_update/(2*ceil(Ntrain/128))).
Relation executes FP32 outside autocast, native accepted CUDA BF16 is unchanged.
Reference sampling does not advance model/dropout/query-sampling RNG. No extra
teacher or reference-particle forwards per update; gather resident cached vectors.

Same authenticated 3M TRAIN/500k validation, paired class/QCD-bin jet-row split;
event independence unverified. 1.5M final test SEALED;105325 unused remain unused.
Exactly20 full epochs, physical128 including tail, original FullStat AdamW,
coordinate-paired seeds, warmup2/cosine20. No accumulation, warm start, early
stopping, alpha grid or fallback. Select main-head full-validation AUC with
existing deterministic tie breaks; use the same checkpoint for R30/R50.

## Direct execution without a standalone preflight

Explicit user-authorized exception: four scientific GPU jobs only, afterok
dependencies, no setup/preflight/reducer/controller job. Exact clean pushed source,
full dry sbatch review and exact plan-hash authorization still required. All new
receipts/targets/source stay in a fresh root; no old files/jobs deleted or changed.
Reserve16 GiB personal durable headroom (filesystem free space is not quota).

User-authorized 2026-10-26 launcher: after one explicit bounded authorization,
detach authentication/submission with nohup/setsid and immediately tail the log.
Authorize exactly these four recipes, pinned source, specified completed parent,
fresh root and declared quota. Persist intent, full dry plan/ledger and exact
plan-hash approval before live sbatch. No second prompt for that automatic path;
manual dry-review/hash execution remains available. Refuse any changed/expanded
plan. Exclusive ROOT.launch/process_claim and live journals prevent duplicate
same-root launch. Never automatically retry/erase claims. SSH disconnect/Ctrl-C
stops only tail; login-node failure, corruption or scheduler errors can still
stop submission. Preserve records and inspect before any further action.

Reuse authenticated original Oscar L40S/6CPU/180000MiB environment/support evidence.
Request max(24h,ceil(1.5*largest inherited split-head request)+60min), reject>48h.
This is a conservative allocation, NOT remotely measured relational throughput.
Inside EACH worker: native FP32/BF16 parity, three disposable physical128
maximum-support updates at full relation weight, CE-only main-head gradient,
finite connected auxiliary/backbone gradients, maximum-validation support,
then four warmup+32 representative full-objective updates and16 validation
batch timings. Include reference sampling/gathers/transfer/backward/optimizer.
Require memory headroom and 1.5x projected20-epoch time+120min <= allocation.
The projection is not a complete measured epoch/validation pass. Technical
failure blocks descendants; no silent fewer epochs/batches/particles. Discard
all technical weights and reset scientific RNG before training.

Recipe/v2 and SPEC/PLAN/SUBMISSION_CLAIM/v3 identify the K2.5 campaign. Existing
K5 specs/plans are rejected, not rewritten or resumed. Use a fresh root and
the revised authorization tokens. Native target formats, parent authentication
and completed historical campaigns remain unchanged.

Publish FIT before outgoing targets, TASK last. Require exact saved native
prediction readback. Recompute report totals in registered teacher order, NOT
JSON mapping order. Preserve partial output on failure, no automatic retry.
Poor finite AUC/rejection/target quality is scientific evidence, never a gate.

## Interpretation and acceptance

Compare completed DE000 with historical original direct/logit, current split-head,
U100-w5/four-teacher references on identical validation rows. Intermediates are
richer-input oracles, not deployable HLT improvements. No fresh matched control
or isolated relation ablation; single-seed adaptive validation, no significance
or independent-test claim. Stronger teacher AUC does not guarantee final gains.

Require focused loss/gradient/RNG/anchor tests, native installed-Weaver parity,
production-worker synthetic four-by20 miniature, bank corruption/role/lineage
failures, canonical JSON report roundtrip and exact mock dry/live DAG tests.
Local GPU evidence is not a real-data Oscar/Tigris miniature. The new combined
objective's Oscar resource acceptance remains the integrated worker check.
Original science donor028c3314394949706787c6d1ef5e632f53ca157c. Execution v2
boundary420d22f8feb9f2237ee2dc2cbf3000d881ecd6a6 permits the separate finalizer
retry on narrow enumerated paths; every parent runtime file must ALSO match
that completed parent's exact pinned bytes. This patch edits no recovery files.

See [contracts](../contracts/LUKA_FULLSTAT_RELATIONAL_LADDER.md) and
[queue/results runbook](../LUKA_FULLSTAT_RELATIONAL_LADDER_RUNBOOK.md).
