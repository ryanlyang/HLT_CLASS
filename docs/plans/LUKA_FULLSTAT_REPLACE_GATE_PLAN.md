# FullStat final-rung cold replacement gate v1

This is a new Oscar experiment, not a continuation of the trial fixed30
campaign. Authority: the user's 2026-10-20 request and
[strategy](../LUKA_REPLACE_GATE_STRATEGY_HANDOFF.md).

Reuse the authenticated FullStat 3,000,000 TRAIN / 500,000 validation cache,
original selected COARSE_D033 (epoch17), its TRAIN probability bank, and
historical M0HLT/OFFLINE/DIRECT_D000/COARSE_D000 comparisons. The 1,500,000
final-test and 105,325 unused rows remain sealed. Jet-row splitting does not
prove event independence or absence of overlap with the earlier trial.

Three fresh fits, all FullStat20/batch128, identical D000 seeds:

1. PRIV_D000: native D000 ParT, gate context D033, labels and D033 logit KD.
   Its validation and teacher export require D033; this is an OFFLINE ORACLE.
2. REPLACE_LOGIT: fresh self-gated D000, labels plus selected PRIV_D000 logit KD.
3. REPLACE_GATE: the identical fresh self-gated D000 with the same supervision,
   plus 0.1 centered bounded log-gate MSE from selected PRIV_D000.

Main objective is .25 CE + .75 T² KL at T=2. Both replacements always classify
through their OWN predicted self-gate; no teacher forcing, inherited weights,
frozen stages, offline inference input, auxiliary embedding loss or full
attention KL. Retain native AND gate weights at deployment.

Reuse unchanged `luka_gating.model`: LN34 -> Linear32 -> GELU -> Linear9;
rank4, FP32 log(4)*tanh(u_i.v_j/2+b_j), shared over heads and particle
blocks6/7, native physics pair bias retained. Zero initial correction, padded
slots excluded. Gate MSE centers each real query over real keys, includes
diagonal, averages pairs within each jet then jets equally.

D000/D033 must share exact prepared lineage, ordered IDs, labels, counts and
canonical HLT slots. Never sort/rematch or pass construction indices into a
model. TRAIN targets only; no validation gate targets.

Unchanged FullStat optimizer, two-epoch warmup, cosine decay, full20 epochs,
per-epoch full-validation selection by AUC/CE/rejection/update tie rule.
Compare REPLACE_GATE minus REPLACE_LOGIT as primary; original coarse/direct
are historical controls, not architecture-matched gate controls.

Operational port: five science jobs (privileged fit, selected-teacher export,
two replacement fits, aggregate), with afterok dependencies. Export compact
FP32 nine-factor ragged shards once, authenticate and load factors contiguously
in host RAM for each consumer; never store dense pair targets or perform random
network shard reads every batch. Targets and immutable epoch checkpoints are
retained in the new root (no automatic deletion/resumption). Declare at least
24 GiB personally available durable space, additionally checked against measured
bank/checkpoint sizes. This intentionally differs from trial scratch retirement.

One real Oscar L40S preflight: installed native/gate FP32/BF16 parity, six
role-specific full-support stresses, all three full-population one-epoch
technical fits, selected-checkpoint readback, full TRAIN factor export/readback,
actual validation, resource/storage projections. Technical teachers/targets
cannot enter science. No score threshold; poor performance remains a result.
Clean exact pushed source, immutable dry review, successful exact gate accounting
and measured <=48h requests are required before separate science authorization.
Local fixture success is not remote admission.
