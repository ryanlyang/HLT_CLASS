# Luka D attention contracts v1

Authority: [implementation plan](../plans/LUKA_D_ATTENTION_PLAN.md).

New `LUKA_D_ATTENTION_{RECIPE,ANCHOR,GATE,SPEC,PLAN,PREFLIGHT,TASK,BANK,
TRAINING_REPORT,RESULTS,SUBMISSION_CLAIM}/v1` artifacts use canonical content
hashes, exact parent hashes and source lineage. This does not extend any old campaign schema.
All records preserve ALLJETS historical-test flags and forbid test operations.

TRAIN banks bind selected teacher checkpoint/report, prepared-data hash,
coordinate, ordered identities/labels/counts and persistent-slot convention.
The slot digest contains ordered jet identities, labels and counts, not a
truth-level particle ID. Its order guarantee additionally requires the prepared
manifest/file hashes and unchanged persistent-view builder authenticated by the
parent replay. Equal counts from arbitrary, separately reordered inputs are
not eligible evidence of correspondence.
Temperature2 probabilities are float32(N,11). Optional attention consists of
float32 flattened real Nparticle-squared matrices, bounded shards, with int64
offsets/counts and exact ordered identities. No padded matrices or model-visible
source/matching IDs. Attention is final particle block7, mean over heads after
softmax, before dropout; native pair bias and masks included. Saved matrices
are unsmoothed, row-normalized probabilities; both sides receive epsilon1e-8
smoothing only in the loss. All real queries/keys, including diagonal/unmatched
slots, participate. Nonfinite/negative/zero-mass rows fail before smoothing.

Immutable publication writes payloads then the manifest/receipt last. Every
payload is hashed before first use; bounded read-only shard caching cannot
replace authentication. Partial directories cannot be reused automatically.
The classifier checkpoint has ordinary native ParT state keys: attention
readout is a training facility, not a new deployment input or parameter.

Each selected checkpoint is chosen on full validation AUC/CE/logR50/update;
the original50k is report-only. Scientific fits are exactly30 passes and
cannot be substituted by technical miniature outputs. Each arm uses its own
immediate parent's probabilities/attention, except common frozen U100.

Gate acceptance requires real site/environment identity, installed-model
parity, six-fit miniature and bank roundtrips, TRAIN backward and validation
inference maximum-support stress, complete-population timing, storage and
host/GPU headroom. Source/data/resource corruption fails closed; poor metrics
do not. Science requires COMPLETED/0:0 exact gate journal/accounting and a
separately reviewed immutable plan. No automatic retries or submission.

The preflight performs one full-population pass per node of both arms, plus
maximum-support stress and selected-checkpoint/native readbacks. These are
technical probes, never scientific teachers. Scientific requests include
measured cache/bank loading, two original-source authentication replays and
acceptance-payload hashing, 30 measured train/validation passes and selected
readback, with 1.75 training / 2.0 export safety factors. The resulting request
must fit the 24-hour debug ceiling; exceeding it blocks science rather than
silently reducing epochs or particle counts.
