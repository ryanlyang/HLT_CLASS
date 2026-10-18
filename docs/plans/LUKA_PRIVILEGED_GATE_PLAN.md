# Luka privileged attention gate: implementation plan

Authorized 2026-10-18: implement a direct test and a small set of variants,
queue-ready for SPORC debug. No submission is authorized by implementation.
This independent `luka_gating` campaign does not modify the running attention
loss experiment, fixed30 campaign, raw data, or Fresh_check.

## Hypothesis and frozen design

Keep native ParT Q/K/V and own-view physics pair bias. An additional learned
positive pair gate multiplies the native attention before renormalization:
`A' = normalize_keys(A * exp(R))`, equivalently `softmax(native_scores + R)`.
The gate is learned for the poorer-view classifier, NOT copied teacher
attention or a predictor of richer physical descriptors. It receives actual
richer inputs throughout the privileged ladder. Only final replacements
predict the extra gate using HLT alone. Values enter from the poorer stream;
later hidden states naturally carry the effects of privileged routing.

Frozen gate v1: token-local LayerNorm(34), Linear(34,32), GELU, Linear(32,9).
Inputs concatenate own-view and aligned richer 17-channel features. Nine
outputs are q(4), k(4), key_bias(1). R_ij = log(4)*tanh(q_i dot k_j / 2 + b_j).
The single head-shared correction is applied to particle blocks 6 and 7 only;
no class attention, native bias replacement, richer values, or second encoder.
Initial key and bias output rows are zero, hence gate identity without a dead
bilinear branch. The bound limits unnormalized multipliers to [1/4,4].
There is no guarantee that the privileged correction is predictable from HLT.

## Registered comparison (13 fits, 23 science jobs)

All three chains warm-start the same authenticated regular COARSE_U100 from
fixed30 commit74aa58c582ff6f497a2c6d91c503f5489fac3e9a and descend D066/D033/D000.
Each chain uses its own predecessor's TRAIN logits and selected weights:

* WARM: native model, no gate.
* SELF: same gate architecture, second input is own view, no privileged gate.
* PRIV: second input is immediately richer U100/D066/D033, respectively.

From one frozen selected PRIV_D000 teacher and identical copied weights:

* REPLACE_LOGIT: own-view/own-view gate, C25/P75 T2 only, all weights trainable.
* REPLACE_GATE: same, plus 0.1 centered log-gate MSE, all weights trainable.
* REPLACE_STAGED: same loss, first3 of30 epochs gate-only, then27 joint epochs.
  Frozen backbone is in eval mode (BN/dropout frozen), not just requires_grad.

SELF_CONT is an extra30-pass self-KD continuation of SELF_D000, matching
endpoint architecture and total number of fits along the route. Comparisons
still differ in teacher information, not strictly total FLOPs. Parent M0HLT,
OFFLINE, DIRECT_D000 and COARSE_U100 remain labeled reference rows.

Every new fit is exactly30 epochs, batch128, accumulation1, existing
warmup3/cosine-to30 AdamW recipe and coordinate seeds; fresh optimizer per fit.
Replacement variants share the D000 sampler/initialization seeds. Selection is
full validation AUC/CE/logR50/update, never original50k or gate-prediction error.
Technical probes are1 epoch and are never scientific parents.

## Data, safety, interpretation

Reuse unchanged trial ALLJETS299234 TRAIN /74895 validation; original50000 is
secondary only. Historical test was evaluated and retired into training: NO
independent test. Only retained U100/D066/D033/D000 caches are loaded. Alignment
is authenticated ordered row identities, labels, cardinalities and unchanged
persistent-view builders, not a truth-matching claim. No assignment/source IDs
or richer validity masks enter deployable inputs. PRIV_D000 is an oracle.
All replacements and SELF_D000/SELF_CONT consume exclusively D000 at inference.

TRAIN probability banks are small. Only PRIV_D000 additionally exports nine
FP32 factors per particle; dense quadratic targets are reconstructed per batch,
not stored for the dataset. Gate losses center over valid keys and average per
query then per jet. Teacher factors are detached, masks exclude padding.

## Acceptance and queue boundary

New v1 artifacts bind source, data, warm start, teacher checkpoint/bank and
full graph. Source replay runs old authentication in the original worktrees;
new modules may not silently rewrite old schemas. Separate dry/live exact-hash
gate and science plans, durable submission journal, no automatic retry.
Manual submission remains the default. The user's subsequent 2026-10-18
authorization adds an explicitly opt-in `auto` controller, described below;
it does not change any scientific recipe, row, or artifact meaning.

Gate: installed-Weaver identity logits/gradients/updates FP32 and BF16;
multiplicative-gate equivalence and privileged-only direct entry path; HLT
endpoint no-context contract; every-node real miniature/selected checkpoint
readback; per-role maximum-support backward/inference stress; full-population
timing, disk and memory measurement. Science uses measured per-kind bounds
with margins and must fit debug24h. No quality threshold, no silent batch,
epoch or particle-count reduction. A successful technical gate is required
before science submission, authorized either separately or through the bounded
follow-up policy reviewed before queueing the gate.

### Opt-in overnight follow-up

`auto` dry review registers the exact gate plan, all23 science command templates,
and ceilings: the same debug/account/QOS/GPU/CPU/memory profile; at most1440min
per measured train/export task; aggregate60min/1CPU/16384MiB/noGPU. Only measured
train/export walltimes and the measurement-bound science-spec hash are deferred.
The user must authorize this distinct follow-up plan's content hash. A gate
plan authorization alone never authorizes science. The immutable gate's
`automatic_followup=False` remains its default; a separate opt-in authorization
artifact records the additional authority without rewriting the gate.

The submit-host controller runs under nohup. It submits exactly one preflight
(or binds an already accepted gate from its exact journal), publishes ARMED,
and polls that exact job every60s, at most14days. COMPLETED/0:0 alone is not
enough: all existing source, payload, parity, miniature, resource and accounting
checks still run before creating science. Failed/cancelled preflight, invalid
evidence or exceeded resource ceilings stop the controller; poor performance
does not. Once admitted, the measured exact plan is saved and dry-checked,
compared against the authorized graph/ceilings, then submitted through the
existing dependency DAG and durable per-job journal.

A permanent exclusive controller claim prevents duplicate arming. Existing
science directories are never taken over. No automatic retry, cleanup,
cancellation or recovery from ambiguous submission is allowed. Preserve logs
and journals after interruption; never also submit science manually while
armed. A controller survives SSH disconnect under nohup, not a login-host
reboot or administrative process termination. ARMED means waiting, not that
science is already queued; FOLLOWUP_COMPLETE means submission, not training
completion. This operational addition uses distinct FOLLOWUP_*/v1 contracts.

Queue-ready means ready to queue this new gate; it does not claim real A100
acceptance from local tests. Original experiments and unrelated jobs unchanged.

### 2026-10-18: frozen-backbone attention execution correction

Job21914569 failed in the first REPLACE_STAGED backward with
`LSE is not correctly aligned (strideH)`. Its ordinary trainable stress did
not exercise the three-epoch gate-only mode. The scientific recipe above is
unchanged: no extra QKV gradients, unfreezing, batch reduction, or loss change.

An explicit ATTENTION_EXECUTION/v1 policy selects PyTorch math SDPA only for
a registered gated CUDA attention call when gradients are enabled, its added
bias requires gradients, and its inputs and attention parameters are frozen.
All other calls retain native dispatch. The previous backend flags are restored
on return or exception. This avoids the Torch2.5.1 efficient backend's omission
of bias-only gradients when deciding whether to save log-sum-exp for backward.
Native Weaver code and state-dictionary keys are unchanged. Backend numerical
roundoff can differ; this is not a bitwise identity claim.

PREFLIGHT/v2 requires early gate-only classification-gradient/update and frozen
parameter/buffer checks in FP32 and BF16, plus unfreeze and inference checks.
FP32 additionally compares logits and gate gradients to full math attention;
BF16 is a finiteness/update check, not equivalence to forcing unaffected layers
to math (which changes their reduction precision). Before any full-population
fit, test true batch128 at maximum TRAIN support with the backbone frozen.
The ordinary joint-training and validation-support stresses remain required:
27 stress cases, not26. The full one-pass technical graph and resource admission
remain mandatory. Missing frozen-mode evidence or old receipts fail closed.

Use new pushed source and a fresh gate root; never edit the failed gate, remove
its follow-up claim, or reinterpret partial probes as accepted evidence. The
existing dry-review/manual/opt-in-auto interfaces remain available. This fix
does not implement partial-probe recovery or submit jobs.
