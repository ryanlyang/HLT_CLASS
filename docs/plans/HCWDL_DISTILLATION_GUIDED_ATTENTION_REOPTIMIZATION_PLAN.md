# HCWDL Distillation-Guided Attention Re-optimization Plan

Status: Phase-I full-four-spine implementation authority. Phase II is
recorded but deliberately deferred. This document authorizes implementation
and dry-run materialization only; live submission still requires the exact
creation/submission phrases and a successful genuine-Tigris gate. It does not
alter, hold, cancel, reprioritize, or write into any running campaign.

## 1. Scientific question

The HCWDL ladders show that a student can sometimes equal or outperform a
richer-view teacher, including early U transitions where a degraded student
can exceed U000. At the same time, Particle Transformer studies show that its
pairwise attention bias and pairwise value/message path are unusually strong
performance levers even at high data count. Ordinary logit KD constrains the
student's final class distribution but only indirectly constrains how its
attention blocks exchange constituent information.

The primary question is therefore:

> Can privileged teacher information guide a dedicated re-optimization of an
> already-trained student's existing ParT attention machinery, preserving or
> improving classification performance more effectively than ordinary logit
> KD or an equal-budget generic refinement?

The aspirational outcome is a ladder whose students remain near or above the
U000 reference. That is a hypothesis, not a completion criterion or promised
property. Finite underperformance is a valid scientific result.

## 2. Deployment and comparison invariants

- Every deployable model consumes authenticated HLT inputs only.
- Offline/repaired inputs, teacher activations, matches, and relational targets
  are training-only.
- Phase I adds no deployable parameters or inputs. It re-optimizes selected
  parameters already present in the canonical Weaver Particle Transformer.
- Every comparison uses the same authenticated student view, population,
  initialization lineage, source checkpoint, sampler, batch size, labels,
  total additional update budget, and validation-selection rule.
- An equal-budget full-model refinement separates attention targeting from
  simply training longer.
- An attention-only CE control separates attention-specific KD from ordinary
  supervised attention fine-tuning.
- Final test remains sealed. Validation metrics never control job success or
  downstream registration.
- No dense token, pair, attention, message, or residual target bank is written
  to durable research-compute storage.

## 3. Relationship to existing work

This design is related to, but distinct from, existing repository mechanisms:

- PMARD `R4_PAIR` matches the actual Weaver pair-geometry/attention-bias
  tensor. Phase I instead optimizes the student's attention parameters and
  primarily distills the *effect* of attention through projected messages and
  block residual updates.
- PRAD predicts a new privileged relation bias and injects it through
  zero-initialized gates. That is closest to this document's deferred Phase II
  attention oracle. Phase I changes no architecture and should be tested
  first.
- TRI60 RSET/RREL distill selected hidden representation surfaces. Phase I
  targets a staged parameter subset and the attention-induced update, rather
  than treating a hidden state as a static endpoint target.
- MHPE probability ensembles have no canonical head ordering or raw attention
  tensor. Ensemble probabilities may remain the logit teacher, while Phase I
  uses an explicitly registered single relational carrier. Averaging
  post-projection messages or block deltas across ensemble components is a
  later ablation.

Nothing in this plan redefines those immutable campaigns or artifacts.

## 4. Why raw attention-map imitation is not primary

Raw attention matrices are not unique explanations of a model's computation.
Heads can permute, per-query logit offsets are softmax gauges, and different
attention/value combinations can produce the same downstream update. Token
counts and support may also differ across U transitions.

The primary relational surfaces are therefore:

1. **Projected attention message:** the result after head aggregation and the
   attention output projection, before the block's residual addition.
2. **Block residual delta:** `state_after_attention - state_before_attention`
   or, where Weaver exposes only a complete block boundary, the registered
   complete-block residual delta with its meaning named honestly.
3. **Centered pair bias:** an optional diagnostic/auxiliary target, centered
   over valid keys per query and head to remove the softmax offset gauge.

Raw attention weights remain a diagnostic ablation. They are not the default
loss and are never averaged across unaligned ensemble heads.

## 5. Phase I: distillation-guided attention re-optimization

### 5.1 Authorized full-four-spine confirmation

The user explicitly promoted Phase I from the originally proposed D-path
mechanism screen to a complete, isolated four-spine experiment. The scientific
paths are exactly the persistent-HLT full-cardinality paths:

```text
DIRECT:     SP4P_U000 -> D000
COARSE:     SP4P_U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:      SP4P_U000 -> U033 -> U066 -> U100 -> D080 -> D060
                       -> D040 -> D020 -> D000
ULTRADENSE: SP4P_U000 -> U020 -> U040 -> U060 -> U080 -> U100
                       -> D090 -> D080 -> D070 -> D060 -> D050
                       -> D040 -> D030 -> D020 -> D010 -> D000
```

This is 30 fresh fits including one shared `SP4P_U000` anchor and 26 compact
single-component probability reducers. There are no ensembles and no weight
continuation. Every downstream node has one immediate parent, used both as
the logit teacher (through its validated compact probability bank) and as the
live relational carrier (through its selected checkpoint).

The support is exactly the registered
`persistent_hlt_skeleton_remove_offline_tail_v1` policy: matched slots carry
offline features throughout U, persistent unmatched-HLT slots remain HLT,
and U transitions remove the offline-only tail. This campaign rebuilds its
own altered `SP4P_U000`; the existing persistent-HLT campaign is a read-only
matched control and need not be complete before this campaign is created or
queued.

### 5.2 Three-stage procedure

```text
Stage 0: 60-pass ordinary C25/P75, T=2 fit from fresh initialization
         -> restore its selected checkpoint

Stage A: attention re-optimization
         -> freeze non-attention backbone and classifier surfaces
         -> train existing pair embedding and particle-attention projections
         -> CE + original logit KD + registered relational KD + trust region

Stage B: joint refinement
         -> unfreeze the complete ordinary student
         -> continue CE + logit KD + relational KD at a lower learning rate
         -> select the best registered checkpoint on validation
```

`SP4P_U000` is the one 60-pass CE anchor and has no attention rescue. Every
downstream fit is fresh and runs exactly 100 passes: Stage 0 passes 1--60,
Stage A passes 61--75, and Stage B passes 76--100. Stage 0 uses the registered
warmup/hold/cosine schedule (warmup 1--3, hold through 45, decay through 60 to
`1.5e-5`). Stage A uses `3e-5`; Stage B uses `1.5e-5`. The optimizer is rebuilt
at both boundaries and carries no optimizer state across them. Performance
early stopping is disabled. The final report retains both selected and final
checkpoints and may legitimately select a Stage-0 checkpoint if rescue does
not help.

### 5.3 Trainable parameter groups

The Stage-A semantic parameter group is:

- canonical `pair_embed` parameters;
- Q, K, and V projections in the registered particle self-attention blocks;
- attention output projections in those blocks;
- attention-path normalization parameters only if the installed Weaver block
  cannot separate them from the attention sublayer.

Frozen in Stage A:

- particle input embedding;
- feed-forward/MLP sublayers;
- class-attention blocks and pooling/aggregator;
- final classifier;
- every teacher parameter.

The exact installed-Weaver parameter names and block indices must be compiled
into a versioned parameter-group artifact. Structural introspection must fail
closed if the installed model differs. Parameter-name substring guessing in a
production worker is forbidden.

Stage B unfreezes the complete canonical student. Phase I does not add a pair
adapter, projection head used at inference, or any new inference-time input.

### 5.4 Loss

The ordinary task objective remains the source transition's registered
C25/P75, T=2 objective:

```text
L_task = 0.25 * CE + 0.75 * KD_logits(T=2)
```

The re-optimization objective is:

```text
L_total = L_task
        + lambda_message * L_message
        + lambda_delta   * L_block_delta
        + lambda_pair    * L_centered_pair
        + lambda_trust   * L_attention_trust_region
```

Rules:

- The initial primary implementation may use message, delta, or both, but its
  exact choice and coefficients are frozen before result-bearing submission.
- Every relational loss is independently normalized over layers, valid
  particles/pairs, and feature width so coefficients have stable meanings.
- Teacher outputs are detached FP32 targets produced in evaluation mode.
- Student relational losses are accumulated in FP32 under the existing BF16
  model-autocast policy.
- Padding is excluded exactly. The primary D-path loss covers all visible HLT
  skeleton positions and does not reveal a private accepted-match mask.
- `L_attention_trust_region` is measured against the immutable Stage-0
  checkpoint and prevents an unconstrained attention rewrite. Its exact
  parameter- or output-space definition must be registered, not chosen after
  results.
- CE remains nonzero so the student is allowed to correct teacher errors and
  exceed the teacher. KD is an anchor, not a ceiling.

The frozen full-spine primary objective is complete-block residual-delta Gram
matching on all eight particle blocks, weighted by `0.25`, plus an
attention-parameter trust region weighted by `0.01`. The token Gram is built
only on the authenticated intersection of transported `(visible_index,
family_code)` identities, excludes padding and diagonal pairs, and is
channel-rotation tolerant. These values are scientific identities; they may
not change within this campaign after seeing validation results.

### 5.5 Deferred mechanism-screen controls

| ID | Starting checkpoint | Stage-A trainable set | Relational KD | Stage B | Purpose |
|---|---|---|---|---|---|
| `SOURCE` | immutable source | none | none | no | ordinary KD result |
| `FULL_REFINE` | source selected | all parameters | none | already joint | equal-budget longer-training control |
| `ATTN_CE` | source selected | attention only | none; CE only | yes | supervised attention control |
| `ATTN_LOGIT` | source selected | attention only | logits only | yes | staged-freeze/logit-KD control |
| `ATTN_REL` | source selected | attention only | message/delta | no | isolates attention phase |
| `ATTN_REL_JOINT` | source selected | attention only | message/delta | yes | primary proposed procedure |

These six arms remain valuable for causal attribution, but they are not mixed
into the authorized full-spine campaign. The existing persistent-HLT
four-spine campaign is the matched no-attention control at every available
rung. A later six-arm screen may separate generic extra optimization from the
attention mechanism without changing this campaign's immutable meaning.

An optional `ATTN_PAIR_ONLY` arm may test centered pair-bias KD, but it cannot
replace the projected-message/block-delta primary arm.

### 5.6 Teacher execution and storage

- The frozen relational carrier runs in `eval()` with gradients disabled.
- Teacher relational surfaces are generated in bounded batches in the same
  job that consumes them.
- Existing compact class-probability banks may be reused after complete
  lineage validation.
- Dense `[N,N]` pair tensors and `[N,C]` token messages remain in device/RAM
  scope only and are released after the batch.
- No pair/message target shard, rolling dense cache, or hidden temporary file
  may be written beneath the project or campaign root.
- A production miniature must measure the cost of the second teacher forward,
  peak GPU memory, host RAM, and throughput before full-data submission.

## 6. Phase-II: privileged attention oracle

Phase II is deliberately separate so a Phase-I gain cannot be attributed to
additional inference capacity.

### 6.1 Oracle teacher

A temporary privileged teacher receives authenticated offline/repaired
pairwise relations and learns an additive pair-bias and/or pairwise-value
correction. It may optimize classification CE plus an anchor to the ordinary
teacher. The oracle is training-only and may be stronger than U000; its purpose
is to construct a better relational target, not merely replay an existing raw
attention map.

Two oracle forms are retained for later study:

1. **Learned privileged oracle:** a PRAD-like offline relation module produces
   bounded centered bias/value corrections and is trained across the train
   population.
2. **Inner-loop attention oracle:** bounded per-batch bias perturbations are
   optimized against train labels under a norm constraint, then used only as
   targets. This is more expensive and remains exploratory until the learned
   oracle is understood.

### 6.2 Deployable student

The corresponding student may receive a zero-initialized HLT-only residual
pair-bias module:

```text
B_student(i,j) = B_ParT(i,j) + delta_B_HLT(i,j)
```

`delta_B_HLT` consumes only canonical HLT quantities. Zero initialization must
give exact baseline logits and gradients under the parity contract. A
matched-capacity CE-only student with the identical module and schedule is
mandatory. This is the decisive control separating privileged distillation
from added model capacity.

### 6.3 Ensemble extension

For an MHPE teacher, ensemble logits remain the probability mean. Relational
consensus is formed only after each component's head aggregation/output
projection, by averaging aligned projected messages or block deltas in FP32.
Raw heads or attention matrices are never averaged without a separately
proved alignment.

## 7. Measurements and interpretation

Primary classification outputs:

- validation accuracy, macro OVR AUC, macro linear R50;
- M0CE60-to-U000 recovery for AUC and R50;
- per-class Hbb, Hcc, Hqq and complete registered class R50;
- selected pass/update, final-pass metrics, and walltime.

Mechanism outputs:

- normalized message and block-delta fidelity by layer;
- centered pair-bias fidelity when enabled;
- attention-parameter displacement from Stage 0;
- logit divergence from both teacher and source student;
- gradient norm and update norm by semantic parameter group;
- matched comparison against `FULL_REFINE` and `ATTN_CE`.

A result above U000 is scientifically valuable but not required. The minimum
claim supported by a positive result is narrower: targeted privileged
attention re-optimization improves compression beyond equal-budget ordinary
refinement and attention-only CE. Claims about an entire ladder require a
predeclared multi-rung confirmation.

## 8. Artifact and failure contracts

Implementation requires distinct versioned artifacts for:

- authenticated source transition and relational-carrier lock;
- semantic parameter-group registry;
- relational surface and normalization specification;
- staged optimization recipe and exact update budgets;
- campaign specification and command plan;
- training/runtime reports with per-stage metrics;
- aggregate, completion, monitor, and source-pinned recovery.

Invalid source lineage, a changed student view, teacher gradients, unregistered
trainable parameters, nonfinite required quantities, stale source, durable
dense-target publication, forbidden final-test access, or parity failure must
fail closed. Poor attention fidelity or classification performance must not.

## 9. Implementation order

1. Freeze the full four-spine graph, immediate-parent teacher/carrier,
   exact Stage-A block set, coefficients, and schedules.
2. Add a parity-safe Weaver surface adapter that exposes pre/post attention or
   honestly named block-boundary deltas without changing ordinary inference.
3. Implement and unit-test the semantic parameter-group compiler and staged
   freeze/unfreeze optimizer reconstruction.
4. Implement normalized FP32 message/delta/pair/trust losses and frozen-teacher
   guards.
5. Add the 30-fit/26-reducer Phase-I trainer, per-stage reports, aggregate,
   completion, and exact source-pinned recovery in an isolated root/job
   namespace.
6. Run local parity/gradient/freeze tests, installed-Weaver parity, and a real
   Tigris production-worker miniature with measured resources.
7. Materialize both gate and science dry-run ledgers. Submit only the three-job
   gate; release the 58-task science plan only after its genuine GH200
   acceptance and parameter locks validate.
8. If Phase I is informative, register the six-arm causal mechanism screen.
9. Only then implement the Phase-II privileged attention oracle and matched-
   capacity residual-bias study.

## 10. Required tests before Phase-I submission

- ordinary canonical ParT logits and parameter state remain byte/tolerance
  identical when the surface adapter is unused;
- teacher stays in evaluation mode with zero gradients;
- Stage A exposes exactly the registered attention parameters and freezes all
  others; Stage B exposes exactly the registered full student;
- optimizer state is rebuilt without silently updating frozen parameters;
- the primary masked residual-delta Gram loss is invariant to padded values
  and fails closed when a batch has no authenticated common ordered pair;
- any later centered-pair ablation must prove invariance to per-query constant
  offsets before it receives result-bearing authority;
- message/delta alignment follows authenticated token identities;
- every downstream rung executes the exact 60/15/25 pass registry and the
  anchor executes exactly 60 passes;
- no offline input or relational target is reachable from deployable forward;
- no dense target artifact is durable;
- final-test capability is absent;
- scientific underperformance still produces complete reports and descendants.

For the full-four-spine Phase-I campaign, the numerical coefficients, all
eight block indices, stage schedules, support policy, and immediate-parent
carrier rule above are frozen scientific identities. Phase-II details remain
research direction rather than executable authority.
