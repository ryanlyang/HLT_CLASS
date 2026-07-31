# Cross-Campaign Experiment Contract

Every campaign plan and immutable campaign specification must instantiate
these rules.

## Required declarations

- hypothesis and allowed claims;
- baseline, candidates, capacity controls, mechanism controls, null controls,
  and oracle diagnostics;
- split sizes and access permissions;
- HLT profile, strength, realization policy, and replicas;
- model architecture and initialization;
- optimizer, update budget, schedule, dtype, and deterministic seed derivation;
- checkpoint selector and exact tie rules;
- metric and statistical definitions;
- selectable versus report-only roles;
- artifact layout, versions, hashes, and parent lineage;
- production DAG and resume policy;
- finalist and final-test execution locks.

Minor conventions must be frozen globally before model results are inspected.

## Canonical initial HLT ParT baseline

Transfer Block 5 will implement and parity-test:

```text
particle features        17
pair relation            Weaver standard four
particle embed dims      [128, 512, 128]
pair embed dims          [64, 64, 64]
attention heads          8
particle blocks          8
class-attention blocks   2
activation               GELU
output classes           10
trim                     enabled
initialization           from scratch unless explicitly declared otherwise
```

The installed Weaver implementation is authoritative. FP32 logits, input and
parameter gradients, masks, and state dictionaries require a focused parity
test with mixed precision disabled.

## Training and execution semantics

- Default training uses a fixed declared epoch/update budget.
- A configured checkpoint selector may select an evaluation checkpoint but
  may not cancel other registered scientific rows.
- Nonfinite required quantities fail; they are not silently skipped.
- Exact resume restores model, optimizer, scheduler, scaler, sampler, replica
  cycle, epoch/update count, and random states.
- The v2 checkpoint also restores Weaver trimmer `enabled` and `_counter`
  runtime state, which is not carried by the model state dictionary.
- Every checkpoint binds its data, configuration, and source parents.

Performance-based termination, cancellation, and row skipping are forbidden.

The reusable baseline trainer freezes AdamW, update-based linear warm-up, and
cosine decay to 5% of peak learning rate. For positive total update count
`U`, the warm-up length is exactly
`min(U, max(1, floor(0.05 * U)))` unless a campaign freezes another fraction.
The first update uses `peak_lr / warmup_updates`; the final post-warm-up update
uses exactly `0.05 * peak_lr`. Very short runs therefore remain finite and
execute their complete declared budget.

The selected checkpoint minimizes model-validation cross-entropy, then
maximizes accuracy, then chooses the earliest optimizer update. Selection
floats are serialized with their exact hexadecimal representation. Selection
does not stop training.

## Metrics

The repository-wide v1 definitions are:

- accuracy uses lowest-index `argmax` ties; cross-entropy uses stable
  log-sum-exp;
- per-class efficiency is recall for each true class; one-versus-rest AUC is
  the Mann-Whitney statistic with exact average ranks for score ties;
- multiclass Brier is the mean over jets of the sum over ten classes of
  `(probability - one_hot)^2`;
- ECE is 15-bin top-label multiclass ECE with edges `i/15`; bins are
  `[i/15,(i+1)/15)` except the final bin includes `1.0`; empty bins have zero
  weight and contribute zero;
- QCD-versus-signal uses `1-P(QCD)`, accepts scores greater than or equal to
  the highest threshold attaining at least 50% signal efficiency, records the
  achieved efficiency after inclusive ties, and reports null rejection plus
  an explicit flag when no QCD row passes;
- paired bootstrap uses seed `8041`, the paired jet as the sampling unit,
  independently resamples within every class while preserving observed class
  counts, and uses NumPy's linear 2.5%/97.5% quantiles;
- metric absence caused by an empty required class is represented explicitly,
  never silently substituted with zero.

## Final-test isolation

Before locking, `stack_val` may contain authenticated deployable predictions
used by a selector. Prediction shards should not contain labels; selectors
join labels from the authenticated split manifest.

Before both finalist and execution locks, no `final_test` model-derived output
is permitted. Final evaluation runs exactly once for the locked graphs and
cannot reopen selection.

## Artifact semantics

Every reusable artifact has:

- contract name and schema version;
- canonical content hash;
- parent hashes;
- source snapshot;
- ordered identity or population binding where applicable;
- validation on every consumer path;
- atomic publication.

An all-reused restart still validates active source and campaign parents before
deciding that no worker is needed.

## Campaign execution

The reusable baseline campaign uses the versioned source snapshot, campaign
specification, storage measurement, task attestation, submission ledger,
monitor report, resume plan, finalist lock, and final-test execution lock
contracts. A completed Slurm state is reusable only when every attested file
still has its recorded hash. Recovery creates a fresh submission ledger for
the failed task and every descendant; it never releases stale dependencies.

Final-test inference atomically creates
`hlt_classification_final_test_execution_claim_v1` before its first model
forward. An existing claim with incomplete predictions forbids automatic
rerun; complete prediction reuse requires the exact matching claim. The
library inference and metric APIs enforce the same claim and
campaign/checkpoint/cache/source lineage, so bypassing the CLI does not bypass
the final-test lock.
