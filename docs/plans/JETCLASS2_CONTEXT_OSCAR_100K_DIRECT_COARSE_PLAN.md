# Frozen CONTEXT_V1: Oscar 100k/50k direct and coarse comparison

## Authority and scientific scope

The user selects Oscar after comparing current GPU queue estimates and supplies
a successful transfer checksum audit: manifest
`2ec2c933a37f50bb85e83577b9c1cf9d31c35a6e0be42089f0088ae62c485efb`,
2,341 blocks, 210 shards and 186 offline files. This is user-reported transport
evidence, not installed-reader or GPU acceptance. No existing campaign changes.

Register 100,000 train and 50,000 validation jets by label-blind SHA256 rank
over a versioned domain, original study hash, role and canonical jet identity.
Restore physical file/block order after selection. Membership is fixed before
labels are read; no balancing or label-based row filtering is added. Test may
be present on disk but neither selection nor classification opens test receipts,
NPZ banks or ROOT files. Original metadata paths/hashes are never rewritten.

The dataset is an intentionally synthetic, context-encoded benchmark, not
CMS HLT. Reuse its frozen physical bytes, recipe, random draw and count rates;
do not regenerate, invert the context map for the classifier, or tune anything.

## Frozen comparison

Three fresh CE controls: M0HLT, pure OFFLINE, U000. DIRECT: U000 -> D000.
COARSE: U000 -> U050 -> U100 -> D066 -> D033 -> D000. No dense branch or imported
models. Nine fits, five probability reducers, aggregate and completion: 16 jobs.
The immediate parent alone supervises each KD student: C25/P75 forward KL,
T=2, one T-squared factor. Preserve the shared batch-256 AdamW/BF16-forward,
FP32-loss recipe, coordinate-paired initialization/sampling, 3-pass warmup,
hold through 45, decay through 60, floor patience 15, minimum 60/maximum 100.
Validation selects checkpoints; accuracy/AUC and classwise QCD rejection at
50% are reported. Recovery is M0HLT=0%, pure OFFLINE=100%; U000 is separate.
Poor metrics are results and never suppress descendants.

Recompute SALIENCE_PT_LINEAR full-cardinality assignments on the new endpoints,
saturating the smaller side, never using generator ancestry. Preserve the
persistent proxy skeleton, offline-only tail and existing U/D interpolation.
D000 is exactly the saved context endpoint. Use the SAME context asinh/log1p
17-channel frontend for all controls, teachers and every U/D view. No tanh or
clipping of tracking, no extra context, inverse, identity or ancestry channels.
The transformed error fields are synthetic encoded scales, not calibrated
uncertainties. Units stay GeV and native Delphes mm.

The earlier pilot plan calls for LOW_NOISE-vs-CONTEXT differences of ladder
gains to isolate the context mechanism. This user-requested campaign is the
CONTEXT arm only; it tests direct versus coarse here, not that causal contrast.
It must not be compared to historical NOISE_V3 scores as an isolated mechanism
effect. A matched LOW_NOISE campaign requires separate registration.

## Execution and admission

Oscar only: gpu partition, default account, norm-gpu QOS, one L40S per GPU job,
six CPUs/workers, 90,000 MiB. This preserves the existing two-slot request shape
(two jobs total 12 CPUs, 180,000 MiB, two GPUs); it does not promise scheduler
concurrency. CPU gate jobs request no GPU. Authenticate/select ordinary data,
construct fresh assignments, then perform a genuine full-selected-population
preflight on Oscar using the production cache/training/prediction path.

Preflight requires installed-Weaver FP32 output and gradient parity, one fresh
CE pass and a KD backward smoke with identity-bound teacher probabilities,
finite results, U000/D050 cache measurements, GPU/environment identity and RAM
bounds. Science uses conservative measured limits, never guesses from CPU
generation time or another site's GPU. Training ceiling 48h, reducers 24h;
never shorten training to fit a queue estimate. No test evaluation is enabled.

Gate and science are separate dry-first exact plans with clean pushed source,
hash-bound artifacts, sanitized Slurm environment, site feasibility checks,
durable exclusive submission claim and exact-ID journals. No automatic followup,
cancellations, overwriting, deletion or remote submissions in local implementation.
Source/data corruption and scope drift fail closed. New source requires its own
Oscar GPU acceptance (the selected-site counterpart to the generic Tigris rule).

## Paths

Dataset container:
`/oscar/home/rlyang/datasets/literature_context_v1_2250k_4f473c3e_r1`

Proxy root is its `jetclass2_literature_context_v1_2250k_4f473c3e_r1` child;
direct provenance is its `oscar_provenance_v1` child. Reuse offline under
`/oscar/home/rlyang/datasets/literature_noise_v3_2250k_ebd5bc1a_r1/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2`.
Profiles resolve under that offline snapshot's parent; authenticate every
reference and pairing using inventory hash + relative file + tree + raw entry.
New gate, matching, checkpoints and logs live outside both source copies.

## Operator sequence

First commit and push only the reviewed implementation and its required source
dependencies. On Oscar, fetch it and create a clean detached worktree at that
exact 40-character commit. Do not use the original generator commit as the new
classifier implementation commit. Do not delete or rewrite either dataset copy.

With `COMMIT`, `PROJECT`, and a fresh sibling `ROOT` set explicitly:

```bash
DATASET=/oscar/home/rlyang/datasets/literature_context_v1_2250k_4f473c3e_r1
bash "${PROJECT}/scripts/queue_jetclass2_context_ladder.sh" \
    "${COMMIT}" "${DATASET}" "${ROOT}" gate
```

Review the three commands and the printed plan `content_hash`. Repeat the same
command with `--execute "${GATE_PLAN_HASH}"`, where that variable contains the
reviewed 64-character hash. The helper reuses the existing dry gate; never rerun
creation at a different root to recover an ambiguous submission.

After gate success (`gate/gate_complete.json`, authenticated by the CLI):

```bash
bash "${PROJECT}/scripts/queue_jetclass2_context_ladder.sh" \
    "${COMMIT}" "${DATASET}" "${ROOT}" science
```

Review the 16-job science plan and its **different** hash. Repeat with
`--execute "${SCIENCE_PLAN_HASH}"`. The whole science DAG is submitted together;
the scheduler resolves the five teacher dependencies. Source validation and
metadata checks may take minutes before jobs appear. For unreliable SSH, run
each helper under `nohup` with a unique log, check its exit code/ledger, and do
not start a second live helper while the first is active.

The shared environment helper initializes Python; it must find Oscar's existing
`atlas_kd_oscar` environment. Live submission uses read-only scheduler shape
tests before submitting anything. If an account, QoS or partition request is
rejected, stop and inspect the error; do not silently migrate the campaign.

Saved validation results, after activating that same environment and setting
`PYTHONPATH=${PROJECT}/src`:

```bash
python -s "${PROJECT}/scripts/jetclass2_context_ladder.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

This reports incomplete nodes alongside available rows; it does not retrain or
evaluate test data. Recovery is referenced to this campaign's fresh OFFLINE and
M0HLT controls, never to an older NOISE_V3 or CMS-proxy result.
