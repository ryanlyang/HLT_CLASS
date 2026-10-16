# Matched batch128 contracts

Authority: [active plan](../plans/LUKA_ALLJETS_BATCH128_PLAN.md).

For the specific completed-probe/sampler failure in job21898881, see
[report-only execution v5](LUKA_BATCH128_REPORT_RECOVERY.md). It preserves this
recipe and original v4 evidence, without marking the failed GPU job successful.

`LUKA_MATCHED_BATCH128_RECIPE/v1` binds the original recipe hash, changing only
batch_size to128 and explicitly inference_batch_size=128, accumulation=1.
All other schedule/loss/selection/optimizer fields are unchanged. The runner
accepts only this exact new artifact; it rejects modified batch256 artifacts
and arbitrary batch sizes. Kernel reports bind the new recipe hash.

`LUKA_ALLJETS_FUSION_{GATE,SPEC,PREFLIGHT,PROBE,PLAN,TASK,RESULTS,DIAGNOSTICS,SUBMISSION_CLAIM}/v4`
distinguish this matched scientific rerun from v1--v3. Counts and exploratory
flags are unchanged: no independent test. Graph17 fits/11 reducers/30 jobs,
no imported weights, banks or comparison rows. Provenance may reference the
original parent but its science outputs are never used as targets or results.
U000 is a local scientific dependency for DIRECT, COARSE_U050 and FUSION_U050.
Technical probe receipts cannot satisfy any scientific edge.

GATE binds the failed21898735 original5cab1830 gate, source/journal/runtime/
failure/log and exact source-difference allowlist. PREPARED/cache contracts
remain unchanged/read-only. Fresh output must not overlap any source, raw,
prepared or prior campaign root. Current source must be clean, committed and
pushed before queueing; original-source validators run in their own worktrees.

Preflight policy `isolated_matched_batch128_v1` registers four probe families:
D000/D000, U050/U000, singleD000 and singleU000. Separate parity/stress/measure
children (12 processes); strict both-precision offload parity, all full128
three-update stresses before any full-population pass, independent longest
validation inference stress. Uniform11-class technical KD (U000 CE-only), no
old bank. Support maxima bound all science families separately for each role.
Only the aggregate can admit: exact child PIDs/exits/source/runtime/log/payload
hashes, complete populations, checkpoint/bank readback, headroom and full
100-pass/1.75 runtime projection <=1440min. Failure preserves evidence only.

SPEC requires v4 admission and its exact journal plus COMPLETED/0:0 accounting.
Every new scientific fit/report/selected checkpoint/prediction and TRAIN bank
is linked through v4 task dependencies and the new recipe hash. Training and
all inference use128; no accumulation, truncation, warm-start or model change.
Full74895 validation selects; original50k merely reports the same checkpoint.
Missing result rows remain unfinished; quality never controls completion.

Gate and30-job science plans require separate dry review and
`AUTHORIZE LUKA MATCHED BATCH128 EXACT PLAN` with exact saved plan hash.
Old plans/receipts cannot authorize new jobs. No automatic followup or retry.
