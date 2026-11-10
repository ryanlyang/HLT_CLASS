# Oscar FullStat subclass student: contrastive coefficient 0.6

User-authorized one-factor experiment, 2026-11-10. Train one fresh HLT-only
student for 20 epochs on the existing 3M TRAIN / 500k development-validation
population. Change the OFFLINE mix contrastive coefficient from 0.3 to 0.6.
The subclass coefficient stays 0.5. Both ramp over the first three epochs,
with independent coefficients and the exact previous subclass ramp arithmetic.
CE remains 1; U100, CONCAT and FUSION embedding SmoothL1 coefficients remain
10/3 each. Temperature 0.1, candidate sampler, 1024 negatives, batch128,
model/predictor initialization, sampling seeds, native architecture, BF16/FP32
policy, optimizer, LR curve, gradient diagnostics and checkpoint selection
remain the completed subclass setup's settings. No new teachers or control fit.

The completed contrastive0.3/subclass0.5 campaign at commit
16546c0e0196ab85cd69e23de005dc5f6f29feb5, job8761518, is the parent. Authenticate
it using its original clean source and COMPLETED/0:0 accounting. Reuse the
same U100/CONCAT/FUSION embedding targets and OFFLINE matching keys. Copy the
parent's subclass manifest, probabilities and prototypes byte for byte; do
not recluster, recalibrate or warm-start from the parent's student. The
classical v1 target-bank contract remains unchanged. QCD10/signal5 subclass
counts, TRAIN-only 75/25 embedding/margin geometry and calibrated softness
remain exactly as in the [original plan](LUKA_FULLSTAT_SUBCLASS_PLAN.md).

Use a fresh result root, separate source namespace and
`LUKA_FULLSTAT_SUBCLASS_C06_*/v1` execution artifacts. No older recipe, source,
result, queue or claim is edited. The existing 0.3-only and 0.3+subclass
results are historical references; neither is automatically retrained.
Final-test access remains sealed. All auxiliary heads are removed for the
same native HLT-only deployment graph.

One Oscar L40S/6CPU/192000MiB/48h allocation verifies targets, runs the full
objective's installed-Weaver/sampler/deployment/gradient checks, measures
resources, discards probe weights, then starts the scientific fit. This is
the same integrated acceptance policy as the parent, with no separate
preflight or setup job. The active Oscar plan supersedes legacy Tigris-only
acceptance language for this campaign. Local tests are not real Oscar data
acceptance. Poor finite scientific metrics never fail or skip the experiment.

Save complete model/optimizer/RNG state each epoch. Explicit same-source,
same-recipe recovery may restore that C06 fit only; a C03 checkpoint is not a
valid C06 resume. With all20 epochs saved, recovery only finalizes publication
after integrated checks. No automatic retry or ambiguous re-submission.

Report AUC, accuracy and R30/R50 at the same AUC-selected checkpoint, alongside
M0HLT, OFFLINE, U100-w5, Multi3, Multi3+contrastive0.3 and subclass+C0.3.
Disclose historical single-seed development comparisons. Numerical improvement
does not establish statistical significance or guarantee improved Hbb rejection.
