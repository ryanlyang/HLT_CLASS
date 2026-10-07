# CORR_MID SPORC 100k/50k direct/coarse contract

Authority: [active plan](../plans/JETCLASS2_CORRELATED_TRACKING_PRODUCTION_PLAN.md).

Shared execution namespace JETCLASS2_CMS_PROXY_LADDER remains tooling only:
RELEASE_REQUEST/RELEASE/FOUNDATION/VIEWS v4, INPUTS v3, GATE_SPEC/RUNTIME_PROFILE
v9, SCIENTIFIC_PLAN/CAMPAIGN_SPEC v5 explicitly identify CORR_MID. Old schemas
are unchanged. SOURCE hashes pin all new/reused implementation and worker files.

Consume only completed CORR_MID production metadata and ordinary-role blocks.
Freeze 100k train/50k validation by label-blind hash rank over selection domain,
original study hash, role and canonical jet identity. No final-test blocks are
opened for training, feature normalization, view selection, or acceptance.
Labels are read only for the selected ordinary entries. No split override.

Verify exact original row-wise p4/charge/category/mask equality before pairing;
persist identity mappings, not a fitted matcher. Reject any mismatch. D066,
D033 and D000 mean retained offline fractions 2/3, 1/3 and 0. Let f be added
noise strength (1/3, 2/3, 1): values = offline + f*(proxy-offline), error^2 =
offline_error^2 + f^2*(proxy_error^2-offline_error^2). D000 and OFFLINE are exact
endpoints, not rounded interpolation. P4/PID/masks never interpolate or change.
Intermediate views require paired offline only during declared teacher/student
training and oracle validation. Final deployable models consume D000 alone.

Common 17-input unclipped asinh/log1p transform applies to every arm. Six fresh
fits: M0HLT, OFFLINE, direct D000, coarse D066/D033/D000. Three probability
reducers: OFFLINE, coarse D066 and coarse D033; aggregate and completion give
11 science jobs. No ascending duplicate, dense branch or extra screening fit.
Same canonical ParT/seed/sampler/CE25-KL75(T=2, exactly one T^2)/schedule as
the prior ladder; no weight continuation. Report accuracy/AUC/QCD rejection
@50% with M0HLT=0% and pure OFFLINE=100% recovery; poor results remain rows.

SPORC debug, one A100 per train/reducer, 6 CPUs, 90000 MiB. Gate authenticates
data, builds full selected foundation then measures real Weaver forward/gradient
parity, one CE and one KD pass, full selected cache, memory and timing. Measured
train/reduce requests must fit debug's 24h limit. No profile-only transfer from
Tigris/Oscar. Acceptance fits are not reusable scientific checkpoints.

Exact dry/live gate and science plans, source-clean pushed commit, durable
submission claim and sanitized Slurm variables are required. No automatic
cancellation or silent submission of another campaign. Test inference sealed.
