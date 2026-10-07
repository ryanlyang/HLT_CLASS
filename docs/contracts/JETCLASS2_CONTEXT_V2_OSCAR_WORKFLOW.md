# CONTEXT_V2 dataset and automatic classifier workflow

Authority: [implementation plan](../plans/JETCLASS2_CONTEXT_V2_OSCAR_WORKFLOW_PLAN.md).

New producer namespace: `JC2_CONTEXT_V2_{KIND}/v1`. Artifacts bind immutable V1
source manifest, original classifier selection, frozen recipe and source commit.
Only materialization receipts/manifests for test truthfully mark test access;
all artifacts mark final-test evaluation false. Ordinary consumers reject test.
No artifact may masquerade as a V1 producer or a CMS-calibrated response.

Shared ladder versions: RELEASE_REQUEST/RELEASE/FOUNDATION/VIEWS v5;
GATE_SPEC/RUNTIME_PROFILE v10; SCIENTIFIC_PLAN/CAMPAIGN_SPEC v7. INPUTS remains
v2 because the actual input transform is unchanged. Older versions are unchanged.

Output retains the existing lossless physical NPZ layout and canonical jet IDs.
Pairing uses the old authenticated release index with newly verified bank refs;
row order and train/validation role are exact, not a new random selection.
Tracking is V2(V1-inverse(saved V1)), with amplitude 1.0, three layers, raw
inverse scaled error <=1e-10. Nontracking arrays and invalid tracking rows are
byte-identical. The imported V1 inverse's file hash must match the historical
producer's source inventory before materialization. Float32 input reconstruction error is measured, not presented
as exact information conservation. No ancestry/keys are model features.

Workflow authorization covers generation, verification, fresh matching,
runtime preflight, and one 16-job DIRECT+COARSE submission within the fixed
policy. It does not authorize further datasets, strength searches, additional
branches, final-test evaluation, cancellation or replacing existing artifacts.
Every worker validates pinned source and exact scheduler identity. Publication
is atomic and immutable; incomplete and corrupted artifacts fail closed.
