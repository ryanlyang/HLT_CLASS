# New CERN FullSim dataset: natural-sampling OFFLINE baseline on Hydra

Authorized 2026-11-13. This additive campaign uses the new QCD Run-2-combined
and Higgs samples, not the original Oscar FullStat snapshot. It trains exactly
one fresh pure-OFFLINE oracle. No balancing or ladder fits are released here.

## Population

Use the downloaded, CERN-SHA256-authenticated 6,990 QCD files and six signal
files under `luka_qcdrun2com_higgs2p_sb_v1`. The source manifest SHA256 is
`80f5bc70db305272632ebd22537a2bd3d286a6e552bde620d2f796ee70e21097`.
The local audit found 7,296,856 raw rows and 5,735,950 eligible recognized-class
nonempty OFFLINE/HLT pairs. Take QCD only from the dedicated QCD source and
recognized signals only from the signal source. Apply no additional pT cut.

The user selected this 5,735,950-row pool after the label-206 audit. Explicitly
quarantine 206 without giving it a class meaning; preserve its raw files and
counts. This supersedes the earlier diagnostic-only exclusion for this specific
known-label baseline, not the global label map. Any other unknown code fails.
Producer confirmation of label 206 and unchanged recognized-code semantics is
still outstanding and must accompany interpretation of the baseline.

Freeze 3,000,000 TRAIN, 500,000 validation, 1,500,000 sealed final test, and
735,950 unused. Reuse the original integer proportional quotas over signal
class / QCD source bin, shared 2048-entry block hash ordering and split seed
20261016. Use a new dataset/identity domain. The paired views share membership.
These are row-disjoint splits; event independence and old-dataset overlap are
unverified because event identifiers are absent. This limitation remains in
every report. Scalar inspection before splitting is allowed; test and unused
particle arrays are never requested. Unknown rows are also excluded from
prepared particles. No final-test evaluator is provided.

## Scientific recipe

Use the original OFFLINE node, native 17-input / eleven-output ParT, full
particle support up to 512 (overflow fails, no trimming), and the unchanged
physical/input kernels. Carry the explicitly provisional mm and error-only
zero-uncertainty conventions from Oscar; do not assert producer confirmation.

Twenty complete natural, independently shuffled without-replacement TRAIN
epochs; batch 128 including the final batch of 64; no gradient accumulation,
class weights, pT weights, embeddings, KD or contrastive loss. CE coefficient
is one. AdamW: betas (0.9, 0.999), eps 1e-8, weight decay 0.01 with the original
class-token exception. Linear two-epoch warmup to 3e-4, cosine to 1.5e-5 at
epoch 20, evaluated at update endpoints. BF16 CUDA forward and FP32 loss.
Reuse original OFFLINE initialization and sampler seeds. Validate all 500k
jets every epoch; select maximum macro AUC, then minimum CE, maximum macro
log R50, earliest update. Save accuracy, AUC, R50, selected predictions and
derive R30 from those same predictions. No quality threshold aborts training.

Save full model/optimizer/RNG at every completed epoch, with a completion-last
receipt, so an interrupted run does not require retraining completed epochs.
Continuation must authenticate source, recipe, population and committed state.

## Transfer and execution

Brown CS Hydra, not Oscar. Transfer raw files plus the frozen scalar foundation
over the user's authenticated SSH route `rlyang@ssh.cs.brown.edu`. Stream a
verified archive directly into a new dataset directory; do not require a
second raw archive copy on Brown. Never overwrite an unrelated destination.
Transfer all source files including unknown/excluded rows. No credentials are
stored in source or reports. Source code travels by a clean pushed Git commit.

Prefer one L40S (matching original Oscar hardware); one L40 or A6000 is an
explicit registration alternative. No automatic multi-GPU/FP32/batch change.
Use current scheduler evidence to compare queue estimates. Prepare only native
OFFLINE TRAIN/validation caches on CPU; preserve raw paired data for future
HLT and balanced campaigns. Reserve disk and RAM using scalar particle counts.

Dry review covers CPU preparation and one GPU allocation. The GPU allocation
must run installed-Weaver FP32 parity, real TRAIN/validation miniature,
full-support stress, reload and measured memory/time checks before training
from a fresh initialization. This Hydra miniature replaces the generic Tigris
acceptance site for this Hydra-only adapter. Local tests cannot establish it.
The user's request authorizes this single baseline after the reviewed checks;
it does not authorize unrelated jobs or cancellation. Submission records exact
IDs and fails on ambiguous retries. Test inference stays sealed.

## Later balanced campaign

Keep this same split. Recompute the 200 <= offline pT < 2000 GeV TRAIN
occupancy and eligible draw count; never copy 2,425,536 from the old snapshot
or use the 4,126,225 whole-pool in-range count as a TRAIN budget. No balancing
choice changes this baseline. A new natural HLT reference will be needed to
calculate new-dataset HLT-to-OFFLINE recovery; old anchors are not transferable.
