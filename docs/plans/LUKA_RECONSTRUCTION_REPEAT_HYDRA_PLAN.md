# Reconstruction repeatability: eighteen four-GPU fits

Active scientific plan, 2026-11-14. The user authorized implementing and
queueing preparation for this campaign on Hydra with four GTX1080 Ti cards
per fit. This plan supersedes generic Tigris site requirements for this
execution. Existing campaigns, source snapshots and artifacts are unchanged.

## Scientific comparison

Train six recipes at each of three predeclared new seed indices **4, 5, 6**:

| Recipe | Reconstruction schedule and conditioning |
|---|---|
| C | CE + U100 embedding SmoothL1 weight5; no decoder |
| R24 | Maximum .3, zero through position3, linear ramp to position6, block8 tokens |
| R20 | Maximum2, linear ramp over positions0–3, block8 tokens |
| R22 | Maximum10, linear ramp over positions0–3, block8 tokens |
| R28 | Maximum .3, linear ramp over positions0–3, concatenated block4+block8 tokens |
| D20 | Decoder weight .3 from first update; encoder zero through position3, half-cosine ramp to .3 at position9; block8 tokens |

Task IDs are `<recipe>_s<seed>`, for example `R22_s4`. Each recipe retains
CE1 + U100-w5 throughout. R20/R22/R24/R28 are the original 58d0c0ed scientific
recipes; D20 is the a549fef1 asymmetric decoder-warm curriculum. No gradient
clipping, adaptive coefficient, optimizer reset, new teacher, extra epochs or
new architecture is introduced. D20 uses separately declared decoder and
encoder gradients, not a single scalar weighted objective.

All fits start from scratch on the same authenticated 299234 TRAIN and 74895
development-validation HLT D000 jets, global batch128 and 30 epochs, with the
original AdamW/learning-rate schedule and original selection key (AUC, CE,
rejection, earliest update). OFFLINE particles are complete TRAIN targets only;
all validation and deployment use native HLT inputs. U100 predictor and decoder
are removed from the native deployment checkpoint. No final-test access.
Historical original50k overlaps validation and is not an independent test.

Seed indices4/5/6 were chosen before this campaign's outcomes. Native
initialization and training permutation seeds are the imported base plus
1000003 times the index. Dropout resets after model construction to the native
seed plus rank. Predictor initialization retains the original isolated rule;
decoder initialization uses its original isolated rule with the new index.
All six fits in a seed group share native/predictor initialization, sampler
and dropout seed identities. Decoder parameters are identical across the five
reconstruction recipes in a seed group (R28 adds memory tokens, not parameters).
Initial component and order hashes are saved and paired identities validated.

## Execution and qualification

All fits use one node, four typed `gpu:geforce_gtx_1080_ti:4`, 12 CPUs and
64000 MiB RAM. Native and decoder computation are FP32, TF32 disabled,
global padding, SyncBatchNorm and sample-weighted gradient SUM. The new driver
uses the immutable 58d0c0ed scientific source beside its own isolated runtime.
The completed 1094dea4 benchmark provides parent feasibility evidence, not
acceptance for the new weight10/two-depth paths. No new data transfer is needed.

Setup performs serial versus four-rank **common-start** update comparisons for
all six recipes at seed4, positions3,3.01,4.5,9 and alternating global16/15
rows. Each comparison restores the complete serial pre-update model and AdamW
state. It uses the diagnostic-v2 tolerances: loss absolute1e-3, gradient and
parameter and moment relative L2 5e-4, buffer absolute1e-4. Named buffers,
integer counters and rank synchronization are checked. Dropout is disabled
only during numerical qualification. This does not require independently
evolving trajectories to remain identical, nor claim bitwise 30-epoch replay.

Setup also checks installed-Weaver/native execution and historical full
validation replay, runs and reloads genuine production miniatures for every
registered fit, measures each fit at full reconstruction weight with twelve
warmed updates, full validation/metrics and probes, and stresses the longest
HLT batch, largest OFFLINE targets, partial11 and the true final batch. R28's
two memory depths and maximum weight10 must qualify explicitly. D20's zero
encoder/active decoder phase is covered by numerical qualification. Each
science allocation repeats a startup miniature and discards its model,
optimizer and RNG state before fresh science initialization.

Use a 6h setup and 48h hard fit envelope. Measured requests use 2x estimated
runtime plus30 minutes, including validation, startup, diagnostics and export.
Require at least13 personally available GiB:12GiB output budget plus1GiB reserve.
Poor finite accuracy or recovery never blocks setup, a fit, or later fits.
Invalid lineage, nonfinite execution, numerical or resource violations fail
closed with evidence retained. No automated retries or GPU fallback.

## Automatic queue and reporting

Bootstrap submits `setup_pascal` and a durable CPU `release_pascal` with
`afterok` on the exact setup ID. Successful setup releases all18 fits without
another prompt or an interactive process. A CPU `report_pascal` waits
`afterany` on all18 IDs. Total:21 jobs. The preauthorized resource envelope,
clean pushed source, scheduler dry review and exact-ID journals are mandatory.

Queue order is seed4 then seed5 then seed6, each ordered
`C, R24, R20, R22, R28, D20`. All scientific fits are independent; submission
order encourages early coverage of every recipe but cannot guarantee starts.

Results and progress authenticate saved outputs without inference. Every run
reports selected AUC/recovery and fresh same-seed control delta. Progress adds
same-epoch comparisons. Per-recipe summaries list each seed, mean and minimum
selected recovery, mean matched AUC delta, and completion/matched-control
coverage. Never silently replace a missing control with a historical one or
claim a three-seed average from two completed seeds.

Stability summaries describe individual checkpoints, not an ensemble: last
epoch, mean recovery over fixed epochs21–30, number of those epochs at or above
52%, and longest consecutive streak at or above52% in that window. An
incomplete window is explicitly unavailable, not averaged over fewer epochs.
52% is a research target, not an execution gate or guaranteed outcome. Three
seeds screen repeatability; the repeatedly used validation set does not become
an independent test. No validation-based choice of seeds or training schedules.

## Readiness boundary

Focused local contracts, actual model/loss tests, distributed surrogate tests,
installed-Weaver CUDA miniatures and dry submission tests establish readiness
to queue setup. Actual Hydra four-card qualification and measured resources
remain mandatory before automatic scientific release. Implementation does not
commit, push, submit jobs or alter any existing campaign.
