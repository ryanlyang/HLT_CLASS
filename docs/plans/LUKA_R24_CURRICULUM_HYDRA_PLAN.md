# R24 curriculum follow-up: twenty four-GPU fits

Active scientific plan, 2026-11-13. This is a new execution family; it does not
modify the completed R8/R24 campaign or its benchmark. Hydra is the authorized
site, superseding the generic Tigris execution requirement for this campaign.

The question is whether delaying reconstruction, preparing its decoder, or
changing its ramp makes the observed R24 benefit repeatable and stronger.
Historical R24 selected epoch 22 at validation AUC 0.96275789 (55.4% recovery).
That single-seed result motivates this experiment; it is not a guaranteed gain.

## Fixed comparison

All twenty fits use the same authenticated 299234 TRAIN and 74895 validation
HLT D000 jets, batch 128, original 30-epoch AdamW/LR recipe, CE weight 1 and U100
embedding SmoothL1 weight 5. No new teacher, data export, optimizer reset, or
extra epochs. Native attention, readout and deployable parameter count stay
unchanged. Final-test access remains sealed; the historical original50k subset
overlaps development validation and is not independent evidence.

Reconstruction uses complete OFFLINE TRAIN particles, the original TRAIN-only
normalization/frame, fixed output support (298 in the original execution),
four width-256 decoder blocks, eight heads, FF1024, chunk8, exact assignment,
equal field weights, global embedding FiLM and live block-eight HLT tokens.
Targets never enter the encoder or validation forward pass. The decoder and
embedding predictor are removed from the selected deployment checkpoint.

Four GTX1080Ti GPUs on one Hydra node run FP32, TF32 disabled, synchronized
BatchNorm and sample-weighted gradient SUM. Padding is computed for each global
batch before sharding. Global batch/LR do not grow with GPU count. Independent
dropout streams vary by rank; therefore fresh matching controls are mandatory.

## Registered fits

| Task | Seed | Delay | Ramp | Shape | Maximum reconstruction weight | Adjustment |
|---|---:|---:|---:|---|---:|---|
| D01 | 0 | — | — | — | 0 | U100-w5 control |
| D02 | 0 | 3 | 3 | linear | .3 | R24 reference |
| D03 | 1 | — | — | — | 0 | U100-w5 control |
| D04 | 1 | 3 | 3 | linear | .3 | R24 repeat |
| D05 | 0 | 0 | 3 | linear | .3 | R8 timing reference |
| D06 | 0 | 2 | 3 | linear | .3 | Earlier start |
| D07 | 0 | 5 | 3 | linear | .3 | Later start |
| D08 | 0 | 8 | 3 | linear | .3 | Long delay |
| D09 | 0 | 3 | 1 | linear | .3 | Fast introduction |
| D10 | 0 | 3 | 6 | linear | .3 | Gentle introduction |
| D11 | 0 | 3 | 10 | linear | .3 | Long introduction |
| D12 | 0 | 3 | 3 | half-cosine | .3 | Smooth introduction |
| D13 | 0 | 3 | 6 | half-cosine | .3 | Gentle smooth introduction |
| D14 | 0 | 3 | 3 | linear | .15 | Lower weight |
| D15 | 0 | 3 | 3 | linear | .5 | Higher weight |
| D16 | 0 | 3 | 3 | linear | 1 | Strong weight |
| D17 | 0 | 3 | 3 | linear | .3 | Fade .3 to .1 over positions 18–30 |
| D18 | 0 | 3 | 3 | linear | .3 | Decoder prepared during delay |
| D19 | 0 | 5 | 3 | linear | .3 | Longer decoder preparation |
| D20 | 0 | 3 | 6 | half-cosine | .3 | Prepared decoder, smooth introduction |

Let p be update/global-steps-per-epoch and u=clip((p-delay)/ramp,0,1).
Linear weight is maximum*u; cosine weight is maximum*(1-cos(pi*u))/2.
Delay3/ramp3 is exactly zero through p=3 and full weight at p=6.

D18–D20 train the decoder at .3 from the first update. Its inputs are detached
for decoder backpropagation. Only the returned global/token condition gradients
are multiplied by encoder_weight/.3 before the native backward pass. Thus the
native model receives only CE+U100 during the delay while the decoder learns.
This intentionally asymmetric gradient policy is not described as one scalar
joint objective. Other arms use identical encoder/decoder weights; zero-weight
decoder updates preserve original R24 optimizer semantics.

## Qualification, release and interpretation

The completed 1094dea4 four-GPU benchmark is required input: parity and full
support passed, with ~12.04h projected total versus 10.98h observed A6000 elapsed.
This is evidence for feasibility, not a full production qualification.

New setup must run the actual four-rank worker: native factory parity, serial
versus distributed update/AdamW parity including uneven batches and warmup,
control identity and no native reconstruction gradient during warmup, genuine
two-epoch fits/reload, full-validation replay, maximum HLT/OFFLINE support,
partial batches, and measured time/aggregate host and per-card memory. It must
cover all twenty cells, including active weights after delay. Poor finite AUC
or reconstruction quality never blocks a registered row. Numerical, lineage,
memory or resource failures do block release, with diagnostics preserved.

Bootstrap queues setup and a durable CPU `afterok` release job. Release queues
twenty independent fits in order D02,D01,D18,D13,D07,D15,D04,D03,D20,D05,D06,D08,
D09,D10,D11,D12,D14,D16,D17,D19, then a CPU `afterany` report. Queue order encourages
mixed early results; Slurm decides actual start order. No scientific fit depends
on another fit and no winner is required to release later arms.

Every fit requests 4 typed GPUs, 12 CPUs and 64000 MiB. Setup is bounded to 6h;
each fit to 48h. Measured requests include full validation, metrics, diagnostics,
checkpoint/reload and startup with a 2x margin. Output allowance is 8 GiB plus
1 GiB headroom. Exact pushed source, immutable plans, full dry review, accepted
job journals and no automatic retry protect against duplicate submission.

Primary comparison is selected development-validation AUC against D01 or D03
with the same seed and four-GPU FP32 execution. Also report same-epoch contrast,
losses, separate encoder/decoder weights and condition-gradient diagnostics.
Historical recovery is a useful scale, not an independent test or uncertainty
estimate. Replicate R24 first; treat selecting among the other arms as exploration.
