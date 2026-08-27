# HCWDL TRI60 D000 Optimization-Budget Screen

## Objective

The original full-data `LOGIT_D000_from_D033E` fit selected pass 60 of 60.
That is evidence that its predeclared 60-pass cosine schedule may have stopped
before the KD edge finished improving, but it does not by itself justify making
180--200 passes the default training budget.  This standalone screen asks a
narrower and more reviewer-friendly question:

> Can a better 60-pass schedule, or a modest 90-pass compromise, recover the
> apparent long-horizon benefit without tripling the epoch budget?

The previously defined 180-pass fit remains an upper-bound diagnostic.  It is
not one of the screen conditions and does not control screen completion.

## Immutable source boundary

Every new fit reuses the completed original TRI60 source campaign only through:

- the exact full-data foundation and train/validation population;
- the persisted `LOGIT_D033E` temperature-2 probability bank;
- the original `LOGIT_D000_from_D033E` report as an imported comparison row;
- the original replicate seed and `D000<-D033E` seed alias; and
- the installed production evidence and endpoint resource lock.

The source campaign need not be globally complete.  There is no scheduler
dependency on it, no source artifact is modified, and no teacher bank or view
cache is copied into the screen root.

## Controlled invariants

All 17 new fits use:

- all authenticated mapped training and validation rows;
- exact HLT inputs at coordinate `D000`;
- fresh initialization with the original paired seed alias;
- batch size 256 and unweighted CE;
- temperature-2 forward probability KD from `LOGIT_D033E`;
- final loss weights C25P75;
- validation after every pass and the established macro-AUC selector;
- no final-test access, no rolling resume, and restart-from-zero semantics.

Only the declared optimization schedule differs.

## Condition registry

The imported reference is the original 60-pass, peak-`3e-4`, 5%-warmup cosine
fit with a 5% LR floor and constant C25P75.

### Delayed-decay and floor axis

`Hxx` means three warmup passes, a peak-LR hold through pass `xx`, and cosine
decay over the remaining passes to the normal 5% floor.

- `P60_H20_LR3E4`
- `P60_H30_LR3E4`
- `P60_H40_LR3E4`
- `P60_H45_LR3E4`
- `P60_STD_F20_LR3E4` (standard cosine, but a 20% terminal LR floor)

### Peak-LR axis

These use the pass-30 hold schedule and vary only peak LR:

- `P60_H30_LR1P5E4`
- `P60_H30_LR2E4`
- `P60_H30_LR4E4`
- `P60_H30_LR5E4`

Together with `P60_H30_LR3E4`, this is a five-point LR comparison.

### Early stronger-KD axis

These use peak `3e-4` and the pass-30 hold schedule.  The prefix loss applies
through the named pass, then switches exactly to C25P75 for the rest of the
fit:

- `P60_H30_KD90_P05`: C10P90 through pass 5;
- `P60_H30_KD90_P10`: C10P90 through pass 10;
- `P60_H30_KD90_P20`: C10P90 through pass 20;
- `P60_H30_KD95_P10`: C05P95 through pass 10.

This tests whether stronger early imitation can establish the useful teacher
basin while later CE restores endpoint adaptation.  It is not an undeclared
dynamic or validation-adaptive loss.

### Ninety-pass compromise axis

- `P90_STD_LR3E4`: the original fractional warmup/cosine schedule at 90 passes;
- `P90_H45_LR3E4`;
- `P90_H60_LR3E4`;
- `P90_H45_KD90_P15`: C10P90 through pass 15, then C25P75.

The 90-pass rows record validation landmarks at passes 20, 40, 60, 75, and 90;
the 60-pass rows record passes 20, 40, and 60.  All rows are nevertheless
selected globally over their complete predeclared horizon.

## Execution design

One authentication job and one storage preflight precede 17 mutually
independent GPU fits.  All fits can run in parallel.  Aggregation waits for all
17, ranks every new and imported row by the established validation selector,
and publishes the top five as diagnostics only.  Results never submit or skip
additional jobs.

Each fit requests one GH200, 72 CPUs, 320 GiB RAM, and three days.  The large
CPU/RAM request supports the established parallel, RAM-resident full-population
preprocessing.  Durable output is limited to the selected/final checkpoint,
report, aggregate, attestations, and logs.

No standalone smoke campaign is required.  The screen reuses the installed
TRI60 parity, production-miniature, and resource evidence already authenticated
by the source campaign.

## Interpretation

A useful practical protocol should beat the original D000 result reproducibly
within 60 or 90 passes and should not merely exchange AUC for a severe rejection
loss.  The screen is exploratory and single-seed: its winner is a candidate for
a matched confirmation, not automatic evidence that the optimizer is generally
superior.  If only the 180-pass upper bound improves, a later headline use must
include an equal-budget CE control.
