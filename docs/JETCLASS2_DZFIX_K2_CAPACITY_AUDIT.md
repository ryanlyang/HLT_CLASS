# Dz-fix JetClass2 K=2 capacity audit

Date: 2026-09-21. Status: completed local count and best-case pT-loss census. This is a
read-only scientific diagnostic, not a matching foundation, classifier result,
or production-resource acceptance. No Slurm jobs were submitted or changed.

## Main result

Of **6,858,920 selected jets**, **32,452 (0.473136%)** satisfy
`N_offline > 2*N_HLT`. Therefore **99.526864% fit K=2 without cropping**.

This supports K=2 as a high-coverage choice for the proposed
[fixed-slot concatenation ladder](plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md),
but not a claim that cropping is rare in every class. The dimuon class has
11.8598% overflow and the hadronic-tau/muon class has 6.5912% overflow. The
overall rate is dominated by the much larger QCD population.

## Population and test protection

The confirmed local dataset is:

```text
C:/Users/22rya/ComputerScience/CERN/data/jetclass2_10M_20260918_dzfix/jetclass2
```

Its authenticated parent inventory contains 331 ROOT files and 9,350,551
selected rows. The active SPORC transfer contains 186 of those files and
5,283,833 selected rows. The existing local full-snapshot split and the
SPORC partial-snapshot split do not assign every shared file the same role.

The audit conservatively excluded the UNION of final-test files from both
registries: **97 files, containing 2,491,631 selected rows**. It read all
selected rows in the remaining **234 files**, not a convenience sample.
Protected ROOT files were not opened for this study; their exclusions were
derived from existing authenticated metadata. Final test remains sealed.

The SPORC inventory and file split were reconstructed from the existing local
partial-transfer plan and required to match its exact projected content hashes.
The reconstructed inventory hash is:

```text
10d41d10cf509e11db432c80ecd844bfdac2d5392ffee93a370e6638d9e57435
```

That is the same inventory identity recorded in the active dzfix matching
handoff. The count audit verifies each included ROOT checksum before and after
reading, latest tree cycle, schema, selected class counts, and count maxima.
It uses the established `hlt_matched`/label/source selection, including the
existing exclusion of QCD labels in nominal signal-source files. There are no
new kinematic, minimum-multiplicity, or performance cuts.

This is NOT the exact TRAIN_500K row profile. It is a larger census of safe
outer-reservoir rows. The conservative union exclusion also removes some
otherwise ordinary SPORC rows that were test in the local parent registry.
No replacement split or role reassignment was created.

| Audited population | Jets | Overflow jets | Overflow rate |
| --- | ---: | ---: | ---: |
| All included files | 6,858,920 | 32,452 | 0.473136% |
| Original local training role, after exclusions | 5,295,917 | 24,946 | 0.471042% |
| Original local validation role, after exclusions | 1,563,003 | 7,506 | 0.480229% |
| SPORC training files, after exclusions | 2,787,630 | 14,095 | 0.505627% |
| SPORC validation files, after exclusions | 806,365 | 3,595 | 0.445828% |
| Combined SPORC ordinary overlap | 3,593,995 | 17,690 | 0.492210% |
| Safe additional files outside the SPORC snapshot | 3,264,925 | 14,762 | 0.452139% |

The local and SPORC rows above are overlapping summaries, not independent
samples to add together. The agreement is descriptive; no IID confidence
interval or cross-file generator-independence claim is made.

## Class dependence

Class names follow the existing JetClass2 label contract. These summaries are
diagnostics, not permission to make class-dependent matching/cropping choices.

| Class | Jets | Overflow jets | Overflow rate | Fraction of that class's offline particles cropped |
| --- | ---: | ---: | ---: | ---: |
| QCD | 5,202,518 | 6,422 | 0.1234% | 0.0326% |
| X_bb | 224,928 | 447 | 0.1987% | 0.0399% |
| X_cc | 225,124 | 438 | 0.1946% | 0.0389% |
| X_ss | 224,000 | 510 | 0.2277% | 0.0467% |
| X_qq | 225,482 | 478 | 0.2120% | 0.0461% |
| X_gg | 233,314 | 322 | 0.1380% | 0.0272% |
| X_ee | 107,478 | 2,905 | 2.7029% | 2.5436% |
| X_mm | 79,276 | 9,402 | 11.8598% | 8.8229% |
| X_tauhtaue | 84,850 | 2,430 | 2.8639% | 1.1688% |
| X_tauhtaum | 82,003 | 5,405 | 6.5912% | 2.6047% |
| X_tauhtauh | 169,947 | 3,693 | 2.1730% | 0.7640% |

An equal-class average of overflow rates is **2.4805%**, compared with the
natural-population rate of 0.4731%. Neither is a classifier-performance loss.
Hadronic quark/gluon channels have roughly 0.12-0.23% overflow; the leptonic
and tau channels require explicit per-class monitoring in the future study.

## How much count cropping is required?

The capacity rule removes exactly `max(N_offline - 2*N_HLT, 0)` offline
particles and never removes the original HLT slots.

- Total offline particles: **281,125,975**.
- Offline particles exceeding capacity: **206,840**, or **0.073576%** of all
  offline particles in the census.
- Among overflowing jets: median **3**, mean **6.37**, 90th percentile **16**,
  95th percentile **23**, 99th percentile **39**, maximum **97** particles lost.
- At least 10 particles would be removed in **6,498 jets (0.094738%)** of the
  full census; at least 20 in **2,360 (0.034408%)**; at least 50 in **123
  (0.001793%)**.
- Within overflow jets, the median fraction of offline particle count removed
  is **33.33%**, the mean **37.48%**, and the maximum **97.98%**. Thus a small
  population-level lost-particle fraction does not imply mild per-jet loss.

All quantiles are weighted nearest-rank observed quantiles, not interpolated
or extrapolated tail estimates.

| HLT particles | Jets | Overflow rate within this band |
| --- | ---: | ---: |
| 1-4 | 75,661 | 26.4284% |
| 5-9 | 190,890 | 3.3475% |
| 10-19 | 701,072 | 0.5241% |
| 20-39 | 2,730,646 | 0.07994% |
| 40-59 | 1,950,864 | 0.01020% |
| 60-99 | 1,084,898 | 0.00092% |
| 100+ | 124,889 | 0% observed |

**81.31% of all overflowing jets have fewer than ten HLT particles.** This
explains the need for a robust, explicit overflow policy even if K=2 remains
the chosen default. Zero observed overflow in the last bin is not a guarantee
for other populations.

## Best-case scalar-pT loss: measured second pass

A second read-only pass rechecked particle-array lengths against the stored
counts for **all 6,858,920 jets** (`part_px`, `part_py`, and `hlt_part_px`). It
then measured all **32,452 overflowing jets**, dropping exactly the excess
offline particles in increasing raw `hypot(px, py)` order.

This is the **minimum possible removed scalar pT for the required particle
count crop**. It is not the final quantized/core-aware salience policy, and
neither scalar pT nor this minimum bounds the loss of classifier information.
A policy that prioritizes features other than raw pT could remove more pT.

- Across all offline particles in the census, the minimum lost scalar pT is
  **0.008093%** of the total.
- Within overflowing jets, the per-jet minimum lost-pT fraction has median
  **0.4490%**, mean **3.4406%**, 90th percentile **7.9788%**, 95th percentile
  **18.7604%**, 99th percentile **53.0529%**, and maximum **87.8694%**.
- **2,767 jets (0.040342% of the full census)** must lose more than 10% of
  offline scalar pT even under this optimal-pT retention rule. The equivalent
  counts above 25% and 50% loss are **1,214** and **388**.

The following medians and percentiles are conditional on overflow. The last
column instead uses ALL offline scalar pT within that class as denominator.

| Class | Median minimum pT loss per overflow jet | 95th percentile | Minimum fraction of all class pT lost |
| --- | ---: | ---: | ---: |
| QCD | 1.3116% | 51.6693% | 0.004139% |
| X_bb | 1.2676% | 17.4923% | 0.003021% |
| X_cc | 0.8398% | 8.9650% | 0.002509% |
| X_ss | 0.9246% | 15.0531% | 0.003024% |
| X_qq | 0.9333% | 11.7001% | 0.002779% |
| X_gg | 1.5820% | 22.0288% | 0.002563% |
| X_ee | 0.3179% | 7.3069% | 0.035836% |
| X_mm | 0.2971% | 4.8611% | 0.118308% |
| X_tauhtaue | 0.5453% | 12.5087% | 0.061902% |
| X_tauhtaum | 0.4020% | 11.3642% | 0.143089% |
| X_tauhtauh | 0.3831% | 14.7562% | 0.047338% |

Thus high count-crop frequency does not necessarily mean high lost pT:
X_mm overflows often, but the median minimum pT loss within those cases is
only 0.2971%. Conversely, the rare QCD overflow tail includes severe cases.
The worst measured minimum-pT-loss jet has one HLT and 51 offline particles,
requiring removal of 49 offline particles. The report preserves its source
identity and the other 19 worst examples for subsequent diagnostics.

## What larger K would change

These are count diagnostics only; no new K variant was selected or queued.

| Rich slots per HLT owner | Total endpoint copies | Jets still overflowing | Rate |
| --- | ---: | ---: | ---: |
| K=2 | 3 | 32,452 | 0.473136% |
| K=3 | 4 | 13,849 | 0.201912% |
| K=4 | 5 | 8,881 | 0.129481% |

Increasing K reduces but does not eliminate the tail. The largest observed
offline/HLT count ratio is 99. Large fixed replication factors would increase
every jet's compute to rescue a much smaller set of low-HLT-count cases.
These measurements do not establish which choice yields the best classifier.

## Expanded input lengths

On the audited rows, HLT x3 has median 114 active tokens, 90th percentile 210,
95th percentile 246, 99th percentile 327, and maximum 831. Mean active length
is 123.09. About 33.39% of all initial x3 slots would be HLT fillers.

This confirms that the new study cannot silently use the old 320-token input
capacity. The 831 maximum is an observation on this audited scope, not an
authenticated capacity bound for all registered rows or an A100 memory gate.
The expanded input still requires its own production acceptance.

## Reproducible artifacts and code

Count evidence is stored locally under:

```text
artifacts/jetclass2_dzfix_k2_capacity_audit_v1/
    scope.json
    joint_histogram.json
    report.json
```

The histogram contains exact `(class, N_HLT, N_offline, count)` bins, not
particle arrays or classifier outputs. The report includes per-file evidence,
population/role/class summaries, tail distributions, source-file hashes, and
explicit final-test flags. The schemas are new diagnostic-only
`JETCLASS2_DELPHES_K2_CAPACITY_AUDIT_{SCOPE,HISTOGRAM,REPORT}/v1` artifacts.

Count report hash:

```text
71911c0616dbe04c67fc56aa396d53b9b19d151d0fb315305e3767cdb8eaf054
```

Source inventory/splits and transfer plan are under
`artifacts/jetclass2_20260918_dzfix_source_v1/`. The reusable read-only module is
`src/hlt_classification/jetclass2_delphes/capacity_audit.py`; its thin CLI is
`scripts/audit_jetclass2_k2_capacity.py`. No legacy donor file was copied.

The separate pT module/CLI are `capacity_pt_audit.py` and
`scripts/audit_jetclass2_k2_pt_loss.py`. Its immutable diagnostic
`JETCLASS2_DELPHES_K2_CAPACITY_PT_AUDIT_REPORT/v1` is saved at
`artifacts/jetclass2_dzfix_k2_pt_audit_v1/report.json`, binds the count report
and scope hashes, and has content hash:

```text
dbfd155ba5726a4c09c735573712655d7d10e10229141d9ec7cf259271240a24
```

Reproduce in the existing local scientific environment, choosing a fresh
output root (do not overwrite the recorded report):

```powershell
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -s `
  scripts/audit_jetclass2_k2_capacity.py `
  --data-root C:/Users/22rya/ComputerScience/CERN/data/jetclass2_10M_20260918_dzfix/jetclass2 `
  --inventory-path artifacts/jetclass2_20260918_dzfix_source_v1/local_audit_v1/inventory.json `
  --splits-path artifacts/jetclass2_20260918_dzfix_source_v1/local_audit_v1/splits.json `
  --partial-plan-path artifacts/jetclass2_20260918_dzfix_source_v1/partial_plan_v1/partial_snapshot_plan.json `
  --output-root artifacts/jetclass2_dzfix_k2_capacity_audit_recheck

& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -s `
  scripts/audit_jetclass2_k2_pt_loss.py `
  --count-root artifacts/jetclass2_dzfix_k2_capacity_audit_recheck `
  --inventory-path artifacts/jetclass2_20260918_dzfix_source_v1/local_audit_v1/inventory.json `
  --output-root artifacts/jetclass2_dzfix_k2_pt_audit_recheck
```

The count census took about 71 seconds, excluding scope replay/reporting;
the pT pass took about 127 seconds.
Machine-specific elapsed time is not a scientific identity or cluster-runtime
estimate. Generated `artifacts/` files are Git-ignored and must be preserved
separately if transferring this evidence.

Local verification: **19 tests passed** across the new audit tests,
partial-snapshot regressions, selection-policy test, and three scaffold
checks. The whole-repository Markdown walk was deselected because unrelated
temporary directories are not part of this study; changed-document links were
checked separately. Tests cover the strict boundary/equality case, exact
denominators, chunk invariance, malformed inputs, union-of-test exclusion,
ROOT reads restricted to safe files, immutable output, and minimum-pT cost.
No installed-Weaver or SPORC GPU acceptance is claimed for this read-only
diagnostic; those remain requirements for any future training implementation.

## Limits and recorded design decision

The rarity question has been answered on all locally available rows allowed
by the conservative two-registry test seal. It does not prove classifier
retention, physical correspondence quality, or queue readiness. Do not label
the entire rich start "uncropped offline" on the overflow rows.

After reviewing these measurements and the K=3 count comparison, the user
chose **K=2 for the first experiment**. Keep the declared offline-only crop
rule and report the low-HLT-count/leptonic tails separately. The initial
D100-versus-pure-offline validation comparison will measure the rich start's
classification performance; outperforming offline is not guaranteed or a
job-success condition. Any later adjustment requires a separately registered
follow-up, not a silent change to K or label-dependent cropping. This decision
does not alter the immutable audit reports or authorize live submission.
