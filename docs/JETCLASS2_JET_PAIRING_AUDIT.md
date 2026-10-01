# JetClass2 raw HLT/offline jet-pair audit

This CPU-only diagnostic investigates whether the two native collections in
a ROOT entry look like the same physical jet. It is separate from optimal
constituent assignment and from model performance. It changes no campaigns,
assignments, splits, targets, or models. Final-test particles are inaccessible.

The versioned output is `JETCLASS2_DELPHES_JET_PAIRING_AUDIT/v1`. It binds the
validated inventory, frozen split profile and role membership, sampled source
files/checksums/entries, ordered row identities, seed, and implementation bytes.
Each sampled source file is checksum/schema verified before and after reading.
The default is training only, up to 16 files per source and 256 uniformly
sampled registered entries per file. ROOT chunks are bounded to 2048 entries;
particle arrays are discarded after computing scalar summaries. Only a small
immutable JSON report is saved. This is not a population-weighted mismatch-rate
estimate. Changing sample settings creates a new report rather than overwriting.

Diagnostics use **global constituent four-vectors**, before per-view centering.
They report constituent-sum jet-axis delta-R, pT, mass, multiplicity, charged
multiplicity, and charged pT fraction. Negative summed mass-squared counts are
reported; the displayed mass uses sqrt(max(mass-squared, 0)). Optional stored
producer matching distance and jet-axis branches are read from the same entries.
Missing/invalid producer distances are counted as unavailable, not assumed zero.
Producer and constituent-sum axes need not be identical.

Twenty reproducible derangements compare each HLT jet with a different offline
partner. Controls preserve (1) class and production source; (2) those plus
HLT-pT bins of width 0.2 in log(pT); and (3) class and source file. Singleton
groups are excluded and explicitly counted; actual/shuffled metrics use the
same eligible recipients. Partner permutations are bijective without self-pairs.
The report gives angular-distance quantiles, rank correlations, absolute feature
differences, per-class results, and the ten largest actual angular separations.
Across-shuffle ranges describe permutation variability, not statistical
confidence intervals. Within-bin pT correlation can be large by construction;
compare it with the corresponding actual value and the other properties.

Actual pairs should be much closer in angle and more correlated than shuffled
pairs if jet pairing is meaningful. Class-preserving controls are necessary:
good classification can survive incorrect within-class pairing. Strong
alignment does not prove exact physical identity, correct constituent matches,
producer event independence, or correct teacher-prediction generation. A poor
result is retained as a diagnostic result; there is no automatic quality PASS.
This diagnostic does not train models and needs no GPU/Weaver acceptance.

Local invocation using the existing checksum-identical snapshot:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -s `
  scripts/audit_jetclass2_jet_pairing.py `
  --inventory artifacts/jetclass2_delphes_20260910_provenance_v1/inventory.json `
  --split-profile artifacts/jetclass2_delphes_scaling_splits_20260911_v1/profiles/TRAIN_500K.json `
  --data-root C:/Users/22rya/ComputerScience/CERN/data/jetclass2_10M_20260910/jetclass2 `
  --output artifacts/jc2_jet_pairing_audit_train_v1.json
```

The CLI also accepts `--campaign-spec <existing MT20 campaign_spec.json>` to
obtain its embedded inventory/profile and data path. `--data-root` may relocate
the snapshot; checksums must still match. The diagnostic output must be outside
the source campaign and raw-data directories. No remaining ladder job is needed.

## Local observed sample, 2026-09-16

The default TRAIN_500K train audit completed on 8,192 jets / 32 files with
inventory hash `91b8a67191eb8f5f3d833bd3347211eafd12f9939263c7dabf89c8159b02603a`
and split-profile hash `4a894300666993ebb900045584e52691bbdfde0772474e491f462c8168060aff`.
Output: `artifacts/jc2_jet_pairing_audit_train_v1.json` (345,550 bytes), hash
`94d4e504ad76e1b7834a2981c8d84eec273819a30d63ec74e8fef85188aed9fe`.

| Class | Sample rows | Median actual jet delta-R | HLT/offline pT Spearman |
| --- | ---: | ---: | ---: |
| QCD | 4096 | 0.02334 | 0.82359 |
| X_bb | 662 | 0.03176 | 0.76867 |
| X_cc | 611 | 0.03382 | 0.71995 |
| X_ee | 60 | 0.52058 | 0.14304 |
| X_mm | 102 | 0.37836 | 0.06329 |

Overall median actual delta-R was 0.02914, versus 1.90000 for class/source
shuffled partners. Stored producer matching distances agreed with recomputed
constituent-sum distances within 2.41e-7. The sample argues against wholesale
random pairing but identifies much weaker geometric/kinematic correspondence
in the electron/muon classes, with larger tails also in tau modes. Distinguishing
reconstruction effects from bad associations needs further producer/physics
evidence; connecting them to KD requires per-class endpoint performance. These
are exploratory observations from the bounded sample, not mismatch fractions
or proof of causation.
