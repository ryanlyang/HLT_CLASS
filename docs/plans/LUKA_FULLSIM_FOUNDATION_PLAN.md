# Luka FullSim 630k trial: stage-1 foundation

User authorization (2026-10-08): implement stage 1 only for the new paired
FullSim export, followed later by a 100k/50k SPORC direct/coarse/fusion study.
This plan overrides the old Delphes and CONTEXT dataset/split assumptions for
this new namespace only. Historical datasets, jobs and code remain unchanged.

## Frozen selection

Source: `ryang@lxplus.cern.ch:/eos/user/l/llambrec/jetclass/output_fullsim_630k_trial`.
Local transfer container: `/home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1`,
with `source.txt`, `source.sha256` and `raw/jetclass2/{train_higgs2p,train_qcd}/fullsim_offline+hlt/*.root`.
The observed snapshot has 890 files and 647935 raw entries; measured eligible
counts are evidence, not substitutes for reauthentication.

Require `hlt_matched`, positive offline/HLT scalar particle counts, and strict
**offline `jet_pt > 200 GeV`**. No HLT pT, delta-R, tight-ID, mass, particle-count
upper bound, or other inherited cut. The offline-pT interpretation is explicitly
user agreed, not producer confirmed. QCD (161--187) is retained only from QCD
files. Use the existing eleven-class order and signal codes 0--3,9--14 as a
provisional code convention; unknown labels fail, never disappear silently.
Signals in QCD files are retained if encountered. Record all exclusions.

Stage 1 reads only label, match, particle-count and offline-pT scalar arrays,
plus ROOT headers. Full source bytes, including future test files, are hashed
opaquely. It does not read particles, invent tracking units, validate particle
physics, build assignments, train models, or certify GPU readiness. Tracking
units/error sentinel conventions and cross-file event independence remain
unverified and must be handled explicitly in stage 2.

## Split registration

Seed 20261008; whole-file reservoirs target 40% train / 20% validation / 40%
sealed final-test reserve. This supplies headroom for exact 100000/50000 draws
from the observed ~374k eligible population. Group placement uses class coverage
then normalized squared deficits, largest rare-class file mass first, SHA256
path tie breaks. File content duplicates, path aliases and source escapes fail.
File disjointness is proven; event disjointness across files is NOT claimed.

Each ordinary role's exact eleven-class quotas use integer largest remainders
against the **global post-cut class counts**, ties by class index. No balancing,
weights, sampling with replacement, or minimum-one adjustment that distorts the
ratios. Zero quotas, absent classes, insufficient per-class reservoir capacity,
and insufficient class coverage in the sealed reservoir fail with counts.
Do not silently change seed, move files, shrink samples or relax disjointness.

Within role/class, rank eligible rows by SHA256 over versioned domain, seed,
inventory hash and source-bound row identity; choose the quota, restore source
file/entry ordering for consumption. Row identity binds relative path, file
SHA256, latest tree cycle and entry, not absolute location, label or split.
Offline and HLT share that same row key (not constituent-level matching).
Unused eligible rows in train/validation files are reserved unused in that
same role, NOT added to test. All eligible rows of final-test files form the
sealed test reserve. Population metadata/labels may be audited pre-split;
ordinary accessors reject the test role. Model inputs cannot include these keys.

## Artifacts and execution

New `LUKA_FULLSIM_*/v1` inventory, split and foundation artifacts, canonical
content hashes, source snapshot, explicit parents, full input/output checksums,
atomic non-overwriting publication. Build requires a clean commit containing the
implementation, checks the requested commit and source again before publication.
Metadata validation replays the split recipe, not just its self-hash. Full
verification hashes raw files and replays scalar inventory from the transfer.
Fresh output outside raw input only; partial directories are preserved and not
silently resumed. The final foundation file is the completion marker.

CLI provides build, verify and summary only. No Slurm submission, GPU gate,
training, automatic follow-up, final-test particle reader or evaluation switch.
A thin SPORC worker invokes only build (1 CPU, 8 GiB, 1h debug, initial
unmeasured envelope); the runbook shows separate resource review/submission.
A pinned SPORC execution is still needed to materialize the real foundation.
Local synthetic ROOT tests do not constitute that execution or Weaver acceptance.
