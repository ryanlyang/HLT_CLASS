# Frozen literature V3 direct and coarse comparison on SPORC

## Scope

Run a fresh 200,000 training / 50,000 validation comparison on the completed
NOISE_V3 literature-inspired synthetic dataset. This is not the learned CMS
proxy, native Delphes HLT, or a detector-realism claim. Reuse the existing
direct/coarse classifier implementation, not its data, assignments or models.
User-supplied generation evidence: array 218347 and successful finalizer
218348 (2026-10-03). Downstream preparation must authenticate saved artifacts;
Slurm success alone is not an input capability.

## Frozen comparison

Fresh CE controls are M0HLT, pure OFFLINE and persistent-HLT U000. DIRECT is
U000 -> D000. COARSE is U000 -> U050 -> U100 -> D066 -> D033 -> D000.
DENSE and ULTRADENSE are absent. There are nine fits, five probability-bank
reducers, one aggregate and one completion task. Each student receives only
its immediate predecessor's supervision: C25/P75 forward KL at T=2, one T²
factor, batch 256, AdamW, BF16 forward / FP32 loss, coordinate-paired seeds.
Reuse the previous 3-pass warmup, hold through 45, decay through 60, floor
tail with patience 15, minimum 60 and maximum 100 passes. The earlier printed
60-pass outcomes were stopping outcomes, not a new fixed 60-pass protocol.
Recovery is M0HLT=0%, pure OFFLINE=100%; report U000 separately.

Match physical endpoints afresh with SALIENCE_PT_LINEAR full-cardinality
matching, saturating the smaller side. Do not use generator ancestry as
matching truth or model features. Preserve the previous persistent proxy
skeleton and offline tail: U000 includes unmatched proxy particles, U100 has
offline features on matched proxy slots and unchanged unmatched proxy slots;
D000 is exactly the saved proxy. U removes the offline-only tail; D changes
features. Native tracking values/errors are mm, momenta GeV. No CMS unit or
sign conversion, rerandomization or recipe calibration is permitted.

## Population and access

Require authenticated literature train and validation role releases. Select
exactly 200k and 50k by smallest SHA-256 rank of versioned domain, dataset
study hash, role and canonical identity; restore physical file/block order.
Selection reads no labels. Then join labels and offline particles using the
same original ROOT file/tree/entry and inventory-bound identity. Store compact
locators and new assignments, not duplicate full views. No native HLT particle
branch is opened. Existing hlt_matched eligibility metadata may be checked.
Final-test files, banks, receipts and inference are excluded. Metadata in
the source study may describe sealed membership; that does not authorize reads.

## Execution and admission

All gate and science jobs use SPORC debug, reu-aisocial, qos_tier3,
atlas_kd_sporc, and at most 24 hours. This explicit new authorization overrides
the old CMS-proxy amendment's measurement-only debug policy for this variant
only. No implicit transfer to tier3. Each fit/reducer requests one A100,
16 CPUs/workers and 160000 MiB; CPU foundation uses 16 workers, no GPU.
Gate tasks authenticate/freeze the ordinary release, construct/audit matches,
then measure the full selected population through the actual training and
reducer kernels. Installed-Weaver parity must pass on the allocated GPU.
The measured 100-pass conservative runtime must fit the 24-hour ceiling;
otherwise stop with evidence, never silently shorten training or move sites.

Use new gate/campaign roots, exact clean pushed source, immutable hash-linked
artifacts, and dry-run-first exact DAG submission. The gate and science have
separate explicit authorization phrases. Science creation is impossible before
the measured gate completes. No remote submission, existing-job cancellation,
data deletion, old-artifact relabeling or final-test unlock occurs implicitly.
Poor scientific metrics are reported and never block descendants.
