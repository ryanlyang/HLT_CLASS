# LUKA_FULLSTAT_PT_AUDIT/v1

Authority: [audit plan](../plans/LUKA_FULLSTAT_PT_AUDIT_PLAN.md).

The report uses `schema_version=1`, canonical sorted compact ASCII JSON with
finite numbers, and SHA256 excluding `content_hash`, compatible with the
existing FullStat convention. `parents` binds the supplied campaign and
original foundation. Absolute metadata/membership refs record path, bytes and
file SHA256. This authenticates saved bindings; it does not certify the raw
ROOT files, scheduler completion, original source tree or split replay anew.

`train_membership_file_entry_order_sha256` hashes source path/hash and canonical
hashes of selected entry indices, mapped labels and FP64 offline pT, in source
file/entry order. It is not the prepared model-cache order hash.

`analysis` defines the TRAIN-only class summaries, candidate grids, inclusive
final upper edge, flows, advisory sparse threshold and hypothetical draw
budget. All classes/cells remain explicit. If any cell is empty,
`feasible_on_this_range=false` and `sampling_projection=null`. CSV per-cell
nominal exposures are labelled hypothetical even when the full grid is
infeasible. Zero-occupancy cell exposures are blank, not zero.

`final_test_accessed=false` means no final-test inference/evaluation or particle
access. `shared_scalar_metadata_read=true` explicitly acknowledges that the
membership archive includes scalar rows from every role; `aggregated_roles`
is exactly `['train']`. Event independence and old-trial overlap remain
unverified. No mask/cut, deployed input, training loss or sampler is modified.

`exports` binds the relative paths, sizes and hashes of `summary.txt` and
`cells.csv`. Each file is published without overwrite; `report.json` is last.
Existing output directories are rejected, including partial audits. The
portable `show` reader validates report and export hashes using only that
directory. Timestamps are not identities. A future change to semantics must
receive a new contract rather than masquerade as this diagnostic.
