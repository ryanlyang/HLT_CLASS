# Luka FullSim foundation contracts v1

Authority: [stage-1 plan](../plans/LUKA_FULLSIM_FOUNDATION_PLAN.md).

`LUKA_FULLSIM_INVENTORY/v1` binds exact transfer marker and manifest bytes,
relative ROOT paths, SHA256, latest tree cycle, entire common branch schema,
raw/source label counts, sequential exclusion counts and selected entry lists
by eleven-class label. Only scalar metadata are decoded. Values required for
selection must have valid types; negative counts, unknown labels, nonfinite
offline pT, duplicate source contents and schema drift fail closed.

`LUKA_FULLSIM_SPLITS/v1` binds the inventory, exact seed, deterministic file
reservoirs, global-proportion integer quotas and exact ordinary memberships.
Validation regenerates the complete canonical partition and memberships.
All ordinary rows join offline and HLT by the same latest-cycle source entry.
Unused ordinary rows never become test rows. Final-test metadata do not grant
particle access; no test accessor is provided.

`LUKA_FULLSIM_FOUNDATION/v1` binds inventory and split content/file hashes,
the original clean implementation source snapshot, runtime library versions,
input location (locator only), and explicit metadata-only readiness flags.
`final_test_accessed=false` means no particle decoding or evaluation; hashing
and pre-split scalar metadata access are separately declared. It is NOT a
particle/matching foundation or admission to science workers.

Relocation leaves row identities and inventory/split content hashes invariant.
Full verification authenticates all bytes, ROOT/scalar selection and split
replay. Ordinary membership access validates saved artifacts before use and
can reauthenticate a supplied raw transfer container. Source assumptions and
unresolved units/event provenance must remain visible to later stages.
