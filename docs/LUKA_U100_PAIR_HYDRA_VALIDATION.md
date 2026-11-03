# Hydra adapter validation and donor record

Local implementation base: `65952ed3805f812c06d1ea54117c35b630cc684e`.
The user reported successful export and per-member verification on Brown of
2,287 files, 3,127,574,502 payload bytes. The new consumer pins that archive and
manifest. This record does not claim an executed Hydra environment or GPU job.

New files: the seven `luka_u100_pair_hydra` modules, CLI, environment installer,
queue helper, Slurm worker, focused tests, plan, execution contract and runbook.
Existing documentation gains pointers/status; original scientific code is unchanged.

Before edits,32 focused transfer/correction tests passed in8.39s. After initial
implementation,83 combined tests passed in59.29s using the PyPI Weaver0.5.3 wheel
in an isolated local test path. They cover archive corruption/exclusivity,
transferred synthetic ROOT-derived view fingerprints and matching joins, particle
target export/reload, native checkpoint probability replay, environment mismatch,
Hydra test-only requests, exact two-lane dispatch, failed setup and duplicate
submission, and the original CPU/CUDA correction/model production regressions.
These are synthetic local checks, not the recorded SPORC environment replay or
real L40 resource evidence. Two additional allocation/source-guard tests then
passed in5.27s (85 passing cases across the focused checks). AST parsing passed
for9 new Python files; all three shell scripts passed bash syntax checks, scoped
whitespace checks passed, and four local documentation links resolved.

Donors (all at the base commit): `luka_u100_pair_correction/{worker,submission}`
adapted for Hydra orchestration; `{targets,campaign,source}` for portable loading,
registration and source guards; `{model,geometry,losses,training,resources}`
imported unchanged. `luka_u100_pair_transfer/bundle` supplies transfer integrity.
`luka_fullsim/{regular_cache,alljets_data,foundation,regular_submission}` supplies
block/foundation/source and environment primitives. `luka_u100_embedding/targets`
supplies the unchanged bank validator. `jetclass2_delphes/{model,runner,reporting}`,
`luka_u100_contrastive/worker`, `cms_proxy_ladder/{contracts,submission}`,
`data/cache_contracts` and `scouting/{hcwdl_exact_dag_submission,hcwdl_recovery}`
provide existing model/metric/artifact/Slurm helpers. Original correction CLI,
queue helper, worker shell and transfer/correction test fixtures informed the
additive counterparts. No Fresh_check donor or runtime dependency.

Execution and extraction use new v1 contracts described in
[the contract](contracts/LUKA_U100_PAIR_HYDRA.md). Scientific contracts are reused
without changing their meaning. The remaining tasks are clean commit/push,
isolated Brown environment matching, extraction, full dry review and genuine
shared setup. Setup owns numerical portability and measured resource admission;
only its successful completion releases the four scientific fits.
