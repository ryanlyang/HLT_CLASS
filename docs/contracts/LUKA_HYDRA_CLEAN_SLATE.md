# Hydra new-dataset clean-slate retirement: v1

Authority: operator request to delete prior `hlt_u100_pairs` contents except
dependencies of the new600k reconstruction study; active plan
[execution revision](../plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md).
Implementation: `operations/hydra_clean_slate.py`, no dependency on an old
campaign execution. Base is exactly `/cs/home/rlyang/hlt_u100_pairs` for rlyang.
No operation alters Oscar, scheduler jobs or the local research dataset.

Keep manifest `LUKA_BALANCED_RECONSTRUCTION_HYDRA_WORKSPACE_KEEP/v1`, schema1,
canonical SHA256 `8d5077aa3d15e303756e03aa55476bee1c2abac2692ae83dac77613ed1ba6822`,
is committed at `luka_balanced_reconstruction_hydra/workspace_keep.json`. Its
source checkout is clean, exact pushed40-character HEAD, ancestor of origin/main,
at `code_balanced_reco_<commit8>` with common Git `code_65952ed3/.git`.

Preserve whole datasets container (including raw, foundation and transfer
records), whole `environment_pair_hydra_v1` interpreter/venv, primary Git
checkout/store, new pushed checkout, every `runs/balanced_reco_600k_*` directory
and `retired_campaigns`. Check interpreter location and editable paths; a300s
CPU import probe verifies torch/Weaver dependencies and actual STORAGE_POLICY/v2
with zero historical reservation/dependency. Imported workspace paths may only
reference the kept runtime/current source/primary checkout. Unknown top-level
directories and dirty/unpublished/separate-Git worktrees are retained and listed.

Eligible old inputs are a fixed named allowlist: bundle_65952ed3,
cache_new_fullstat_offline_a865e5b6_r1, contrast_delta_54473f10,
cross_message_encoder_73abecfd, frozen_adapter_delta_4b9630fa, imports,
inputs_65952ed3, offline_teacher_6fe7e349, pretext_validation_54ff594d,
raw_next_0a6721ed. All other immediate run directories are old campaigns and
may be retired, including previous full-size new-dataset runs, their scientific
models, predictions and caches. Old code directories qualify only as clean
registered worktrees in the retained Git store with pushed main ancestry.
Remove them through `git worktree remove`, without force or pruning history.

Before mutation, list every target/keep/skip, validate canonical owned roots,
reject redirected paths, and verify all removals are disjoint from kept paths.
Require empty Slurm queue initially and immediately before each removal. No
submissions should be made while cleanup is running. Top-level loose files and
symlink directories are not targets; child symlinks are never traversed.

Archive regular compact text metadata before deleting old run/input trees:
json/jsonl/csv/tsv/txt/log/out/md/yaml/yml/sha256 and README, at most32MiB per
file. Larger logs and scientific binary arrays are retired without reading or
hashing their payloads. Pushed source history remains in Git rather than this
archive. Verify archive member inventory and per-member SHA256, publish a
hashed archive index, and verify again before deletion. Recheck original
metadata identities and directory signatures before each root removal. This
does not establish original model/array payload integrity or scientific lineage.

Operation records under `retired_campaigns/new_dataset_clean_slate_v1` use
`LUKA_HYDRA_CLEAN_SLATE_<KIND>/v1`, schema1, canonical content hashes, explicit
parent hashes and `final_test_inference=false`: immutable INTENT, ARCHIVE index,
fsynced DELETED journal and completion-last COMPLETE. The archive itself has
SHA256. Intent records exact source/helper identities, keep manifest, target
root signatures and code commits. Partial failure retains all records; refuse
blind repeat whenever that operation directory exists. Never change original
scientific manifests to pretend retired binary payloads still exist.

Explicit CLI token is `AUTHORIZE_HYDRA_NEW_DATASET_CLEAN_SLATE_V1`, already
authorized by the operator's request. The PowerShell launcher supplies it after
uploading/hash-checking the helper and preparing exact pushed current source.
Preview lists a plan without deleting files or creating retirement records.
The helper uses standard-library filesystem/Git tools, plus the bounded kept
runtime import probe. It performs no training, inference, array deserialization,
final-test evaluation, model loading or job submission/cancellation. Deleted old
weights/predictions require another copy or retraining to restore.
