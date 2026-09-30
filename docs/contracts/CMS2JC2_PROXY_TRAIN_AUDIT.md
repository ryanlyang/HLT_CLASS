# CMS2JC2_PROXY_AUDIT v1

Authority: `docs/plans/CMS2JC2_PROXY_TRAIN_AUDIT_PLAN.md`.

Artifacts use `CMS2JC2_PROXY_AUDIT_{SPEC,SOURCE,SHARD,REPORT,OUTPUTS,PLAN,LEDGER}/v1`,
canonical content hashes, explicit parent hashes and `final_test_accessed=false`.
Validation/native-HLT access is false. These are diagnostics, never dataset
completion markers or classifier training authorizations. Existing production
schemas and scientific semantics are unchanged.

SPEC binds the original study file and content hash, all ordered train receipts
with byte hashes, frozen copied CMS real-reference histograms/ranges, source
snapshot and fixed CPU resources. SHARD binds SPEC and the source receipt; its
observables use canonical histogram definitions and include exact count-change
histograms and processing time. REPORT binds SPEC and all SHARD artifact hashes,
retains CMS coverage/status, and labels all cross-dataset distances descriptive.
OUTPUTS binds REPORT and authenticates CSV, JSON, text and PDF bytes. A completed
report is reusable only after checking this receipt, not by existence alone.

PLAN binds SPEC and exact sbatch argv. Submission is dry by default; live
submission requires the reviewed plan hash. An exclusive submission intent is
written before sbatch; ambiguous outcomes must be investigated, not retried
automatically. No mutation of any original dataset or campaign artifact is
permitted. Work requires all train receipts, but not a full dataset manifest.

No per-jet cross-dataset pairing or inference uncertainty is implied. Proxy
replicas are not independent observations. Frozen confirmation may be reduced
or scientifically unsuccessful; audit retains that status without relabeling
the mapping as genuine HLT or production-qualified.
