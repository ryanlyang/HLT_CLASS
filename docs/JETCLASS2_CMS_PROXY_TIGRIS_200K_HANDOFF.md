# JetClass2 CMS-proxy Tigris 200k handoff

This is the operator handoff for the new CMS-calibrated proxy-HLT comparison.
It does not reuse the old Delphes HLT endpoint, old assignment banks, SPORC
execution, or old fitted models.

## Fixed inputs

```text
proxy study:
/home/ryreu/atlas/datasets/jetclass2_cms_proxy_joint_2250k_b6f88def_r1

paired original offline ROOT:
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
```

The release contains 200,000 train and 100,000 validation jets.  It is selected
without labels from authenticated committed ordinary-role proxy receipts.
Final test is not read.

## Graph

```text
DIRECT: U000 -> D000
COARSE: U000 -> U050 -> U100 -> D066 -> D033 -> D000
DENSE:  U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000
```

Fresh controls are `M0HLT`, pure `OFFLINE`, and `U000`.  Recovery is reported
from `M0HLT` (0%) to pure `OFFLINE` (100%).

## Staged execution

The first queue action is only the source-pinned Tigris gate.  It freezes the
release, recomputes full-cardinality salience assignments for the proxy
endpoint, audits endpoints, and measures a real GH200 full-population pass.
The full 17-fit campaign is created and queued only after that gate completes.

The exact commands are provided in the completion response for the pushed
commit.  Do not substitute the main checkout for the detached source-pinned
worktree, and do not submit science before `gate_complete.json` and
`evidence/runtime_profile.json` exist.

## Durable versus RAM-only data

Durable outputs are limited to compact identity/source locators, assignment
indices, selected checkpoints, reports, and probability banks.  Full model
views and feature caches are rebuilt in RAM and never persisted.

## Repository-local donors

The implementation was started from repository commit
`3655955f6ef7184cc6fdb3b0e68bdc6d089e9fe2` and reuses, without redefining
their old artifact families:

- `cms2jc2_production/{campaign,contracts,output,population}.py` for the
  authenticated proxy study, committed shard receipts, physical NPZ schema,
  and exact original offline membership;
- `cms2jc2_response/bridge.py` for the common physical particle representation
  and dz-fix offline conversion;
- `scouting/hcwdl_fullcard_salience_{matcher,contracts}.py` for the exact
  transferred `SALIENCE_PT_LINEAR` assignment;
- `jetclass2_delphes/{model,runner,banks,reporting,execution}.py` for the
  installed-Weaver 17-input/11-output model, registered optimization kernel,
  probability banks, metrics, and Tigris allocation checks;
- `scouting/hcwdl_exact_dag_submission.py` for crash-safe exact-ID submission.

All new endpoint/release/view/foundation/campaign semantics live under
`src/hlt_classification/cms_proxy_ladder/`; no external donor or `Fresh_check`
runtime import is used.
