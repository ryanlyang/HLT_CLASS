# Full-population CONTEXT_V1 Oscar ladder contracts

Authority: `docs/plans/JETCLASS2_CONTEXT_OSCAR_1M_DIRECT_COARSE_PLAN.md`.

New versions, not replacements for old artifacts:

| Artifact (JETCLASS2_CMS_PROXY_LADDER prefix) | Version |
| --- | --- |
| RELEASE_REQUEST, RELEASE, FOUNDATION | 6 |
| GATE_SPEC, RUNTIME_PROFILE | 11 |
| SCIENTIFIC_PLAN, CAMPAIGN_SPEC | 8 |

Request domain `JETCLASS2_CONTEXT_OSCAR_1M_250K/v1`; exact counts 1M/250k;
frozen CONTEXT_V1 manifest and relocated reference checks unchanged. Every
committed ordinary row is selected label-blind. No final-test decoding.
CONTEXT input/v2 and view/v3 contracts are reused unchanged. Full FOUNDATION
additionally binds `assignment_slots` (total mapping elements) and
`cache_preparation=source_file_bounded/v1`. Validators verify the slot count
against the assignment bank. All release/foundation/model/probability products
retain content hashes, parent lineage, immutable source and atomic publication.

Cache tasks are single source files in release order. No more than `workers`
futures are submitted/retained concurrently. Resident bound charges all rows
at capacity; worker/IPC bound charges the largest source files; additional
per-worker allowance covers full mapping/index copies and interpreter/I/O
overhead. Combined train/validation bounds must fit 75% of requested RAM.
Actual resident bytes are checked while accepting blocks. No truncation,
spilling or different model input semantics is permitted for a memory failure.

Fresh controls and DIRECT/COARSE nodes use CTXV1M names and the same coordinate
seeds and recipe as the smaller experiment. Full-population measured acceptance
is mandatory; a 100k runtime profile or foundation is not interchangeable.
Resource binding: Oscar L40S, six CPUs/workers, 180000 MiB. Test remains sealed.

Live phrases:

- `AUTHORIZE JETCLASS2 CONTEXT 1M OSCAR GATE`
- `AUTHORIZE JETCLASS2 CONTEXT 1M OSCAR DIRECT COARSE SCIENCE`

Both require exact dry command-plan hashes, clean pushed pinned source and
exclusive live-submission journals. No dataset writes or modifications to the
old 100k execution are authorized by this contract.
