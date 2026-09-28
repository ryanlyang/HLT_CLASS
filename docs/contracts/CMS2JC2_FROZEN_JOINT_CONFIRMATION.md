# CMS2JC2 frozen JOINT confirmation v1

Authority: `docs/plans/CMS2JC2_FROZEN_JOINT_CONFIRMATION_PLAN.md`.
Additive CMS2JC2_RESPONSE_FROZEN_{PROTOCOL,MEMBERSHIP,REUSE,STAGE,
ACCEPTANCE,ACCESS,SHARD,REPORT}/v1 artifacts. Old artifacts and semantics remain
unchanged. Every artifact is content hashed, exact-parent validated, source
bound, atomically published and final_test_accessed=false.

Freeze only the completed JOINT winner plus non-selecting B_DZ control. No
production_qualified or transfer_authorized flag may become true here. Gate
reads residual only; confirmation requires full canonical outer-confirm
membership, untouched-population assertion, frozen protocol/model/map hashes,
real acceptance and per-task execution claim before any particle access.
Only original CMS train/response_confirm files may be opened. Missing rows,
duplicate identities, changed source, wrong receipt, nonfinite quantities and
forbidden access fail closed; poor fidelity publishes a normal report.

Source-file bootstrap, never shard/replica bootstrap. No fallback to a smaller
population or a different winner; supported/rejected/inconclusive is an honest
collection-check result, not detector-simulation certification. Read-only
results validate all required output receipts. No automatic next stage.
