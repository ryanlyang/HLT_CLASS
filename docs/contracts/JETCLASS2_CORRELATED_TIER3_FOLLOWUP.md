# CORR_MID: debug gate to tier3 automatic continuation

Authority: the [active operational amendment](../plans/JETCLASS2_CORRELATED_TIER3_FOLLOWUP_PLAN.md).
Original GATE_SPEC/RUNTIME_PROFILE v9 and CAMPAIGN_SPEC v5 remain unchanged.

New artifacts in `JETCLASS2_CMS_PROXY_LADDER_`:

- `CAMPAIGN_SPEC/v6`: original measurement-campaign file bytes/hash, new executor
  pin/root, exact original science and a separately authenticated source transfer.
- `TIER3_SOURCE_TRANSFER/v1`: measurement/executor source parents, exact approved
  dispatch insertion and added adapter. Other measured source hashes must match.
- `TIER3_RUNTIME_PROFILE/v1`: original debug measurement parent, unchanged
  timing/memory/GPU/environment/acceptance evidence and measurement source commit,
  explicit debug measurement site and tier3 execution site. Uses the registered
  `sporc_debug_to_tier3_same_a100_environment_resources_v1` policy.
- `CORRELATED_TIER3_FOLLOWUP/v1`: immutable authorization, exact-plan review and
  submitted ledger evidence. Completion of the controller is not training success.

All artifacts use canonical content hashes, parents, atomic immutable publication
and final_test_accessed=False. No physical test blocks are read. The frozen
dataset manifest is `ef015d3c86f8188784169f67723a0f9772af96c4dab0a06387b208d9d33b4610`.

The gate's original source commit is
`062713ead3e00721d9f3cb7073e88aef83ebd325`. The old pinned CLI authenticates the
exact live gate ledger/journal before arming and again after the wait. All three
jobs must account as COMPLETED/0:0, with exact names, user and reu-aisocial account.
Gate hashes, output receipts, genuine installed-Weaver/GPU acceptance and the
full-population runtime profile are then validated by the existing scientific
validators. Never infer successful gate completion merely from a file existing.

The new clean pushed executor must preserve every file in that measurement's
source snapshot. The sole existing-file change is the exact version-6 dispatch
inside production.validate_campaign. Even another comment added to that file
fails the transfer check. The new correlated_tier3.py module is explicitly added.
No changed model, view, generator, matching, kernel, seed or schedule is accepted.

One new science root, 100k/50k, six fits/three reducers/two reporting jobs. All
commands target tier3/reu-aisocial/qos_tier3; train/reduce each use one A100,
6 CPUs, 90000 MiB and exactly the measured time within 24h. The unchanged worker
checks actual allocation, exact measured GPU identity and installed environment.
Different A100 variants or software do not silently reuse incompatible evidence.

No old campaign/job is mutated or cancelled. Existing or ambiguous debug-science
submission blocks arming. `science/` may hold the original unsubmitted measured
spec; tier3 artifacts live in `science_tier3/`, controller receipts in
`followup_tier3/`. The launcher is read-only without explicit --execute. Arming
authorizes a deferred **full** dry plan and policy-bound exact-hash live submit,
not arbitrary future jobs. Submission requires the canonical dry ledger, site
test-only probes, exclusive live claim and exact command journal. Failed or
ambiguous submission stops without resubmission; recovery needs inspection.

This operational change does not establish a classifier gap, ladder advantage,
real detector fidelity, or a shorter queue wait. Poor metrics remain results.
