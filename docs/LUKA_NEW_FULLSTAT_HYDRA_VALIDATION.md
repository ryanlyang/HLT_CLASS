# New FullSim OFFLINE Hydra baseline: validation evidence

Recorded 2026-11-13. This is local implementation evidence, not a claim that
Hydra has accepted or run the campaign. See the [runbook](LUKA_NEW_FULLSTAT_HYDRA_RUNBOOK.md)
and [scientific plan](plans/LUKA_NEW_FULLSTAT_HYDRA_OFFLINE_PLAN.md).

## Population and transfer

All 6,996 raw ROOT files were checked against the CERN manifest. The new
known-class, source-compatible paired population has 5,735,950 rows. The
frozen split contains exactly 3M TRAIN, 500k validation, 1.5M sealed test and
735,950 unused rows. It applies no pT cut or sampling weights. Raw label 206
is quarantined and preserved; its producer meaning remains unconfirmed.

Foundation content hash:
`2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`.
Producer record content hash:
`5ac841f9f154838e3c0b7c473a5d9c085181ed6ee6c34e8577d92db20b0b4f85`.
The producer record binds the actual generator bytes separately from base
commit `ff22b20f27fdc67d0fbd1a1df7005eb4626d19b7`; it makes no clean-worktree claim.

Transfer inventory is 13,995 files / 26,912,173,193 bytes, including raw files,
memberships, source manifest and producer evidence. Transfer manifest hash:
`7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4`.
Scalar counts estimate 15.80 GiB resident OFFLINE TRAIN/validation inputs.
The GPU request is 40000 MiB; disk registration requires 27 GiB additional
available quota after upload. These are budget calculations, not measured
Hydra peak usage. GPU startup measures the actual memory and timing envelope.

## Focused checks

With `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=src`:

```text
python -B -s -m pytest -q -p no:cacheprovider tests/test_luka_new_fullstat_hydra.py tests/test_luka_fullstat.py
35 passed in 28.05s
```

Coverage includes production population identity, exact proportional split,
unknown-label quarantine, sealed-row exclusion from particle reads, ordered
cache joins, bounded OFFLINE preparation, binary transfer authentication,
same-transfer retry and foreign-file preservation. It also checks the unchanged
CE recipe against the original kernel, exact epoch recovery including dropout
and optimizer state, R30/R50 thresholds, submission dependencies, ambiguous
submission handling, and wrong-memory/GPU rejection. Both shell entrypoints
pass `bash -n`; the CLI help command succeeds.

## Actual local CUDA miniature

`reports/luka_new_fullstat_baseline_validation/run_real_miniature.py` uses
44 TRAIN and 22 validation rows from the frozen new dataset, with all eleven
classes represented and no final-test or unused particle reads. It runs the
native model through the installed Weaver implementation on an NVIDIA RTX
2000 Ada Generation Laptop GPU. FP32 forward/feature/parameter-gradient parity
passed. Two training epochs, selected checkpoint reload, and validation replay
passed. The new recovery-capable runner produced exactly the original CE
kernel's validation history and bitwise-identical selected weights.

Evidence:
`reports/luka_new_fullstat_baseline_validation/real_miniature_v2/evidence.json`,
content hash `4dd771b3908f39d14a2d4dc1855e4690e48ba4f951e8e2a5c530de97d7331793`.
These tiny-set scores have no scientific performance meaning and do not
establish Hydra throughput or GPU capacity.

## Transfer confirmation and remaining execution evidence

Subsequently, the user supplied receiver output ending in `TRANSFER VERIFIED`
for the registered Hydra dataset directory, with progress over all 13,995
files. Remote upload completion is now supported by that terminal output.
The agent's independent unattended SSH attempt still failed authentication;
post-upload free space and scheduler admission have not been rechecked.
The following paragraphs retain the earlier failure and correction history.

The first user upload attempt failed before the receiver started: its SSH
invocation omitted the nondefault key used by the user's working login.
The corrected sender accepts `--identity-file`; the updated runbook supplies
the confirmed Brown key path. Ten focused transfer cases passed in 4.66s,
including key forwarding without reading key contents, EPIPE/EINVAL cleanup,
binary roundtrip, exact retry and corruption refusal. Dataset and transfer
manifest identities are unchanged. The subsequent user confirmation above
establishes successful transfer after this correction.

No remote transfer or Slurm submission was performed by the agent. The user
completed the transfer through interactive Brown authentication. Next,
commit/push the explicit workflow files and use the detached launch block.
Hydra performs a full dry resource review, CPU preparation and an integrated
installed-Weaver/real-data GPU qualification before fresh scientific training.
No full-data accuracy, AUC, R50 or R30 result exists for this baseline yet.
Event independence, overlap with the old dataset and producer label semantics
remain unverified. No test inference has been performed.
