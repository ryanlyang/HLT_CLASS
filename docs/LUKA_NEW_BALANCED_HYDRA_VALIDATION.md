# New balanced FullSim Hydra validation

Date: 2026-11-14. Implementation donor/base commit:
`a865e5b65bc0271d1359644b15956d98f94caca3`.
The [scientific plan](plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md) and
[v1 contract](contracts/LUKA_NEW_BALANCED_HYDRA.md) define this new dataset
adapter. The natural OFFLINE baseline and Oscar campaign are unchanged.

## Real population audit

The locally downloaded source is authenticated against the parent foundation
`2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`.
The new extension was built and independently replayed from saved scalar
memberships, including original split replay and every new role payload.

Frozen new population hash:
`7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`.
Production registration and loading must reproduce this hash.

| Population | In range | Outside range |
| --- | ---: | ---: |
| TRAIN | 2,425,536 | 853,628, excluded |
| Validation | 500,000 | 136,976, diagnostic |
| Final test | 1,200,000, sealed | 413,361, sealed |
| Unused | 689 | 205,760 |

All 198 TRAIN class/bin cells are nonempty. The smallest has 306 jets;
its expected appearances per jet per draw epoch are 40.03. Equal sampling
means equal probabilities, not exactly equal realized counts. Each epoch
records the actual cell counts and unique/repeated exposure.

The full-support maxima from scalar counts are:

| Population | HLT | OFFLINE | CONCAT |
| --- | ---: | ---: | ---: |
| TRAIN | 251 | 270 | 521 |
| In-range validation | 221 | 251 | 467 |
| Outside validation | 240 | 291 | 531 |

There is no 512-particle truncation. Full-support stresses use the registered
maxima in each science allocation. The audit writes about 150 MiB of new
population metadata. Evidence is under the local, uncommitted
`reports/luka_new_balanced_hydra_validation/population_v1/` directory.
Label 206 stays quarantined; event independence and old-dataset overlap remain
unverified. This audit does not read final-test particle arrays or evaluate
test predictions.

## Local implementation checks

- New focused tests: 28 passed, 2 deselected in 49.10 s after the final quota
  accounting and Bash-launch changes. The two separately executed CUDA
  integrated-acceptance tests passed in 35.33 s. These cover 30 distinct tests.
- Unchanged donor regression suites: 71 passed in 109.54 s.
- Synthetic ROOT tests exercise OFFLINE, D000, CONCAT, U100 and D050;
  every requested slice is checked against forbidden rows. Retained and fresh
  ROOT metadata readers produce identical tensors and matching hashes.
- The six production worker functions complete a reduced native-architecture,
  20-epoch fixture, with real losses, optimization, teacher banks, selected
  predictions, per-bin/outside metrics and fresh-anchor recovery. Scheduler,
  allocation acceptance and the fixture's small contrastive population are
  substituted; this is not remote acceptance.
- An interrupted epoch resumes with exact optimizer/RNG continuity and matches
  uninterrupted training bitwise. Export failure after all 20 committed epochs
  recovers publication with zero additional scientific optimizer updates.
- Submission tests cover exact teacher dependencies, duplicate/ambiguous
  submissions, terminal-job checks and completed-stage reuse. Storage tests
  reserve room for the separate natural baseline and exclude abandoned attempt
  files from the new campaign's committed-output quota credit.
- An actual Bash/nohup smoke test confirms that registration and submission
  finish after the tail/parent shell exits, and duplicate launch is refused.
  Python operations, Git HEAD and tail are fixture commands; this is not a
  remote SSH disconnect test.
- Full native installed-Weaver CUDA qualification runs for UE100 and DE000,
  including two-epoch full-objective miniature/reload, serial/optimized
  negative matching, actual candidate workers and main-classifier CE-only
  gradients. Windows host RSS is substituted; GPU measurements are real.
- Static checks pass: 14 Python ASTs, eight relative documentation links,
  whitespace, Bash syntax and LF line endings. Actual frozen parent producer
  bytes still authenticate to
  `5ac841f9f154838e3c0b7c473a5d9c085181ed6ee6c34e8577d92db20b0b4f85`.

The real-new-data six-stage CUDA miniature **passed** using the full native
architectures and installed Weaver. Evidence is under
`reports/luka_new_balanced_hydra_validation/real_miniature_v2/evidence.json`:

Content hash:
`7e1215acb1eeb261006795bb25fed071f6fba23f5003fa7cd07896dc888c3ba1`.

- 5,544 actual TRAIN jets (28 per class/bin cell), 22 in-range validation and
  22 outside-validation witnesses, chosen with bounded particle support for
  the laptop GPU; no test particles.
- OFFLINE, M0HLT, CONCAT, UE100, DE050 and DE000 each execute two reduced epochs
  of 44 balanced draws. All native parity checks, selected native reloads and
  five TRAIN teacher-bank exports/validations pass.
- U100 and D050 independently reconstruct identical physical matching hashes.
  The final student uses actual OFFLINE hard/same-class/global negatives and
  all four real native teacher banks.
- Restoring each completed fit executes no additional scientific updates and
  exactly reproduces its selected student state. Primary, per-bin and outside
  metric code executes on the same selected checkpoint.

This is a local technical probe of production components, not a scientific
fit, full-data timing measurement, or Hydra worker admission. A first technical
attempt completed OFFLINE and was interrupted during M0HLT preparation after
exposing repeated ROOT metadata opens. Its partial files are retained in
`real_miniature_v1/`. The metadata-only cache optimization passed tensor/mapping
parity tests before the successful six-stage run. No scientific result is
selected from these technical witnesses.

Reproduce focused checks in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_new_balanced_hydra.py
```

The local GPU is an NVIDIA RTX 2000 Ada Generation Laptop GPU. The existing
`tmp/r8-followup-weaver` installation supplies authoritative model parity.
No environment packages were installed or changed for this implementation.

## Hydra status and remaining acceptance

The user verified upload to
`/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1` and
submitted natural-baseline prepare 174642 / train 174643. At the last supplied
measurement, 29 GiB was available and that baseline's growing cache used
3.7 GiB. These are user-provided observations, not a live scheduler query.

New particle views are constructed only in RAM. The new result budget is
14 GiB, including all epoch optimizer/RNG states, five TRAIN banks and
selected predictions. The separate baseline retains a total 17 GiB budget;
only its remaining unwritten footprint is reserved. Registration, submission
and worker entry check remaining filesystem space and declared quota.

No balanced Hydra jobs were submitted, no remote GPU acceptance ran, and
no final-test inference occurred during implementation. The
[runbook](LUKA_NEW_BALANCED_HYDRA_RUNBOOK.md) contains selective commit/push,
detached launch, progress, result and recovery commands. Exact pushed source
and successful scheduler dry review are required. Each science job performs
installed-Weaver parity, full-support/gradient checks and conservative resource
qualification before starting or resuming its fit. The 48-hour request is a
ceiling, not a measured completion promise. No separate preflight job is needed.
