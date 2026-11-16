# Balanced main-head KD validation

Implementation date: 2026-11-16.

Baseline before edits: the original balanced ladder's model/workflow/launch
suite passed 50 tests in 110.12 seconds, including local installed-Weaver CUDA
checks. The original balanced scientific modules are unchanged.

Final combined focused/regression validation: **79 passed in 168.93 seconds**,
with no skips:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_balanced_main_kd.py tests/test_luka_fullstat_balanced_main_kd_workflow.py tests/test_luka_fullstat_balanced_main_kd_launch.py tests/test_luka_fullstat_balanced_ladder.py tests/test_luka_fullstat_balanced_ladder_workflow.py tests/test_luka_fullstat_balanced_ladder_launch.py
```

Evidence includes the independently calculated T2 forward KL, exact deployed
classifier gradient equality to CE0.25 + KD0.75, nonzero KD gradients, identical
native/auxiliary initialization and RNG streams, connected finite gradients,
and native deployment independence from auxiliary heads. The full production
architecture passes installed-Weaver parity on local CUDA in FP32, with BF16
objective/deployment checks.

A real integrated-startup synthetic population (5,500 TRAIN rows) exercises
the full architecture, maximum fixture support, 1,024 negatives, serial versus
optimized candidate parity, three candidate worker processes, 3 stress steps,
68 timing updates, validation and probe discard. GPU memory/timing are measured;
Windows host RSS is substituted because Linux `/proc` is unavailable. This is
not a full-data Oscar timing measurement.

Production worker miniatures reuse five original native teacher outputs, train
the complete new objective for 20 short epochs, publish native selected metrics
and predictions, preserve every parent file, and generate no outgoing bank.
An epoch-10 continuation reproduces every later model state/loss exactly;
20-epoch continuation performs publication without updates. Admission/scheduler
and the tiny worker fixture's candidate population are mocked; the objective,
optimizer, checkpointing, prediction and metrics are real.

Registration tests cover the five references without original DE000 completion,
wrong/missing teacher and corrupted references, source-isolated authentication,
semantic drift, quota/root overlap, exact single-job dry/authorized submission,
duplicate/ambiguous submission, failure evidence and allocation mismatches.
The detached Bash test demonstrates background completion after its tail exits;
on Windows `setsid` is a fixture, not a real Linux HUP test. Bash syntax passed.

All 392 files in the conservative original production import closure match
donor commit `ff22b20f27fdc67d0fbd1a1df7005eb4626d19b7` byte-for-byte. No original
balanced scientific module was modified. CLI help, 18 new Python-file ASTs,
new Markdown links and whitespace checks passed. The full student has 2,314,411
parameters and an 8.83 MiB state; 20 model/two-Adam-moment tensor payloads
project to 0.517 GiB before metadata, RNG and selected predictions. The 6 GiB
output budget and 8 GiB minimum available quota retain ample headroom.

No real Oscar or Tigris allocation has been used for this follow-up, and no
jobs have been submitted. Queue readiness means the local implementation is
complete, not that remote qualification already passed. The requested workflow
does production resource/full-data acceptance inside the science allocation,
with no separate preflight job. Next: commit/push the documented files and use
the [detached launcher](LUKA_FULLSTAT_BALANCED_MAIN_KD_RUNBOOK.md).
