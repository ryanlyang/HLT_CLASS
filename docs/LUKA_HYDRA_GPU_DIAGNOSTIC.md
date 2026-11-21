# Diagnose the four-GPU failure before another long campaign attempt

**Operator result2026-11-21:** original219050 and Gloo-only219051 failed known
tensor SUMs on one or more ranks. Gloo/no-P2P219052 completed0:0 in1m27 with
saved four-rank OFFLINE parity. This selects the no-P2P configuration as the
campaign candidate. The revised worker pins that environment and tests all9
arms before decoding. Use the [fresh-source queue commands](LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md);
there is no need to rerun the original three profiles. The commands below remain
available to read the original evidence. The helper bytes/hash are unchanged.

Job218567 completed full input construction, then failed during the first
qualification arm. Ranks1--3 waited20 minutes in an NCCL control-object broadcast;
rank0 had passed Weaver parity and entered OFFLINE full-weight parity. The
earlier stack now shows rank0 waiting in SyncBatchNorm while peers were in
that previous broadcast. The ranks were in different collective calls;
the stacks alone do not establish a hardware/P2P fault.
No arm proof or scientific fit completed. The next step is short diagnostics,
not another full input rebuild or larger timeout.

The new helper submits three five-minute jobs, serialized on gpu1905:

| Profile | What it checks |
| --- | --- |
| original | Original Context/NCCL communication, then production OFFLINE parity |
| gloo_control | Gloo CPU control plus a barrier after root callbacks; tensors remain NCCL |
| gloo_no_p2p | Same Gloo control, with direct GPU peer transfer disabled |

All use sixteen synthetic jets and the original authenticated untrained starts.
There is no dataset decoding, real-jet inference, scientific fit or training
release. All temporary updates are discarded. Five minutes is the hard runtime
limit per job; queue wait is separate. See the
[v1 contract](contracts/LUKA_HYDRA_GPU_DIAGNOSTIC.md).

Run in **local PowerShell**, no commit/push needed:

```powershell
Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\diagnose_luka_hydra_gpus.ps1
```

Enter the Brown key passphrase when prompted. The helper is uploaded, SHA256
checked and submission detached. Look for three SUBMITTED IDs and DIAGNOSTIC
OUTPUT. Do not launch twice. An uncertain submission preserves the intent and
requires inspection; retrying with the same helper refuses an existing output.

After the jobs finish, fetch the diagnostic summary and rank0 traces from the
same local project (without changing the helper between submit/status):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\diagnose_luka_hydra_gpus.ps1 -Status
```

The summary prints saved profile evidence, Python errors, rank0 stack samples
and exact Slurm states. A saved pass must also have COMPLETED/0:0 accounting.
GPU-stage and NCCL debugging output is in the printed output directory; every
rank has its own stack file so root-rank evidence is not lost in merged tails.

If original fails and Gloo passes, that narrows the issue to the original
control path. If only no-P2P passes, GPU peer transport is implicated. If all
pass, inspect the original full-data path before claiming resolution. These
comparisons are diagnostic evidence, not a performance ranking or proof that
an entire scientific campaign will pass. The helper does not change the old
execution's source or training settings.

The separate local worker repair uses that same Gloo control implementation
and NCCL_P2P_DISABLE=1, with a five-minute srun check before full decoding in each GPU allocation.
Qualification's early check covers all nine arm paths; real-input qualification
still follows. It requires a fresh pushed source/spec after the diagnostic
result is reviewed. Do not run the old long-queue command as a diagnostic.

Local checks cover bounded requests,
exact source/hash/parent/job binding, dry-review failures, uncertainty/duplicate
guards, control routing, no real data/training calls, absent evidence and each
rank-driver profile with mocked CUDA. The current test evidence is in
[validation](LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md). These do not
substitute for the actual four-GPU jobs. No remote submission was performed locally.
