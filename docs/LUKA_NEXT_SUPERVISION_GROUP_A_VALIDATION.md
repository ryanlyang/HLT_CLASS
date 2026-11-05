# Group A validation and execution boundary

Implementation baseline: `ddcdd5fa22ce2455b12a62d61a926caa07969159`.
New package: `luka_next_supervision_hydra`; contract namespace
`LUKA_NEXT_SUPERVISION_*/v1`. Donors are recorded in
[LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md). Existing runtime is unchanged.

## Completed local checks

Local environment: Python3.10.19, Torch2.5.1, installed Weaver-core0.5.3 from
`tmp/hydra-weaver-fixture-20261102`, NVIDIA RTX2000 Ada Generation Laptop GPU.
These tests use synthetic data. They are not Hydra throughput measurements or
scientific recovery results.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_next_supervision_hydra.py
```

Final focused run: **60 passed in57.56s**. Coverage includes:

- Exact40-row registry and22 Group A configurations; unchanged U100 coefficient,
  matched seeds, original native initialization/RNG, FP32 installed-Weaver
  forward parity, and exact original-U100 optimizer-update parity.
- Parent-local coordinate/scale invariance, tracking validity, TRAIN-only raw
  export/relocation, per-member checksums, ordered joins and corrupt payloads.
- Full500-particle pools across all five non-mined laws, caps, exclusions,
  low-multiplicity anchors, population capacity, fixed4096 semi-hard proposals
  and exact original1024-negative J candidate identities, including J6.
- Intact unique pairs/triplets, same G1/G5 targets, member/candidate permutation,
  balanced per-jet BCE, masked singleton padding, no candidate-target gradients,
  and real native-encoder/auxiliary-head gradient flow.
- Full-candidate production objective fits, native-only save/reload, full30
  budget despite poor finite results, epoch/report/receipt tampering, frozen
  HLT TRAIN key rebuilds, and exact setup/science worker publication paths.
- Warmed resource projection/admission and support-equivalent measurement
  reuse; failed/running/unknown exact setup accounting cannot release science.
- Exact independent commands, reused controls, full40 rejection, real journal
  idempotence, ambiguous partial claims, transferred SPORC J5 authentication,
  read-only metric/recovery replay and provisional progress.
- Production-width CUDA128-row and partial batches for P/G/J heads.
- Chunked CPU dispatch and staged batched mining produce byte-identical targets
  to serial execution without consuming model RNG. Windows tests emulate the
  process transport; actual Linux fork/PSS accounting remains real Hydra setup.

Related regression command:

```powershell
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_particle_teacher_hydra.py tests/test_luka_u100_contrastive.py tests/test_luka_u100_contrastive_hydra.py
```

Result: **99 passed in74.09s**. Pre-edit focused baseline:89 passed with the
correct Weaver fixture path. An initial baseline command omitted that path and
failed on missing Weaver; no donor implementation was changed for that issue.
CLI help, Python AST, both Bash entrypoint syntax checks, documentation links,
fences and whitespace are also checked separately.
Final static checks passed for16 Python files, all20 required files and721
documentation links. Focused scaffold checks:3 passed,1 known scratch-link
scan deselected. Tracked-document `git diff --check` passed.

## Candidate construction cost

Serial timing exposed a CPU bottleneck before release. Batched NumPy parent
draws, bounded region caches and vectorized intact-group inventories replace
population scans/per-particle NumPy work. Four CPU-only fork workers share the
read-only TRAIN arrays on Linux, start before the student/autograd graph, and
return only NumPy candidates. CUDA mining stays in the parent and scores up to
eight queries together. Every anchor and original pool/proposal size remains;
there is no auxiliary-anchor subsampling or shorter scientific budget.

The miniatures time this complete path, including source loading, CPU workers,
IPC, mining and optimization. Host admission adds child proportional RSS to
the parent's peak RSS conservatively. Serial laptop timings do not justify a
Hydra walltime request; production measurement remains authoritative.

## Remote qualification still required

No new remote job, actual-data miniature, Tigris run or recovery measurement was
performed during this implementation. The explicit Hydra request specializes
the generic Tigris rule to the qualified RTX A6000 profile; real production
miniatures remain mandatory.

Commit/push clean source, export/copy the missing TRAIN measurements, register
the immutable spec and run the reviewed setup from the
[runbook](LUKA_NEXT_SUPERVISION_GROUP_A_RUNBOOK.md). It authenticates the old
A6000 qualification and new native parity, prepares TRAIN statistics/HLT keys,
proves capacity, and measures genuine warmed full/partial fits with the full
TRAIN candidate universe. Incompatible lineage, nonfinite data or resource
exceedances fail closed with diagnostics. Poor finite performance never rejects
a fit or cancels another row.

Science requires a separately reviewed plan after exact COMPLETED/0:0 setup.
It queues Group A's remaining rows independently in filtered global order,
reusing C0/C1 and compatible J5. B/C and the combined40-row release remain
unimplemented. No existing jobs, claims, output roots or shared data were changed.
