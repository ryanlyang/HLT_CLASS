# Shared-message Hydra validation

The implementation is additive relative to commit
`546077aa7fce428bf3f9f136ab2277ac5eed5f85`. No donor runtime or remote artifacts
are changed. New contracts are `LUKA_SHARED_MESSAGE_HYDRA_*/v1`.

Local validation uses the installed Weaver fixture at
`tmp/hydra-weaver-fixture-20261102` and the `tagging-hlt` Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_shared_message_hydra.py tests/test_luka_cross_message_hydra.py tests/test_luka_cross_message_a6000.py tests/test_luka_cross_message_ce_hydra.py
```

Final combined result: **126 passed in127.58s** (38 new shared-message tests and
88 existing cross-message/A6000/CE regression tests). Local Python3.10.19,
Torch2.5.1, Weaver-core0.5.3; CUDA tests used an NVIDIA RTX2000 Ada Generation
Laptop GPU. This is not the remote A6000 environment or a throughput claim.
Both Bash entrypoints passed `bash -n`; CLI help, AST parsing, required-file
coverage and whitespace checks passed. Existing donor runtime has no diff.

Covered invariants include native zero-injection parity, one shared prefix/two
shared downstream forwards, matched initialization, native pairwise-bias
gradients, both-producer versus detached-target agreement, all twelve production
objectives, full/partial CUDA batches with production widths, HLT-only deployment
replay, no privileged fields at deployment, agreement warmup/fade, exact-ID
bootstrap/release/report DAGs, no dependencies among science fits, idempotence,
ambiguous-submission rejection, source/input proofs and resource-only admission.
Exact-job worker tests execute fresh full fits and authenticate their receipts;
controls load no privileged particle tokens. The final miniature sampler keeps
the largest128 rows together and a partial batch of at least two jets, avoiding
the installed pair-embedding BatchNorm's one-value training edge case. Preparation
memory peaks remain included even when warmed GPU peak counters are reset.

Local synthetic fixtures do not establish scientific performance or A6000
throughput. Remote setup uses real authenticated rows, worst real multiplicities,
every class, full and partial batches, warmed optimizer steps and checkpoint
publication/reload for all twelve configurations. Source, installed software and
GPU are authenticated against the existing completed A6000 qualification. Only
TRAIN U100 features are extracted; validation/deployment uses HLT alone.

Remote execution remains pending until the new commit is pushed and queued.
No jobs were submitted or cancelled during implementation; no new recovery result
is claimed. The older generic Tigris miniature requirement is applied here to the
explicitly requested Hydra A6000 allocation; real production miniatures remain
mandatory before scientific release.
