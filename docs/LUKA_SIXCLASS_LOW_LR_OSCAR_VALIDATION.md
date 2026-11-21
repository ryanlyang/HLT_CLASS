# Local validation: lower-LR six-class Oscar mini campaign

Implementation is under `src/hlt_classification/luka_sixclass_low_lr_oscar/`.
No remote jobs have been submitted for this campaign.

## Evidence

Before edits, the original six-class campaign and full-Hbb evaluator suites
passed: **76 tests in141.05s**. The first expanded mini-campaign suite passed
**31 tests in69.25s**, including actual installed-Weaver CUDA acceptance for
all three production architectures on an NVIDIA RTX2000 Ada laptop GPU.
Final expanded regression: **111 passed in182.28s** (35 new-campaign tests,
54 original-campaign tests and22 Hbb-evaluator tests), including the actual
local CUDA checks. Python AST syntax, CLI help/import resolution and Bash
syntax checks pass. Original campaign modules and tests have no diff.

The scientific runtime for local checks is `tagging-hlt` plus the isolated
`artifacts/testing/hbb_weaver_053` installation of weaver-core0.5.3. Tests set
`PYTHONDONTWRITEBYTECODE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable pytest
cache and use a fresh workspace temporary directory.

Coverage includes exact native-classifier CE gradients, connected embedding/
auxiliary/matching heads, native export parity, immutable new-OFFLINE target
binding, same original draw seeds, 20-epoch continuation including optimizer
and RNG, finalization after export failure without retraining, disjoint pooled
validation, independent Hbb versus primary selection, censored/tied rejection,
historical-source replay and payload corruption, sequential submission,
ambiguous-submission refusal, and real Bash detachment with mocked commands.

The integrated GPU test uses the production model architecture and actual
FP32/BF16 forward/backward, installed-Weaver parameter/input-gradient parity,
miniature checkpoint reload, contrastive serial/parallel candidate parity,
and CUDA memory accounting. Synthetic variable-length input jets replace the
remote population. Linux RSS is stubbed only in this Windows test. It does not
establish Oscar full-population memory, throughput or queue availability.

## Storage witness

Production-architecture AdamW states were serialized locally after allocating
both moment buffers:

| Stage | Parameters | Bytes per complete model/optimizer epoch state |
|---|---:|---:|
| OFFLINE | 2,142,970 | 25,976,900 |
| DE000 | 2,311,186 | 28,027,044 |
| M0HLT | 2,142,970 | 25,976,900 |

All60epoch states total1,599,616,880bytes. The single new OFFLINE bank's
array payloads total955,549,760bytes. Uncompressed selected in/out predictions
for all three models total122,262,912bytes. Subtotal: **2.494GiB**, before
small headers, selected native checkpoints, diagnostic miniatures and reports.
The campaign reserves **8GiB** of new HOME output headroom. It creates no
persistent particle cache and copies no historical bank or raw dataset.

## Remote boundary

Exact pushed source, healthy matching persistent HOME runtime, original
completed teachers/population, at least8GiB current HOME headroom, and Oscar
scheduler dry review are required at launch. Each of the three science jobs
performs its own integrated acceptance. No separate preflight job is queued.
No Oscar or Tigris miniature or full-data training has been performed here;
real resource acceptance remains inside the authorized Oscar allocations.
