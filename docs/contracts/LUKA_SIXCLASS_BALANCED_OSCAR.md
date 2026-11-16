# Six-class balanced Oscar contracts v1

Authority: [scientific plan](../plans/LUKA_SIXCLASS_BALANCED_OSCAR_PLAN.md).
Implementation: `src/hlt_classification/luka_sixclass_balanced_oscar/`.
The `LUKA_SIXCLASS_BALANCED_OSCAR_* /v1` namespace is intentionally distinct from
the eleven-class `LUKA_NEW_BALANCED_OSCAR_* /v1` artifacts. Old artifacts are not
accepted as fresh six-class teachers, epochs, results, or recovery anchors.

## Identities and population

- Foundation: `2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`.
- Original extended role population:
  `7f2413152ecb588764c862b4b83bd6f900320a1b915a3b77f95540dd6e283939`.
- New six-class population:
  `5092ce94a5e32fac10bdbd5b91b1becddc3fe9123affc41a93a0b312db2ea616`.
- Transfer manifest:
  `7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4`.

The foundation and original unused-only extension are replayed unchanged under
their own contracts. `population/role_parent` stores that authentication evidence.
The new population filters the frozen extended roles by labels 0–5 and
`600 <= offline jet_pt < 2600`. It exports canonical TRAIN, in-range validation,
and outside validation identity/label/pT arrays only. Test scalar counts can be
audited; no test row arrays or particle access are exposed to a training worker.

The new `POPULATION` binds both foundation and extended-role parent hashes,
policy, counts, class counts, particle support, 6×20 TRAIN cell counts, the
parent report file reference and canonical row payload. Build/load replay all
scientific content. Rehashing modified rows or policies does not bypass replay.

Counts: TRAIN 1,658,937; validation 328,589; sealed test 822,313; unused 78,034.
Outside counts: TRAIN 1,253,533; validation 237,443; test 610,553; unused 103,443.
All classes and pT cells must be present; no empty-cell redistribution or caps.

## Models, targets and optimization

`MODEL` records the installed Weaver factory configuration, six class names,
17 input features, six outputs and native features/vectors/mask interface.
The encoder and 128-dimensional representation follow the pinned donor design.
Capture validates an exact 128→6 classifier; no discarded eleven-class outputs.
Auxiliary embedding, six-logit and final matching heads are absent from native
deployment checkpoints. No teacher output enters the native classifier.

`RECIPE` and `nodes` bind all settings in the plan: 20 epochs, 1,658,937 replacement
draws, 128 batch size, 57-row last batch, 12,961 updates, exact optimizer/schedule,
six labels, edges, main-head predecessor, auxiliary teacher order, initialization
and head seeds, target precision/standardization, contrastive definition, selection
and evaluation scope. `main_head` is CE1/KD0 for baselines, CE.25/KD.75 for all
three students; both KD paths use T2 and T² forward KL, with distinct heads.

`BANK` binds selected checkpoint/report, new spec/population/recipe, coordinate,
unique TRAIN row order, native128 embedding targets, six softened probabilities,
FP64 means/scales and payload checksums. Publication is atomic and completion-last.
OFFLINE keys are the FP32 unit-normalized TRAIN-standardized native embeddings.

`SIXCLASS_BALANCED_MATCH_OCCURRENCE/v1` uses epoch, draw position, identity,
component and stream in deterministic RNG. Hard proposals span the other five
classes; same-class/global quotas remain 256/512. Minimum available per-class
capacity is 820, sufficient for the largest five-class 4096-candidate quota.
Serial, vectorized and CPU-worker paths must agree, including ties and repeated
positive occurrences. Neither diagnostics nor probe training initializes a fit.

## Execution, evidence and recovery

`SPEC` pins pushed source, donor blobs, installed package/Weaver source fingerprint,
population, transfer and original producer lineage, resource/site policy, storage,
recipe and six-fit DAG. Persistent runtime is restricted to Oscar HOME environments;
scratch is raw storage, not the runtime. Reserve 14 GiB results. Runtime bootstrap
restores exact recorded builds only if absent, otherwise verifies its completion,
package payloads and installed fingerprint. Current free quota governs reuse.
Partial runtime restorations fail closed. Bootstrap never submits the old ladder.

`STARTUP_PHASE`/`STARTUP` bind installed-Weaver parity, two-epoch miniature/reload,
full-support stress, main-classifier gradient equality, native deployment parity,
contrastive serial parity, measured memory and projected walltime. These run in
each science job; there are zero separate preflight jobs. Site validation uses
`sbatch --test-only` before any real submission. Failed technical acceptance is
not a scientific result; poor accuracy or rejection is a valid scientific result.

`EPOCH_COMMIT` references complete optimizer/model/CPU-CUDA-RNG state, row loss
means, metrics, schedule and deterministic sampling evidence. Explicit continuation
authenticates job IDs and resumes committed epochs; an export failure after epoch
20 cannot repeat those updates. Submission journals, exact intent/receipt bindings
and a process lock prevent duplicate or ambiguous retries.

`METRICS` reports six-class macro OVR AUC/CE/accuracy and five signal-vs-QCD R50/R30
tables at the same AUC-selected checkpoint. Ties pass the threshold; zero passing
QCD is censored, with one-event resolution reported. Outside-range/per-bin metrics
are diagnostic. Fresh M0HLT/OFFLINE metrics on each identical subset define linear
recovery; missing/censored/nonpositive gaps are unresolved. Final test remains
sealed; no final-test CLI is supplied.
