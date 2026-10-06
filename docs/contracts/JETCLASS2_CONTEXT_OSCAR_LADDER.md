# CONTEXT_V1 relocated Oscar ladder contracts

Authority: [active plan](../plans/JETCLASS2_CONTEXT_OSCAR_100K_DIRECT_COARSE_PLAN.md).
This is a synthetic context-encoding experiment, not a detector-HLT response claim.
No frozen generator, pilot, particle bank, split registry or old frontend is edited.

## Version boundaries

The shared prefix below is `JETCLASS2_CMS_PROXY_LADDER_` (a software namespace,
not a claim that these particles came from the CMS-calibrated generator).

| Artifact | Context version | Meaning |
| --- | --- | --- |
| `JC2_CONTEXT_RELOCATED_READER` | v1 | External Oscar path resolution; immutable RIT provenance |
| `RELEASE_REQUEST`, `RELEASE`, `FOUNDATION`, `VIEWS` | v3 | Fresh CONTEXT population, pairing and matching |
| `INPUTS` | v2 | Shared 17-channel frontend with the frozen context tracking encoding |
| `GATE_SPEC`, `RUNTIME_PROFILE` | v8 | Oscar-native acceptance, CE and KD timing |
| `SCIENTIFIC_PLAN`, `CAMPAIGN_SPEC` | v4 | 100k/50k, controls plus DIRECT and COARSE |

Validators dispatch explicitly by version and compare canonical semantics, not
only hashes. Old CMS-proxy and NOISE_V3 versions retain their existing meaning.
The complete producer manifest hash is pinned to
`2ec2c933a37f50bb85e83577b9c1cf9d31c35a6e0be42089f0088ae62c485efb`.

## Relocation and ordinary-role access

`literature_context_consumer.RelocatedDataset` resolves the proxy, offline and
provenance roots separately. It never falls back to RIT paths or rewrites saved
JSON. It authenticates the complete manifest's metadata, source study, frozen
bundle, input contract, inventory, profiles and three pilot references. Requested
ordinary role releases and receipts must match the complete manifest. Bank and
ROOT bytes are checked on use. The standalone paired reader uses bounded ROOT
windows and checks the source again on early iterator closure.

Canonical identity is the producer identity over inventory hash, relative ROOT
path, exact tree cycle and raw entry. It is checked against each bank row, not
inferred from particle count, file sort order or array position alone. Particle
counts may differ; matching is freshly derived by SALIENCE_PT_LINEAR, never from
construction ancestry. Native offline p4 is GeV and tracking values/errors are
mm. Native Delphes HLT particle arrays are never read. Class labels are optional
reader metadata or explicit training targets, never selection keys or inputs.

Only `train` and `validation` can be decoded by this reader and campaign.
`final_test` requests fail before opening a role receipt or particle bank. A
complete manifest may describe all 2.25M jets without authorizing test inference.
User-reported opaque test-byte transport checks are distinct from test evaluation.

For a bounded standalone diagnostic after activating the scientific environment
and setting `PYTHONPATH` to the new checkout's `src`:

```python
from contextlib import closing
from itertools import islice
from hlt_classification.literature_context_consumer import RelocatedDataset
from hlt_classification.cms_proxy_ladder.context_inputs import build_inputs

data = RelocatedDataset.from_oscar_copy()  # Explicit Oscar defaults above.
with closing(data.iter_pairs("train", labels=True)) as pairs:
    for jet in islice(pairs, 16):
        x_hlt = build_inputs(jet.proxy)
        x_offline = build_inputs(jet.offline)
        print(jet.identity, jet.label, len(jet.proxy), len(jet.offline))
```

Both views refer to the same original jet. Their particle indexes are **not**
one-to-one; the classifier campaign derives its own authenticated assignment.
Standalone diagnostics do not authorize full test decoding or model selection.

## Model inputs

Use `cms_proxy_ladder.context_inputs` for **all** controls and ladder coordinates.
It calls the frozen `literature_context.transform.build_inputs`: tracking values
use `asinh(value/[0.1,0.2])/4`, errors use `log1p(error/[0.02,0.05])/2`, without
clipping. Other channels and own-view geometry retain the physical 17-input
adapter. No inverse context map is applied, and no latent context, source ID,
particle key, matching index or construction flag enters model features.
Capacity is 512; overflow fails, never truncates. D000 is the exact saved proxy;
OFFLINE is the native pure-offline control. The original physical frontend is
still imported by the frozen generator and must not be replaced globally.

## Execution and publication

Gate: CPU authentication -> CPU fresh foundation -> one Oscar L40S preflight.
Preflight measures the full selected population, installed-Weaver forward and
gradient parity, one CE acceptance pass, train T=2 probability publication,
validation prediction, one KD acceptance pass, and U000/D050 cache preparation.
Acceptance models are discarded and cannot initialize science. The larger CE/KD
pass time sizes the registered 100-pass upper envelope, with 1.75 timing margin.

Science contains nine fresh fits, five reducers and two reporting jobs. Each
student depends on its immediate parent's authenticated teacher bank. Same D000
rows, seeds and inputs are used by CE, direct KD and coarse KD. Poor metrics do
not block registered later fits. Stale source, nonfinite values, wrong identities,
failed parity and invalid resource shapes do fail closed.

Each phase first writes an exact dry plan and ledger. Live execution requires
that plan hash, the phase-specific authorization phrase, a clean pinned pushed
checkout and feasible Oscar scheduler requests. An exclusive durable submission
claim plus per-task journals prevents duplicate submission. Ambiguous interruption
is not auto-retried; preserve evidence for explicit repair. No cancellation,
partition migration, auto-followup or final-test job is part of this helper.
