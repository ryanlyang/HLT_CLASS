# JC2_CONTEXT_PRODUCTION/v1

Authority: [frozen production plan](../plans/JETCLASS2_LITERATURE_CONTEXT_PRODUCTION_PLAN.md).

New artifacts use `JC2_CONTEXT_PRODUCTION_<KIND>/v1`, schema 1, canonical
content hashes, parent hashes, immutable publication and truthful test flags.
Kinds include BUNDLE, STUDY, ATTEMPT, PLAN, TEST_BUILD_LOCK, PREFLIGHT,
RESERVATION, SHARD, ROLE_MANIFEST, MANIFEST and submission journals/LEDGER.
The reused CMS2JC2_PROXY_POPULATION/v1 is metadata selection only, never the
generator identity. Original literature pilot/production schemas stay unchanged.

BUNDLE authenticates the approved context pilot spec/receipt/report, original
calibration lineage, exact CONTEXT_V1 recipe, input contract, execution
environment and unchanged scientific source. STUDY binds this bundle, existing
TRAIN_1M population, original offline source, shards and storage attestation.
No runtime knob may select LOW_NOISE, modify response constants, refit rates,
change replicas, shrink the registered population or unseal test evaluation.

Physical NPZ banks contain exactly: offsets (int64, N+1), jet_identity (uint8,
N x 32), p4 (float64, M x 4; px/py/pz/E, GeV), charge/category (int8, M),
tracking (float64, M x 4; d0/dz/d0err/dzerr, native mm) and valid (bool, M x 4).
Invalid tracking is zero with explicit masks. Categories 0..5 are charged
hadron, neutral hadron, photon, electron, muon and unknown. CONTEXT tracking
scales are synthetic encodings, not calibrated uncertainties. No particle
ancestry, keys, labels or random-state features are exported.

SHARD binds study/attempt/shard and exact ordered membership, block hashes,
byte totals, physical digest, construction counts and inverse-integrity maximum.
Production requires raw inverse scaled error <=1e-10 per jet. Empty or unusual
physical distributions are not reasons to drop rows. No test distributions or
classifier metrics are exported. Manifests explicitly identify CONTEXT_V1 and
include the required unchanged `JC2_LITERATURE_CONTEXT_INPUTS/v1` contract;
readers reject a missing/different recipe or input contract.

Train/validation ROLE_MANIFEST files require all and only that role's shards;
they can be published before test completion without opening test banks.
The full MANIFEST requires exact 1M/250k/1M totals. Test-build locks grant only
deterministic construction, with final_test_evaluated always false. They are
not test-evaluation authorization. Public read_role refuses final_test.

Consumer example, from the source revision in study_spec.json (set dataset_root
to the new registered root first):

```python
from hlt_classification.literature_context_production.output import read_role
from hlt_classification.literature_context.transform import build_inputs
from hlt_classification.cms2jc2_response.bridge import Particles

for identity, physical in read_role(dataset_root, "train"):
    p = Particles(
        physical["p4"], physical["charge"], physical["category"],
        physical["tracking"], physical["valid"],
        tuple(str(i) for i in range(len(physical["p4"]))),
    )
    inputs = build_inputs(p)
    # inputs.features and inputs.vectors are unpadded arrays.
    # A separately registered training collator handles padding and its mask.
    # Join offline by identity; never feed identity or ordinal keys to a model.
    # Do not substitute the old tanh frontend.
```

Original offline entries are in study.population.files. Use the production
population.entries_for/ids/iterate helpers on train/validation shards and
assert the canonical identity join. Labels require a separately authorized
downstream reader over those same entries. Stored roots are producer paths;
relocation requires a new explicit authenticated adapter, not JSON path edits
or use of the old NOISE_V3 Oscar loader without namespace support.
