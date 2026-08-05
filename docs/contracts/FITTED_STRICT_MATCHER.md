# Fitted-strict constituent matcher contract

Contract: `hlt_classification_fitted_strict_matcher_v1`

The production implementation is
`src/hlt_classification/scouting/fitted_strict.py`. It ports the canonical
7,500-jet `fitted_strict` selective matcher and authenticates the vendored
edge, confidence, and independent-audit JSON before use. The stable entry
point is:

```python
from hlt_classification.scouting.fitted_strict import ConstituentMatcher

matcher = ConstituentMatcher.canonical()
result = matcher.match_jet(hlt_particles, offline_particles)
```

`result.match_index` has one entry per HLT token. `-1` means abstain,
`result.match_mask` identifies accepted pairs, and
`result.match_confidence` is zero for every abstention. Accepted offline
indices refer to the repository's native combined array, including the index
offset caused by excluded appended lost tracks.

The frozen algorithm is:

1. use all visible HLT candidates, regular charged offline candidates, and
   neutral offline candidates; exclude appended offline lost tracks;
2. compute every HLT/offline `deltaR`, `log(HLT pT/offline pT)`,
   `log(HLT energy/offline energy)`, PID transition, and charge transition;
3. score those values through the vendored empirical LLR tables and
   nonnegative meta-weights;
4. require `deltaR <= 0.15`, both absolute log responses `<= 2.5`, exact PID,
   and exact rounded charge;
5. solve the rectangular Hungarian problem with one private `-20` dummy per
   HLT row;
6. compute the ordered ten assignment diagnostics, calibrated confidence,
   and accept only confidence `>= 0.9828147479721088`.

The result is deliberately partial. Unmatched HLT tokens remain HLT and may
not be replaced by zeros, padding, or an arbitrary offline candidate. The
legacy `FULL_PARTICLE_ENDPOINT/v1` contract still requires 100% assignment and
therefore cannot consume this matcher. A separately versioned selective
all-field repair contract is required before the revised PMARD campaign can
launch.

Run the local artifact validator with:

```text
python -s scripts/validate_fitted_strict_matcher.py
```

Reference parity and boundary tests are in
`tests/test_fitted_strict_matcher.py`.
