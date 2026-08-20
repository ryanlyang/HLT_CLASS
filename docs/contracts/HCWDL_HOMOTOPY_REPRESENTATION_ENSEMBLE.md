# HCWDL Homotopy Representation Ensemble Contract

This validation-only post-hoc ablation evaluates the frozen three-member
probability ensemble at `D40`, `D20`, and `D0`:

```text
p_ensemble = (p_LOGIT + p_RSET + p_RREL) / 3
```

The weights are fixed before evaluation and cannot be tuned from validation
metrics. Each member must be the authenticated selected checkpoint at the
identical rung of the paired HCWDL-U-RKD campaign. The evaluator reconstructs
the shared authenticated validation view once, verifies each checkpoint's
standalone metrics against its immutable training report, and then computes
the ensemble metrics from the arithmetic mean of the three softmax vectors.

The report contract is
`HCWDL_HOMOTOPY_REPRESENTATION_POSTHOC_ENSEMBLE_REPORT/v1`. It binds the source
campaign, six member report/checkpoint hashes, M0 and TOFF baseline reports,
validation identity order, exact member order, weights, formula, recomputed
member parity, ensemble metrics, and recovery. `final_test_accessed` is always
false. These exploratory results do not mutate the source campaign, select a
campaign finalist, or become confirmatory evidence without a separately
predeclared repeated-seed study.
