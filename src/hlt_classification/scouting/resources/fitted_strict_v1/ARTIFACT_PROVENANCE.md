# Canonical artifact provenance

This directory is an exact copy of the complete, self-consistent 7,500-jet run
originally written to `../selective_matcher_profile/` on 2026-08-05. It was
promoted to a separate canonical directory after overlapping processes made
`../selective_matcher_results/` internally inconsistent.

The model and summary in this directory belong to the same run:

- sample: 7,500 jets;
- split: 3,750 train / 1,875 validation / 1,875 test;
- fitted model written before the six-candidate evaluation;
- candidate summary, confidence models, threshold scan, and campaign validation
  written together at campaign completion;
- post-campaign independent stricter audit written afterward.

SHA-256 hashes:

```text
campaign_validation.json    820C4540D369A72B57A378787CF939692CF8F399CA7621B1DA0B8617D3A935F5
candidate_summary.csv       85307D504D2D167B6A8D406B831C33D04A219E3FCFF042D499054DAA4A4498C2
confidence_models.json      DC1CE2F3F889BEF3E3C6CDC46DA70BC14E79DFF518D6D4B1D759FAC51F093735
fitted_edge_model.json      2CFD51B209435164263247252A35CF83708FA71FC2580301C35BB5D905D41142
threshold_scan.csv          A99038C0EF17BDDF363249555FF6A421215EE73448A70C57A85B963371038E91
independent_validation.json AF1D2B009E7C9186B454ACD461E4CF40335261AE4B491F1C82DD1957E941687E
```

The independent audit recomputes the mixed-event Wilson bound missing from the
older campaign schema and recommends `fitted_strict` under the stricter current
gate set.


