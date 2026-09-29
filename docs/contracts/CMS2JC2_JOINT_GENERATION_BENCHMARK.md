# CMS2JC2 JOINT generation benchmark v1

Authority: `docs/plans/CMS2JC2_JOINT_GENERATION_BENCHMARK_PLAN.md`.
Additive `CMS2JC2_RESPONSE_GEN_{PROTOCOL,MEMBERSHIP,STUDY,STAGE,BUNDLE,
GATE,RUN,REPORT}/v1`. All records bind content/parent hashes and source bytes,
are atomically published and retain `final_test_accessed=false`.

Training-only engineering authorization is separate from scientific transfer
qualification. Source/read capabilities must not be inferred from old DEV_STUDY
flags. The new GEN_STUDY and GEN_STAGE are explicitly validated by their own
dispatcher. Old scientific files and old immutable artifacts are unchanged.

Fitted JOINT parameters, random-key domains, replica 0, native units and physical
validity remain exact. 10k unique TRAIN rows; up to eight source files; at least
four. No JC2 HLT branches, jet labels, validation/test particles or CMS particles.
Population membership comes from authenticated existing train masks only.
Bounded offline-branch ROOT windows can decode unused TRAIN rows between mask
entries; only registered rows are generated, and no other role file is opened.

Benchmark NPZ stores offsets, identities, p4, charge, category, tracking and
valid; never construction keys. Per-row physical digests are independent of
process count/chunk size/compression. A separate generation-key digest checks
those keys without publishing them as model inputs. File checksum and exact
numeric readback precede output receipt publication. No pickle load.

Two explicit stages, pushed clean source, real allocation and exclusive claim,
durable intent/receipt ledger, bounded queues, measured admission and storage
guards. Incomplete or mismatched runs cannot produce a recommendation. Fastest
elapsed and best CPU efficiency are separate results. Projection excludes
scheduler wait and does not qualify the response or unlock a production test.
