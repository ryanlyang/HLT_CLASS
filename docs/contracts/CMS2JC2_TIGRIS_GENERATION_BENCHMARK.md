# CMS2JC2 portable Tigris engineering contracts v1

Authority: `docs/plans/CMS2JC2_TIGRIS_GENERATION_BENCHMARK_PLAN.md`.
Additive PORT_{PROTOCOL,STUDY,STAGE,PACKET,EXPORT,GATE,RUN,REPORT}/v1 use the
CMS2JC2_RESPONSE prefix, immutable content/parent hashes and no final-test access.

Export must inherit a successful real GEN_GATE and its frozen 10k TRAIN
membership, original SPORC numerical environment, JOINT model/maps and source.
Portable packet binds source, review, membership, allocation, export stage,
ordered identities, all input/output block hashes and the complete frozen bundle.
Input keys are verified `part:i` and reconstructed, never taken from labels.
Only a trusted externally supplied packet SHA-256 permits cross-site import.

Import records a new environment without weakening old environment validators.
Exact code and discrete semantics are mandatory. Cross-site float tolerance is
fixed at atol=1e-12, rtol=1e-10; exact parity is reported separately. Same-site
worker-count and trace changes require exact parity. All 10k reference outputs
are checked in every screen run. Fail closed on drift, corruption or truncation.

Tigris has no hard-coded SPORC QoS: omit a QoS request, capture the site's assigned
QoS in scheduler evidence, and still require exact account/partition/resources,
job comment/command/workdir/owner and submission intent. Workers activate
`/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris`; numerical environment is
rechecked on the allocated node before reading particles. CPU-only, no GPU.

No old job/artifact writes, source rewrites, auto-migration, test capability or
production follow-up. Export/gate/screen are separate reviewed submissions.
Use existing durable DEV submission/output receipts and exclusive claims.
Portable output NPZs remain engineering artifacts, not deployable model inputs.
