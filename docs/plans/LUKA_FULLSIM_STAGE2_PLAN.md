# Luka FullSim stage 2: particle audit and technical admission

## 2026-10-09 operational recovery amendment

The user authorized the recommended fix after SPORC preflight 21834332 reached
fusion and exhausted its 90000 MiB host-RAM allocation. Preflight v2 changes
execution, not the scientific recipe. Preserve prepared-v1, provisional mm,
error-only zero masking, all particles, assignments, model and training kernel.

Before full caches, reconstruct the first four TRAIN rows and first actual
maximum-length TRAIN witness per required view, using authenticated prepared
maxima, scalar cardinalities and assignments. Only selected TRAIN candidate
particles are decoded. Run installed/native-mask and FP32/BF16 offload parity,
and the unchanged longest-TRAIN batch256/three-step stress for all four probes.
Reserve full-cache headroom; fail stress before loading full populations.

Then build U000/D000/U050 together in one selected-particle pass per role,
rechecking every prepared-input fingerprint. Share these RAM caches across
probes and the D000 object across both HLT-pair branches. Full-population
one-pass CE/KD and checkpoint/TRAIN-bank readbacks remain mandatory.

V2 requires exactly 6 CPUs, one SPORC A100 and **256000 MiB host RAM**. This is
an unmeasured recovery envelope, not a sufficiency guarantee. Keep 85% host /
90% GPU headroom checks and phase timing/RAM logs. Set the deterministic cuBLAS
workspace before Python. Do not change model/offload math or weaken stress.

Explicit old/new-source reuse requires both clean worktrees authenticated and
every existing tracked `src/` file byte-identical. Only the three named new v2
execution modules may be added. Record distinct preparation/execution parents;
never rewrite old snapshots or prepared products. Historical v1 code remains
intact. No automatic retry/submission, science graph or final-test access.

## Original stage-2 scope

User authorization: implement stage 2, 2026-10-08. The producer has **not**
confirmed tracking units or zero-error semantics. Stage 1 remains authoritative
for source selection, natural class proportions and exact file-disjoint splits.
No prior proxy, Delphes, CMS Scouting or CONTEXT dataset is substituted.

Stage 2 has three separately invoked steps. No step submits follow-up jobs.

1. Audit all selected train/validation particles in stored units, independently
   for native HLT and offline, including per-PID tracking distributions, zeros,
   negative errors, nonfinite values, charge/PID consistency and p4 pathologies.
   This step requires no unit assumption. Diagnostic anomalies are reported;
   corrupt provenance, invalid joins or jagged lengths fail. No test particles.
2. After reviewing that evidence, register explicit offline/HLT length units
   and error/value sentinel policies, with evidence and the audit hash. Units
   cannot be inferred from distribution widths. Convert to mm, retain signs,
   and validate every selected physical particle. Negative errors, nonfinite
   required values, ambiguous PID and nonphysical p4 fail; no new row cuts.
   Zero-error policy explicitly chooses error-only invalidity or invalidating
   the associated value too. Neutral tracking is explicitly non-applicable.
   Label convention and unresolved cross-file event independence are acknowledged,
   never upgraded to producer confirmation by the implementation.
3. Build/verify full-cardinality salience correspondences and every coarse view,
   then run a real SPORC A100 GPU technical preflight with the same installed
   Weaver, native fusion, batch 256 and training kernel as the recent CONTEXT
   study. Full-population one-pass CE/KD probes and full validation are explicitly
   NONSCIENTIFIC. Probe probabilities are TRAIN-only, generated on this dataset,
   never imported from a prior scientific campaign. Measure resource use and
   conservative 100-pass projections; do not tune based on probe metrics.

Endpoints use the current 17-feature asinh/log1p frontend (mm), five known PID
flags plus all-zero unknown, own-view p4 axes, no constituent truncation, minimum
padding 16. Missing PID is explicit unknown; multiple PID flags fail. Source
`deta/dphi` are audited for finiteness but not used as model features: coordinates
are reconstructed from p4. No inference feature contains labels, row IDs,
source indices, assignment indices or correspondence validity.

Reuse the registered persistent-shell salience view algorithm and its existing
deterministic switch domain deliberately, under a NEW Luka view contract. The
name `proxy` in the donor function is an argument name only: D000 is the actual
native HLT endpoint, OFFLINE the native offline endpoint. U000 includes unmatched
HLT slots and is not the pure OFFLINE control. U050/U100 remove offline-only tail;
D066/D033/D000 progressively remove offline features. No smearing/generation.

Matching products bind row identity, source file, audit, conventions, foundation,
source snapshot, view and input contracts. Outputs are immutable, published
completion-last. Model caches are RAM-only. No test role, silent recovery,
automatic convention selection, training graph, scheduler mutation or science
authorization is provided. Technical success is not detector qualification or
proof of event independence. Real execution evidence is still needed after local
tests; stage 3 must bind the exact accepted products and measured environment.
