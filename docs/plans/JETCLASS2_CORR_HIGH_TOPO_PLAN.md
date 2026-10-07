# CORR_HIGH_TOPO: stronger correlated tracking plus particle removal

## Authority and scope

User authorized this combined recipe on 2026-10-07 after inspecting CORR_MID.
This is a new synthetic mechanism study, not a revision of the historical
CORR_MID recipe or a CMS response claim. The earlier instruction not to tune
CORR_MID after results continues to protect that artifact; this explicitly
named successor is exploratory and must be reported as such.

Materialize the **same 100k train / 50k validation rows** as the completed
CORR_MID release. Preserve its ordered identities and original offline source.
This study-sized dataset does not create another 2.25M production release.
No final-test particles, old-data deletion, old-job changes, or extra fits are
authorized. A larger export needs a separate explicit request.

## Frozen recipe

Use the existing CORR_HIGH endpoint (amplitude 2, four times MID added variance).
Verify the historical MID endpoint against replay before deriving any row:
physical structure and ineligible tracking exactly; eligible tracking within
fixed rtol=1e-12, atol=1e-12 mm for ARM/x86 floating-point roundoff only. Record
bitwise counts and maximum absolute/normalized differences, failing beyond
these bounds. Newly generated serial/process and readback parity remain exact.
The original correlated kernel and its assumed perigee geometry remain unchanged.
Then apply NOISE_V3-style independent soft drops and disjoint neutral merges:

- import exact p_drop/p_merge from a content-authenticated completed NOISE_V3
  calibration, without rounding, refitting, or looking at classifier metrics;
- eligible drops: original charged hadrons, neutral hadrons, photons, pT<2 GeV;
- merge candidates: surviving same-category neutral hadrons or photons,
  original delta-R<0.03, canonical random-priority greedy disjoint pairs;
- retain original IDs/charge; **no PID conversion or extra p4 smearing**;
- sum four-vectors for merges; merged neutral tracking is explicitly invalid;
- preserve singleton tracking masks/placeholders; no artificial nonempty rescue;
- fixed identity-keyed draws, invariant to process/chunk scheduling.

Merge candidates differ from NOISE_V3's post-PID candidates. Its mean of 38
particles is not a promised outcome. No metric or count target gates execution.

## Ladder semantics

OFFLINE is the original unmodified particle set. All D coordinates share the
final degraded topology. An authenticated construction-only mapping projects
original tracking onto surviving singleton rows; merges have invalid tracking.
For f=1/3,2/3,1 at D066,D033,D000, add f times the frozen HIGH tracking residual
and f-squared times its added error variance. D000 is the exact saved endpoint.
Topology is not interpolated or reconstructed by a deployable model. Ancestry,
seeds, source indices and dropped tokens never enter its 17 inputs.

Two CE controls (M0HLT, OFFLINE); direct OFFLINE->D000; coarse
OFFLINE->D066->D033->D000; six fresh fits, three reducers, aggregate/completion.
Same paired seeds, frontend, model and CE/KD schedule as CORR_MID. No ascending
duplicates or dense branch. Shared final D000 for all deployable comparisons.
This changes tracking strength and topology simultaneously, so cannot attribute
an observed gain to either separately or establish significance from one seed.

## Execution and acceptance

SPORC debug: source-pinned CPU materialization, foundation, genuine full-size
Weaver/A100 CE+KD/resource preflight, then separately reviewed science DAG.
Materialization includes serial/process replay on its first chunk, physical
readback, per-block hashes and exact identity/count accounting. Save train-only
offline/proxy diagnostics. Source, corruption and resource failures fail closed;
poor scientific outcomes do not. Dry run and exact authorization precede each
live submission. Never claim local synthetic tests establish RC acceptance.
