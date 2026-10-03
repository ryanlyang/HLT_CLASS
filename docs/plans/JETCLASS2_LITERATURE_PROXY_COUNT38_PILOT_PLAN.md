# Literature proxy v2: count-targeted stronger pilot

## Authority and scope

This implements the user's approved separate v2: about 40.9 offline particles
to **38 proxy particles per jet on average**, roughly two soft losses and one
neutral merge, stronger PID/tracking response and modestly stronger kinematic
noise. It supersedes v1 only for new v2 artifacts. The v1 recipe and completed
pilot remain unchanged. This is a synthetic benchmark, not measured CMS HLT.
Numerical strengths are choices, not literature-derived detector efficiencies.

No CMS fit, native HLT, labels, classifier/KD scores, validation particles or
test particles enter this pilot. It does not qualify or generate production data.

## Exact population and input reuse

Reuse all 20,000 training jets from the completed v1 pilot at
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_literature_pilot_08cbf063_r1`.
Authenticate its spec, completion receipt, per-file reports, physical blocks and
lineage. Authenticate inventory/profile metadata and recompute v1's train-only
selection. Smaller completed v1 populations are allowed for implementation tests.
Load OFFLINE and NOMINAL physical arrays from those saved blocks, not ROOT or
CMS arrays. NOMINAL is the unchanged paired control. Preserve all rows, order,
units and masks. No new split, cut, particle truncation or class balancing.
The inherited parent population was label-stratified/HLT-matched; only this new
calibration and generator are label-blind. Keys reconstruct native offline order.

## Response and reproducibility

Use v1 crowding, original coordinates and canonical native-key ordering. Use
the same v1 operation streams for common standardized PID/tracking/p4 draws.
Versioned v2 operation domains supply new drop, pair-priority and merge draws.
Worker count/scheduling cannot change a jet's result in a fixed environment.

1. Multiply NOMINAL charged-hadron-to-neutral-hadron and electron-to-photon
   probabilities by **3**. Muons/unknown identities unchanged. Converted tracks
   become neutral with zero charge and unavailable tracking, not extra particles.
2. Multiply NOMINAL **added tracking noise amplitude by 3**: added variance is
   `9*((k*k-1)*sigma*sigma+b*b)`, k=1.5+0.5*c, b=0.020/0.050 mm.
   Add Gaussian noise to d0/dz, and set error to sqrt(old variance + added variance).
   Never multiply displacement itself by three. Preserve unavailable-error cases.
3. Multiply NOMINAL relative-pT and eta/phi noise amplitudes by **1.5**.
   Mean-one positive lognormal pT, Gaussian angles, mass-preserving p4 rebuilding
   and converted-category response are unchanged. No total-pT rescaling.
4. Independently drop each eligible ORIGINAL CH/NH/photon with pT<2 GeV with
   frozen probability p_drop. No one-drop cap. Electrons, muons and unknowns
   remain ineligible even after PID conversion.
5. Among remaining post-conversion NH/NH or photon/photon pairs with ORIGINAL
   deltaR<0.03, rank edges by fixed random priority and greedily form a maximal
   disjoint candidate matching. This is not a maximum-cardinality matching.
   Accept each candidate pair independently with probability p_merge. Sum their
   smeared four-vectors. Each original particle participates at most once, with
   no cascading/re-pairing after a rejected pair and no second-generation merges.

Original same-category neutral pairs may merge; cross-type/charged/far pairs may
not. No splitting, padding, hard-object forced removal, reclustering or invented
particles. Empty outputs are recorded, not rejected as a quality failure.

## Training-only count calibration, exactly two passes

All selected training jets contribute with equal jet weight. Let N be jet count,
P total input particles, E eligible soft particles, and D=max(P-38*N,0).
The intended drop budget is min(2*N,D). Set p_drop=min(1, budget/E), or zero
when E=0. Apply the frozen per-particle draws and new PID conversions. In a second
pass record the realized number of drops L and total disjoint candidate pairs M.
Set remaining merge budget R=max(D-L,0); p_merge=min(1,R/M), or zero when M=0.
Publish this calibration artifact **before** generating/reporting the comparison.

Only these two scalar count probabilities are calibrated. No iterative search,
per-jet count fixing, CMS matching, or discrimination optimization is permitted.
The predicted mean conditional on realized drops is (P-L-p_merge*M)/N; the final
mean fluctuates with merge draws. A capacity shortage or mean already below 38
is reported honestly: never add particles or relax eligibility to hit the target.
Report counts, probabilities, expected/realized loss, capacity shortfall and final
mean. Future populations need not have mean 38 with these frozen probabilities.

## Execution, outputs and next decision

One CPU-only Tigris job: 16 CPUs, 64 GiB, 2 hours; up to 16 per-file processes,
one thread per numerical library. v1 took 2m07s for 20k jets on job 218291;
v2 adds two bounded calibration passes, so that is context, not a runtime promise.
No extra submission stage between calibration and generation. Print pass/file
progress. Actual 64-jet serial/process replay precedes bulk generation.

Compare OFFLINE, NOMINAL (saved v1), COUNT38_V2 on identical jets. Save exact
moments, fixed-bin histograms with tails, PID/particle/jet counts, kinematics,
tracking values/errors/significances/masks and original-coordinate conditional
diagnostics. Count-delta bins now cover multiple losses. Save CSV, overlay PDF,
mechanism-selected examples, v2 physical blocks and separate diagnostic lineage.
Ancestry, keys and degradation metadata are forbidden model inputs and are not
an authorized replacement for a future ladder's particle matcher.

Content/byte hashes, source commit, parent receipt and per-file manifests bind
all artifacts. Publish a completion receipt last. Do not overwrite or delete old
outputs. Invalid input, drift, corruption or replay mismatch fails closed; poor
statistics and unmet count targets do not fail or discard rows.

Local tests precede a clean pushed worktree, explicit dry review and exact plan
hash authorization. Helper: `queue_jetclass2_literature_proxy_count38.sh COMMIT
PARENT_SPEC NEW_ROOT [--execute REVIEWED_PLAN_HASH]`. There is exactly one job,
no auto-followup and no classifier/final-test evaluation. Inspect this real
Tigris pilot before freezing any production recipe.
