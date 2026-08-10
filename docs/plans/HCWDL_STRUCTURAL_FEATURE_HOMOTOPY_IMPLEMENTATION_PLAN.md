# HCWDL Structural-Feature Homotopy Implementation Plan

Status: **implementation-authoritative supplemental plan; not yet implemented
or authorized for live submission**.

Short name: **HCWDL-UJ**. Here `U` denotes the upper structural-support
transition and `J` denotes the joint structural-plus-feature transition.

Activated by the user's explicit planning request on 2026-08-10. This plan
governs only new HCWDL-UJ artifacts. It does not mutate, relabel, or reinterpret
the implemented primary HCWDL graph, the ten-point or five-point dense cold
supplements, their completed Tigris outputs, or the native-offline TOFF oracle.
Those remain governed by:

- [`HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`](HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md);
- [`HCWDL_DENSE_COLD_PILOT.md`](../contracts/HCWDL_DENSE_COLD_PILOT.md);
- [`HCWDL_DENSE5_COLD_PILOT.md`](../contracts/HCWDL_DENSE5_COLD_PILOT.md).

The initial HCWDL-UJ execution is a validation-only
300,000-train/100,000-validation study. It imports the authenticated
unweighted 300k HCWDL parent read-only, runs no final-test task, and cannot
publish a confirmatory test claim.

## 1. Executive decision

The missing bridge between native offline and D100 is primarily structural,
not another ordinary unsmearing step:

```text
native offline support                         fixed HLT support
complete model-visible offline particle set -> D100 Shell Exact particle set
```

The new study represents that bridge with a deterministic residual-shell edit
path. Every edit atomically substitutes, removes, or inserts one complete real
particle record. No residual particle is continuously interpolated, no PID or
charge becomes fractional, and no fake track record is invented.

The main experiment compares two equal-depth cold-start paths:

```text
factorized: projected offline -> U010 -> ... -> U100
                                -> D90 -> ... -> D0 -> M1F

joint:      projected offline -> J005 -> ... -> J100 -> M1J
```

The factorized path changes structural support first and matched-particle
fields second. The joint path changes both coordinates together. Each path
contains exactly 20 predecessor-KD transition students plus one born-again M1
student. Both use the same data, architecture, update budget, optimizer,
unweighted loss, checkpoint selector, and transition-index seed pairing.

The factorized result is the primary mechanistic interpretation because it
isolates the structural and feature axes. The joint result is a fully
registered co-adaptation ablation, not an after-the-fact alternative.

## 2. Scientific questions and hypotheses

The ordered questions are:

1. Can a sequence of small structural-support changes transfer more of TOFF
   into exact D100 inputs than the direct one-step `TOFF -> D100offkd` edge?
2. Does a factorized `U then D` path retain more privileged performance at
   exact HLT than the existing direct or coarse descent?
3. Does the joint path outperform the factorized path by allowing feature and
   support co-adaptation, or does factorization win by isolating one mechanism
   at a time?
4. Does a final born-again HLT student retain or improve the exact-HLT endpoint
   reached by each path?
5. Are any gains specific to the changing views, or can they be explained by
   repeated same-view born-again distillation alone?

The predeclared hypotheses are:

- **H1:** factorized `U100` has higher validation macro OVR AUC than a paired
  direct `D100offkd` student.
- **H2:** factorized `D0F` and joint `J100` retain more of the M0-to-TOFF macro
  AUC gap than the imported coarse D0 control.
- **H3:** `D0F` versus `J100` and `M1F` versus `M1J` are two-sided comparisons.
  Neither path is declared the expected winner.
- **H4:** a view-changing path should outperform an equal-depth stationary
  same-input chain before the result is attributed specifically to homotopy.

Finite results against any hypothesis are valid scientific results. They do
not fail a job, remove a rung, change the path, or suppress a registered
descendant.

## 3. Imported parent and fixed population

The study imports one exact executable unweighted 300k HCWDL parent. Creation
must authenticate and bind:

- the parent `HCWDL_CAMPAIGN_SPEC/v7` and source commit;
- the whole-file-disjoint split and exact 300k/100k row selection;
- the `HCWDL_RECIPE/v4` all-ones unweighted loss lineage;
- train and validation `HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2` artifacts;
- the high-coverage assignment and Shell Exact qualification locks;
- the selected M0, D100, and TOFF reports and checkpoints;
- the architecture, schema, transform, matcher-resource, and repair hashes;
- the completed dense10/dense5 reports as contextual controls when their
  lineage is exact.

Assignments are read in place. They are not copied and the matcher is never
rerun. The new structural coupling is built once from the selected train and
validation rows and stored compactly. Repaired U/J particle datasets are never
persisted.

The screening seed is `1337`. The initial study uses train and validation only:

| Role | Rows | Use |
|---|---:|---|
| train | 300,000 | coupling calibration and model optimization |
| validation | 100,000 | checkpoint selection and all scientific comparisons |
| final test | 0 accessed | structurally absent from the campaign DAG |

The parent pilot's test role has already participated in earlier work and is
not silently reused as a fresh confirmatory holdout.

## 4. The representation boundary that must remain explicit

### 4.1 Canonical native TOFF

The existing TOFF oracle is not an ordinary 21-channel Scouting ParT. It uses:

- a charged `ParticleInputs` stream with 19 features and maximum length 90;
- a neutral `ParticleInputs` stream with 7 features and maximum length 60;
- separate charged and neutral Particle Transformers;
- a fusion classifier over their two outputs.

That architecture and tensor representation remain named **TOFF-native** in
this plan. Its checkpoint is imported unchanged and provides logits to the
first U and J students.

### 4.2 Projected offline carrier endpoint P0

Every U/J/D student uses the ordinary unified 21-channel, maximum-200
Scouting Particle Transformer. The structural path therefore needs a
single-stream offline endpoint. Define **P0** as follows:

1. take exactly the particles visible to TOFF-native after its registered
   bounds: the first at most 90 `cpfcandlt` entries and the first at most 60
   `npfcand` entries;
2. retain each particle's persistent native index in the full offline layout;
3. project every particle into the same 21 raw endpoint fields and four-vector
   convention already used by full Shell Exact repair;
4. apply the canonical HLT-feature transform only after the raw endpoint
   record is selected;
5. feed the resulting active set to the ordinary unified 21-channel ParT.

P0 preserves the complete **model-visible TOFF particle multiset and endpoint
values** under a unified projection. It is not byte-identical to
`NativeOfflineInputs`, does not use the TOFF-native architecture, and is not
allowed to inherit the old TOFF name or checkpoint.

In this plan, `U000 = P0` means exact particle-multiset identity after the
declared projection. It does not mean tensor, ordering, adapter, architecture,
or logit identity with TOFF-native.

Two mandatory adapter diagnostics make this visible rather than confounded:

- `P0CE`: a label-only unified ParT trained on P0;
- `P0KD`: a fresh unified P0 student trained from TOFF-native logits.

Neither diagnostic is an implicit predecessor of the primary U/J paths. By
the user's selected graph, `U010` and `J005` receive their first KD targets
directly from TOFF-native. Their first edge therefore includes the disclosed
native-to-unified representation change.

### 4.3 Exact D100 target endpoint B

For each visible HLT slot `i`, the imported high-coverage assignment provides
either a native offline index `j` or dustbin `-1`. Define the ordered D100
endpoint set `B`:

```text
B[i] = exact projected offline particle j   when assignment[i] = j >= 0
       exact observed HLT particle i        when assignment[i] = -1
```

`B` is the current `HIGHCOV_SHELL_EXACT/v1` alpha-one view. It has the exact
HLT token count, HLT slot order, mask, and padding. Assigned slots carry all
21 projected offline fields and offline p4. Dustbin slots remain all-field
exact HLT.

### 4.4 Exact HLT endpoint

The lower endpoint is the canonical HLT view. At the end of either complete
path:

```text
D0F  = exact HLT ParticleInputs
J100 = exact HLT ParticleInputs
```

Only the subsequent `M1F` and `M1J` are deployable scientific finalists in
this supplemental study. D0F and J100 consume HLT tensors but retain their
teacher-domain roles.

## 5. Exact per-jet endpoint partition

The implementation must not assume that every assigned D100 offline endpoint
is visible inside the 90/60 P0 bounds. Native indices use the repository's
full layout:

```text
0 ... n_cpfcands+n_lts-1                     cpfcandlt entries
n_cpfcands+n_lts ... +n_npfcands-1           npfcand entries
```

Lost tracks may appear in P0 but are excluded from high-coverage assignment.
Conversely, a legal assignment can theoretically reference a regular charged
or neutral particle beyond P0's bounded visible prefix.

For one jet, define:

- `A`: the exact P0 source particles with persistent native identity;
- `B`: the exact ordered D100 target tokens from Section 4.3;
- `C`: particles in `A` whose native offline identity is also used by an
  assigned D100 slot;
- `O = A \\ C`: P0-only residual offline particles;
- `R = B \\ C`: D100-only residual target tokens.

`R` contains two explicitly typed subsets:

- `R_hlt`: HLT dustbin tokens retained by D100;
- `R_off`: assigned offline endpoints that lie outside P0's 90/60 visible set.

The simplified interpretation `TOFF = C + O` and `D100 = C + H` is valid only
when `R_off` is empty. The implementation audits and reports `R_off`; it never
assumes it away.

Required identities and conservation are:

```text
C intersect O = empty
C union O = A
C intersect R = empty under endpoint identity
C union R = B
|A| <= 150
|B| <= 200
```

An HLT dustbin token is not declared the same particle as an offline source
merely because their numeric fields happen to be close.

## 6. Residual-shell coupling

### 6.1 Scientific meaning

The residual-shell coupling is a deterministic minimum-disruption edit
script between `O` and `R`. It is **not** physical constituent matching,
truth association, confidence calibration, or a claim that an HLT dustbin was
generated by a particular offline particle.

The imported high-coverage matcher remains the only correspondence system.
The new coupling answers an operational question:

> If one must transform the exact P0 particle multiset into the exact D100
> particle multiset through real-particle substitutions, insertions, and
> removals, what is the least disruptive deterministic one-to-one edit script?

### 6.2 Edit types

Let `n_pair = min(|O|, |R|)`. The coupling produces exactly:

```text
n_pair             O -> R substitutions
max(|O|-|R|, 0)    O -> padding removals
max(|R|-|O|, 0)    padding -> R insertions
```

Every substitution consumes one unique O and one unique R. Every residual is
used exactly once. Dummy edits arise only from endpoint cardinality imbalance;
the optimizer cannot choose a dummy instead of a real-real pair.

Each edit switches atomically. Before its switch, the slot contains the exact
source O particle or padding. After its switch, it contains the exact target R
particle or padding. All 21 raw fields, four-vector components, validity and
applicability state, identity, charge, quality, mask state, and padding state
move together.

### 6.3 Endpoint record used by the coupling

Every real endpoint is represented for coupling purposes by:

- physical p4 and derived finite `pT`, eta, phi, and energy;
- five-category identity and charge for offline/valid classified endpoints,
  plus the explicit `unclassified=-1` state allowed only for an observed HLT
  dustbin target;
- the 21 raw endpoint fields;
- the eight existing validity groups;
- charged applicability and track-validity state;
- persistent source native index or target HLT slot;
- target kind `assigned_offline` or `hlt_dustbin`.

Relative fields such as `phirel`, `etarel`, and the stored logarithmic scale
features are copied from their original endpoint. They are not recomputed from
the changing U particle set or from a newly derived jet axis. Recomputing them
would introduce a third global transition and would break exact D100 equality.

### 6.4 Frozen minimum-disruption cost

All coupling inputs are label-free. Jet class, labels, classifier logits,
validation metrics, and downstream endpoint performance are forbidden.

For real source `o` and real target `r`, define five bounded group distances:

```text
d_kin   = mean(
            min(deltaR / s_deltaR, 1),
            min(abs(log(pt_o / pt_r)) / s_logpt, 1),
            min(abs(log(E_o / E_r)) / s_logE, 1)
          )

d_id    = mean(category_o != category_r, charge_o != charge_r)

d_valid = mean over the eight validity groups of group-validity disagreement

d_track = group-balanced mean transformed-field displacement on common-valid
          track groups, with charged-applicability disagreement contributing 1

d_field = group-balanced mean absolute displacement for quality,
          relative-kinematic, scale-kinematic, and lost-inner-hit groups,
          divided by each channel's full clipped span
```

If both endpoints are track-inapplicable, `d_track=0`. If applicability
differs, `d_track=1`. If both are applicable but no track channel is valid at
both endpoints, `d_track=0` and `d_valid` carries the missingness difference.
Otherwise `d_track` is the mean over common-valid track channels after the
canonical clipped transform. A non-track field group with no common-valid
channel contributes zero to `d_field`; its missingness remains in `d_valid`.

The kinematic scales are frozen by a train-only, label-free calibration:

```text
s_deltaR = max(train residual-pair 90th percentile, 0.02)
s_logpt  = max(train residual-pair 90th percentile, 0.25)
s_logE   = max(train residual-pair 90th percentile, 0.25)
```

Percentiles use the deterministic complete 300k train selection and no class
weights. The v1 accumulator uses 65,536 fixed bins on `[0,5]` for delta-R and
`[0,8]` for each absolute log response; overflow enters the last bin. The
reported quantile is the upper edge of the first bin whose cumulative count
reaches 90% under the standard left-to-right order. Empty train residual-pair
populations use only the stated floors. Validation never refits a scale.

The selected v1 cost is:

```text
cost(o,r) = 0.30 d_kin
          + 0.20 d_id
          + 0.15 d_valid
          + 0.20 d_track
          + 0.15 d_field
```

Every group lies in `[0,1]`; therefore cost lies in `[0,1]`. A missing common
track measurement contributes through validity rather than a fabricated zero
residual. The exact field-group averaging, empty-group convention, percentile
algorithm, floors, and weights are hashed scientific configuration. Changing
one creates a new coupling family rather than editing v1.

An `unclassified=-1` HLT dustbin is a valid structural target and always
incurs category disagreement against an offline source. It remains byte-exact
HLT in every activated view and can never be reinterpreted as an assigned
offline identity.

### 6.5 Global solver and deterministic ties

For nonempty O and R, solve a rectangular Hungarian minimum-cost assignment
that pairs exactly `n_pair` real endpoints. Costs are quantized as:

```text
cost_q = round(cost * 1_000_000)
```

The primary objective minimizes total `cost_q`. Exact ties are resolved by a
stable lexicographic order over source native index, target HLT slot, target
kind, and target native index. Row, chunk, worker, and input permutation cannot
change the result.

No angular gate, PID gate, confidence threshold, or dustbin penalty is used.
Those would incorrectly imply another physical matcher. The solver is allowed
to pair distant residuals because an O-to-R pair is only an edit carrier; its
cost controls how late and how much that edit counts in the path.

### 6.6 Fixed carrier layout

The logical carrier has exactly:

```text
|C| + max(|O|, |R|) = max(|A|, |B|) <= 200
```

slots, so no U or J view requires hidden truncation.

Carrier placement is:

1. every common C particle occupies its eventual D100 HLT slot;
2. an O-to-R substitution occupies R's target HLT slot;
3. a padding-to-R insertion occupies R's target HLT slot;
4. unmatched O-to-padding removals occupy tail slots after the HLT skeleton,
   ordered by native offline index.

For model input, active logical slots are compacted deterministically: active
HLT-target slots in increasing HLT index, followed by active tail slots in
increasing native index. The mask is a contiguous active prefix and all
padding bytes are zero.

Consequences:

- at structural progress zero, the active multiset is exactly A, although its
  order is a deterministic carrier permutation rather than the native TOFF
  tensor order;
- at structural progress one, every HLT-target slot is active, every tail is
  padding, and the tensor is byte-identical to current D100;
- intermediate active count lies between `|A|` and `|B|` and never exceeds
  200;
- no source or target particle can appear twice.

The ordinary Particle Transformer is set-based, but carrier order is still
contractual and tested so endpoint equality and resume do not rely on an
informal permutation argument.

## 7. Information-mass coordinate for structural progress

Raw edit count is not a good U coordinate: deleting one soft neutral and
replacing a leading charged track should not count equally. Every edit receives
a positive, label-free information mass.

For edit `e`, let missing source or target pT/energy be zero. Within its jet:

```text
pt_share(e) = max(pt_source, pt_target)
              / sum_edits max(pt_source, pt_target)

E_share(e)  = max(E_source, E_target)
              / sum_edits max(E_source, E_target)

disruption(e) = cost(source,target) for substitution
                1.0 for insertion or removal

mass(e) = 1.0 + 4.0 pt_share(e) + 2.0 E_share(e) + 2.0 disruption(e)
```

Empty denominators contribute zero shares. The constant term preserves edit
count, pT and energy emphasize important particles, and disruption carries
identity, track, validity, and all-field changes. Mass is never a model input.

Construction uses three explicit passes to avoid circular calibration:

1. a train-only residual-pair pass freezes the kinematic scales in Section
   6.4 without solving or storing particle views;
2. source workers solve the train couplings once and publish compact staged
   edit shards containing cost and mass but no switch coordinate;
3. the staged train edits build a frozen 4,096-bin mass-weighted empirical CDF
   of quantized disruption on `[0,1]`; final train shards attach coordinates,
   and validation workers use the already frozen scale/CDF directly.

Within a disruption bin, a SHA-256 coordinate derived from the coupling
contract, canonical jet identity, edit kind, source index, and target index
spreads indivisible tied edits deterministically across that bin's train mass.
Staged edit shards are compact authenticated coupling metadata, not repaired
particle data. They are deleted only after final train shards validate.

Each train or validation edit receives a stored switch coordinate `u_e` from
that frozen train CDF. Validation uses the train histogram unchanged. The
compact artifact stores:

```text
switch_u16 = round(u_e * 65535)
```

For `0 < s < 1`, structural view `s` applies an edit when:

```text
switch_u16 <= round(s * 65535)
```

The endpoints are explicit branches:

```text
s = 0: apply no edits
s = 1: apply every edit
```

Thus edits are strictly nested. Once an edit switches, it never reverts at a
later U or J rung. The ten-point U rungs target approximately equal cumulative
train information mass, subject to indivisible edits; validation realizes the
frozen coordinate rather than rebalancing itself.

Reports must show, at every rung, realized switched edit count, information
mass, scalar pT, energy, category changes, charged/neutral changes, validity
changes, active-particle count, and per-jet transition distributions. Class
slices are reporting-only and never influence the coordinate.

## 8. One generalized view builder

The implementation must use one generalized raw-endpoint builder rather than
separate ad hoc U and J code paths. Define:

```text
V(s, f)

s in [0,1]   structural conversion from P0 support to D100 support
f in [0,1]   matched-field corruption from exact offline to exact HLT
```

The exact corners are:

```text
V(0,0) = P0 projected-offline particle multiset
V(1,0) = byte-identical D100 Shell Exact endpoint
V(1,1) = byte-identical canonical HLT input
```

`V(0,1)` is not a registered physical endpoint and is not needed by either
path.

### 8.1 Structural source and target selection

For a requested `s`, the builder first resolves every atomic edit from its
stored switch coordinate:

- common C slots are always active;
- an unswitched O-to-R slot contains exact O;
- a switched O-to-R slot contains exact R;
- an unswitched O-to-padding tail contains exact O;
- a switched O-to-padding tail is exact zero padding;
- an unswitched padding-to-R slot is exact zero padding;
- a switched padding-to-R slot contains exact R.

No RNG is sampled at view-build time. The structural decision is already
contained in the authenticated coupling artifact.

This structural layer always selects exact source/target records. The
independent feature layer in Section 8.2 may then apply the already registered
Shell Exact interpolation to assigned target slots when `f>0`. Therefore the
atomic-exact statement applies to the support edit itself; U views (`f=0`)
retain exact records, while D/J views intentionally retain the existing
mixed-field repair semantics.

### 8.2 Feature state at coordinate f

Let:

```text
alpha = 1 - f
```

For a carrier slot corresponding to an assigned offline D100 endpoint with an
HLT counterpart, use the existing `HIGHCOV_SHELL_EXACT/v1` strength and shared
discrete repair seed at that alpha:

```text
a_i(alpha) = 0                                  alpha = 0
             alpha ** (2 - 1.3 q_i)             0 < alpha < 1
             1                                  alpha = 1
```

The exact existing all-field semantics apply:

- p4 and valid continuous values interpolate using `a_i`;
- phi follows the wrapped shortest path;
- identity, charge, quality, lost-inner-hits, and validity groups use the
  existing identity-bound nested switches;
- track applicability follows the identity decision coherently;
- explicit alpha endpoints directly select HLT or offline values.

Other structural particles do not acquire invented counterparts:

- an O particle still present before removal remains exact projected offline;
- an activated `R_hlt` dustbin remains exact HLT;
- an activated `R_off` assigned endpoint uses its declared HLT-slot counterpart
  and follows Shell Exact at the current `f`;
- padding remains exact zero.

At `s=1`, this construction must be byte-identical to the existing PMARD
Shell Exact builder at `alpha=1-f`, including discrete groups. The shared
repair seed, assignment manifest, confidence quantization, raw projection,
and endpoint branches are the same parents. A second implementation that is
merely numerically similar is not accepted.

### 8.3 Projection and axes

The raw 21-field offline projector currently embedded in `repair.py` becomes a
public reusable helper. U/J code must call that helper. It cannot duplicate a
second table of feature suffixes, validity groups, neutral fill behavior,
normalization, or identity checks.

Particle-local stored relative and log fields are copied from the selected
endpoint and then transformed canonically. The changing active set does not
redefine the jet axis, recompute constituent-relative variables, or alter a
particle that is supposed to remain common. This keeps the U axis purely
structural.

### 8.4 Exact U pipeline

Define:

```text
U000 = V(0.00, 0)
U010 = V(0.10, 0)
U020 = V(0.20, 0)
...
U100 = V(1.00, 0)
```

The trained U pipeline begins at U010; U000 is the untrained semantic endpoint
used for audits and the P0 controls. The teacher chain is:

```text
TOFF-native -> U010 -> U020 -> ... -> U100
```

At every U rung:

- all currently active particles are exact observed offline or HLT endpoint
  records;
- no active field is between endpoints;
- the only change from the previous U rung is a nested set of complete
  structural edits;
- all common C particles remain exact offline;
- U100 is byte-identical to imported D100 input tensors.

This is the central upper bridge. It asks whether a model can carry native
offline knowledge through gradual support changes before any matched field is
corrupted toward HLT.

### 8.5 Exact factorized D continuation

After U100, hold structural support fixed at D100 and traverse the existing
Shell Exact field coordinate:

```text
D90F = V(1, 0.10)   alpha 0.90
D80F = V(1, 0.20)   alpha 0.80
...
D10F = V(1, 0.90)   alpha 0.10
D0F  = V(1, 1.00)   alpha 0.00 = exact HLT
```

These names retain the established D convention: D90 means nominal Shell
Exact alpha 0.90, not 90% realized feature-information mass. All D rungs use
one shared identity-bound repair seed, so discrete all-field switches remain
nested.

The factorized teacher chain is therefore:

```text
TOFF-native
  -> U010 -> U020 -> ... -> U100
  -> D90F -> D80F -> ... -> D10F -> D0F
  -> M1F
```

`D90F` receives KD from U100 evaluated on U100/D100. `D0F` receives KD from
D10F evaluated on D10F. `M1F` is a separate fresh HLT student taught by D0F
on exact HLT.

## 9. Joint structural-plus-feature path

The joint path follows the diagonal of the same two-dimensional builder:

```text
J005 = V(0.05, 0.05)   structural s=0.05, Shell alpha=0.95
J010 = V(0.10, 0.10)   structural s=0.10, Shell alpha=0.90
...
J095 = V(0.95, 0.95)   structural s=0.95, Shell alpha=0.05
J100 = V(1.00, 1.00)   exact HLT
```

The teacher chain is:

```text
TOFF-native -> J005 -> J010 -> ... -> J100 -> M1J
```

At a joint rung:

- structural edits with `u_e <= s` have occurred;
- surviving O particles remain exact projected offline until removed;
- activated HLT dustbins are exact HLT immediately;
- common and activated assigned-offline slots use Shell Exact at
  `alpha=1-f`;
- every discrete/validity group uses the same shared nested repair coordinate;
- J100 is byte-identical to HLT, not merely set-equivalent.

The primary joint coordinate is deliberately simple and reviewer-readable:
equal nominal progress in the frozen structural-mass coordinate and existing
Shell Exact alpha coordinate. It is **edge-count and update-budget matched** to
the factorized path, not claimed to be physically equal arc length. Shell
Exact's confidence warp and discrete switches make realized feature movement
nonlinear.

Every report therefore includes realized per-edge structural mass, normalized
21-field displacement, switched discrete-group mass, pT displacement, active
count, and measured GPU time. An arc-length-remapped path would be a separately
versioned ablation; it cannot replace this table after results are seen.

## 10. Exact primary node registries

### 10.1 Factorized registry

The factorized registry contains 21 new cold-start students:

| Transition | Node | Student view | Sole teacher and teacher view |
|---:|---|---|---|
| 1 | U010 | `V(.10,0)` | TOFF-native on native offline |
| 2 | U020 | `V(.20,0)` | U010 on `V(.10,0)` |
| 3 | U030 | `V(.30,0)` | U020 on `V(.20,0)` |
| 4 | U040 | `V(.40,0)` | U030 on `V(.30,0)` |
| 5 | U050 | `V(.50,0)` | U040 on `V(.40,0)` |
| 6 | U060 | `V(.60,0)` | U050 on `V(.50,0)` |
| 7 | U070 | `V(.70,0)` | U060 on `V(.60,0)` |
| 8 | U080 | `V(.80,0)` | U070 on `V(.70,0)` |
| 9 | U090 | `V(.90,0)` | U080 on `V(.80,0)` |
| 10 | U100 | exact D100 | U090 on `V(.90,0)` |
| 11 | D90F | Shell Exact alpha .90 | U100 on exact D100 |
| 12 | D80F | Shell Exact alpha .80 | D90F on D90F |
| 13 | D70F | Shell Exact alpha .70 | D80F on D80F |
| 14 | D60F | Shell Exact alpha .60 | D70F on D70F |
| 15 | D50F | Shell Exact alpha .50 | D60F on D60F |
| 16 | D40F | Shell Exact alpha .40 | D50F on D50F |
| 17 | D30F | Shell Exact alpha .30 | D40F on D40F |
| 18 | D20F | Shell Exact alpha .20 | D30F on D30F |
| 19 | D10F | Shell Exact alpha .10 | D20F on D20F |
| 20 | D0F | exact HLT | D10F on D10F |
| 21 | M1F | exact HLT | D0F on exact HLT |

### 10.2 Joint registry

The joint registry contains 21 new cold-start students:

```text
J005, J010, J015, ..., J095, J100, M1J
```

For transition index `k` from 1 through 20:

```text
t_k = k / 20
student J(5k) view = V(t_k, t_k)
teacher = TOFF-native when k=1, otherwise J(5(k-1)) on V(t_(k-1),t_(k-1))
```

`M1J` is transition 21, consumes exact HLT, and receives KD from J100 on exact
HLT.

### 10.3 Equal-depth meaning

The two primary paths each have:

- 20 privileged transition students;
- 20 predecessor-KD edges from TOFF-native to exact HLT;
- one additional HLT-to-HLT born-again M1;
- identical per-node train rows, passes, validations, batch size, and update
  count;
- paired initialization and stochastic seeds by transition index.

Variable active particle counts mean actual FLOPs and walltime may differ.
Reports say **edge/update matched**, not exactly compute identical, and publish
measured GPU-hours.

## 11. Required controls

### 11.1 Imported contextual controls

The aggregate imports, without retraining or relabeling:

- M0, the ordinary CE-only HLT baseline;
- D100, the label-only Shell Exact endpoint;
- TOFF-native, the native offline oracle;
- completed dense10 and dense5 rungs with exact lineage;
- their existing direct `D100offkd` results as contextual evidence.

Imported results are not treated as paired if their seed or source lineage
differs from this study.

### 11.2 New paired endpoint controls

The new study trains:

- `P0CE` and `P0KD`, defined in Section 4.2;
- `D100direct`, a fresh exact-D100 student taught directly by TOFF-native;
- `D0direct`, a fresh exact-HLT student taught directly by TOFF-native;
- `M0self`, a fresh exact-HLT student taught by a paired CE-only M0.

`D100direct` shares the U100 endpoint seed and sampler trajectory.
`D0direct` shares the transition-20 endpoint seed used by D0F and J100.

### 11.3 Stationary-depth controls

Without stationary controls, U100 could beat a one-step D100 student simply
because it received ten rounds of born-again optimization. The campaign
therefore registers two same-view chains before any results are inspected.

**D100 stationary chain:**

```text
TOFF-native -> S100_01 -> S100_02 -> ... -> S100_10
```

All ten students consume exact D100. `S100_01` is the stationary chain's
one-edge direct control; each later student learns from the preceding
same-view D100 teacher. It is distinct from `D100direct`, whose initialization
is endpoint-paired to U100 rather than transition-paired to U010. Compare U100
primarily with `S100_10`, and secondarily with both one-edge controls.

**HLT stationary chain:**

```text
TOFF-native -> S0_01 -> S0_02 -> ... -> S0_20 -> S0_21
```

All students consume exact HLT. `S0_01` learns directly from TOFF-native and is
distinct from endpoint-paired `D0direct`; later students learn from the
preceding HLT teacher. Compare D0F and J100 with `S0_20`, and compare M1F and
M1J with `S0_21`.

These chains control optimization depth, repeated self-distillation, and the
number of fresh restarts. They do not change the equal-depth primary graphs.

## 12. Training protocol

### 12.1 Shared optimization recipe

Every non-root primary or stationary node uses the locked unweighted primary
single-teacher recipe:

```text
loss = 0.25 * unweighted per-jet CE + 0.75 * forward-KL KD
privileged or TOFF-native teacher temperature = 2
HLT same-view teacher temperature             = 1
peak learning rate                            = 3e-4
effective/microbatch                          = 256/256
gradient accumulation                         = 1
AdamW betas/epsilon/weight decay               = .9/.999, 1e-8, .01
gradient clipping                             = disabled
schedule                                      = 5% warmup, cosine to 5% peak
model forward                                 = BF16
CE, logits, softmax, KL, coefficients, means  = FP32
passes                                        = 60 complete natural passes
validation                                    = after every pass
```

`P0CE` and the paired M0 root use unweighted CE only. Coefficients remain
constant. There is no performance early stopping.

For temperature routing, the teacher's declared input domain governs:

- TOFF-native, P0, U, J before exact HLT, and D alpha-positive teachers use 2;
- D0F, J100, and stationary HLT teachers use 1 when teaching the subsequent
  same-HLT born-again child;
- D0F and J100 themselves still receive temperature-2 KD from their richer
  immediate predecessors.

### 12.2 Cold initialization

Every new student is cold-started. It loads no predecessor weights. Knowledge
enters only through logits. Optimizer, schedule, scaler, sampler, and RNG state
start new at each node.

This plan has no warm path. Adding warm initialization creates a separately
identified ablation.

### 12.3 Paired transition seeds

The seed domain is based on `(campaign_seed, transition_index,
architecture_hash)`, not the human node name. At each primary transition:

- factorized and joint students share initial parameters;
- sampler order, dropout, augmentation, optimizer RNG, and validation order
  are paired;
- the corresponding stationary-depth control shares the same transition seed;
- only the declared input view and teacher lineage differ.

U100, `D100direct`, and `S100_10` also receive an explicit endpoint-paired
comparison seed record. D0F, J100, `D0direct`, and `S0_20` do likewise. M1F,
M1J, `M0self`, and `S0_21` share the M1-capacity endpoint seed domain.

Pairing does not imply identical floating-point execution when active particle
counts differ; it removes avoidable stochastic confounding.

### 12.4 Checkpoint selection and resume

Every node completes all 60 passes and emits exactly 60 validation rows.
Checkpoint selection uses the established total order:

1. highest validation macro OVR AUC;
2. lowest multiclass CE;
3. highest mean log QCD rejection at 50% signal efficiency;
4. earliest optimizer update;
5. canonical checkpoint identity.

The final partial batch is retained and every selected train identity appears
once per pass. Selected and pass-60 model-only checkpoints are retained, plus
one rolling exact-resume checkpoint.

Resume restores model, optimizer, schedule, scaler, sampler cursor, update,
validation history, best-state selection, interval loss, and all CPU/CUDA RNG
states. Device-mapped generator states are validated as one-dimensional uint8
tensors and normalized to contiguous CPU storage before restoration. A source
fix cannot be applied by requeueing an immutable job under its old commit.

## 13. One-time coupling, views, and teacher targets

### 13.1 Persistent coupling only

The durable structural artifact stores the compact edit script, not particle
features. The selected v1 ragged payload is:

```text
entries                       int64
row_offsets                   uint64
edit_kind                     uint8    substitution/removal/insertion
source_native_offline_index   int16    -1 for padding source
target_hlt_slot               uint16   65535 for padding target
target_kind                   uint8    assigned_offline/hlt_dustbin/padding
target_native_offline_index   int16    -1 unless target is assigned offline
cost_q                        uint32   round(cost * 1e6)
mass_f32                      float32  diagnostic/validation value
switch_u16                    uint16   frozen nested structural coordinate
```

An implementation may use an equivalent smaller signed representation only
if the exact logical arrays above remain reconstructible and logically hashed.
The sentinel values, dtypes, endianness, and array order are contractual.

One shard is published per authenticated source unit and role. Train and
validation receive separate manifests under one coupling lock. Final test has
no coupling artifact in this campaign.

Every shard and manifest binds:

- source path/tree/file hash and selected entries;
- split and row-selection hashes;
- parent high-coverage assignment shard/manifest and assignment lock;
- matcher resources and Shell Exact qualification lock;
- Scouting schema, native-index, TOFF-bound, p4, 21-field projection, validity,
  and transform hashes;
- coupling cost config, train scale-calibration artifact, mass formula,
  CDF histogram, hash-tie domain, and solver version;
- producer source commit, logical array hashes, byte hash, and atomic
  publication record.

Path existence never authorizes reuse.

### 13.2 Full-role authorization

The coupling lock requires complete train and validation scans. For every
selected jet and in aggregate it proves:

- scanned rows equal expected selected rows;
- A, B, C, O, R, `R_hlt`, and `R_off` counts reconcile exactly;
- every A residual is consumed once and every B residual is produced once;
- no common or residual endpoint is duplicated;
- substitution/removal/insertion counts equal the cardinality equations;
- every registered U/J rung has active count at most 200 and no truncation;
- U000 reconstructs the exact A multiset and all projected fields;
- U100 is byte-identical to imported D100;
- J100 and D0F are byte-identical to HLT;
- switch coordinates are finite, nested, and generated from the train-only
  calibration;
- labels and final-test branches were not read.

Perfect conservation over 99,999 of 100,000 expected validation jets is not
complete and cannot authorize training.

A deterministic identity sample is recomputed from ROOT and the parent
assignment before locking. Count-only full-role audit and sampled all-field
recomputation are separate requirements; a bounded sample cannot replace the
complete count audit.

### 13.3 Process-local view caches

Each training job builds its student train and validation view exactly once
from ROOT, imported assignments, and stored coupling. It stores canonical
FP32 features/vectors, boolean masks, int32 lengths, labels, and identities in
process RAM and replays the existing deterministic epoch schedule for all 60
passes.

The cache header adds:

- coupling manifest and lock hashes;
- `V(s,f)` coordinate, hexadecimal floats, and coordinate-table hash;
- projection, carrier-order, structural-switch, and shared repair-seed hashes;
- endpoint equality audit hash;
- exact array bytes and safe-memory calculation;
- `durable_repaired_dataset=false` and `matcher_callable_present=false`.

Before allocation, the exact shape estimate must fit below the smaller of the
configured cache cap and 75% of the Slurm memory request. No undocumented
truncation or alternate sample is a fallback. An authenticated assignment-
and-coupling-backed streaming fallback may rebuild the identical view without
rematching.

### 13.4 Teacher targets

Each frozen teacher is evaluated once over canonical train identities on its
own declared domain. Fifteen FP32 logits are joined to the student only by
canonical jet identity. The teacher object and temporary teacher view are
released from GPU after target construction.

Examples:

- U010 targets: TOFF-native on native charged/neutral inputs;
- U020 targets: U010 on `V(.10,0)`;
- D90F targets: U100 on exact D100;
- J050 targets: J045 on `V(.45,.45)`;
- M1F targets: D0F on exact HLT.

A missing, duplicate, reordered, wrong-domain, nonfinite, wrong-class-order,
or wrong-checkpoint target fails closed.

Compact durable logit caches are allowed only by a versioned last-consumer
contract when one teacher has multiple registered consumers or measured
preemption cost justifies it. They contain identity-ordered FP32 logits only,
never particle inputs.

## 14. Scientific comparisons and reporting

### 14.1 Primary metric

Validation macro OVR AUC is primary because it has been the most stable signal
in the current ladders. Multiclass CE and mean log QCD rejection at 50% signal
efficiency are required co-reported metrics, but an R50 fluctuation alone does
not select or redefine the path.

Required secondary metrics include overall and balanced accuracy, per-class
OVR AUC, Xbb/QCD and Xcc/QCD false-positive rate and rejection at 50% signal
efficiency, Brier score, top-label 15-bin ECE, confusion matrix, and natural
class counts.

### 14.2 Recovery quantities

For larger-is-better metric `m`:

```text
recovery(X; M0, TOFF) = (m_X - m_M0) / (m_TOFF - m_M0)
```

For CE the sign is reversed. A zero or numerically unresolved denominator is
reported undefined. Recovery may exceed 100% or be negative; it is never
clipped.

Every rung reports:

- M0-to-TOFF recovery;
- retention relative to its immediate teacher;
- gain over the paired stationary-depth control at the same transition;
- factorized-minus-joint paired difference where transition endpoints are
  comparable;
- realized structural and feature displacement;
- active count, pT, energy, category, track, and validity transition summaries;
- selected pass, full 60-pass history, and measured GPU-hours.

### 14.3 Ordered comparisons

The primary comparison table is frozen before execution:

1. `P0CE` versus TOFF-native and imported TOFF metrics: representation/capacity
   diagnostic, not an endpoint claim;
2. `P0KD` versus `P0CE`: transfer across the native-to-unified adapter;
3. U100 versus `D100direct` and `S100_10`;
4. D0F versus imported coarse D0, `D0direct`, and `S0_20`;
5. J100 versus `D0direct` and `S0_20`;
6. D0F versus J100 on identical exact-HLT inputs;
7. M1F versus M1J and `S0_21` on identical exact-HLT inputs;
8. every factorized and joint rung versus its immediate teacher.

U100 versus `D100direct` alone is descriptive because depth differs. U100
versus `S100_10` is the path-specific depth control. The same distinction
applies at exact HLT.

### 14.4 Paired statistics and multiplicity

The screening run is one seed and cannot establish a final claim. Compact
identity-ordered validation logits may be retained under a versioned manifest
for paired bootstrap comparisons; at 100k rows and 15 FP32 logits they are
small relative to particle data.

If screening motivates confirmation, a separate predeclared validation-only
confirmation registry uses seeds `(11, 22, 33, 44, 55)` for at least:

- paired M0 and TOFF context;
- U100 and S100_10;
- D0F, J100, and S0_20;
- M1F, M1J, and S0_21;
- P0KD when the adapter gap is material.

Confirmation cannot add an attractive unregistered intermediate after the
screen. A new untouched final-test source or an explicitly descriptive reused
holdout contract is required for any later test wave.

## 15. Campaign graph

The 300k campaign DAG is:

```text
authenticate parent spec/split/selection/recipe/assignments/checkpoints
  -> build train-only residual-pair scale calibration
  -> build staged train coupling shards (cost + mass)
  -> freeze train-only switch-CDF calibration
  -> finalize train coupling shards -----+
  -> build validation coupling shards ---+-> manifests + full-role audit
                                             -> coupling lock
                                             -> bounded V(s,f) cache miniature
                                             -> endpoint equality lock
                                             -> immutable graph/recipe lock

  -> P0CE || P0KD
  -> factorized U/D path (sequential)
  -> joint path          (sequential, parallel with factorized)
  -> D100 stationary     (sequential, parallel)
  -> HLT stationary      (sequential, parallel)
  -> paired direct/self controls (parallel when parents exist)
  -> validation aggregate
```

There is no finalist lock, execution lock, test assignment, or final-test
evaluation task in the initial supplemental DAG.

Dependencies express artifact availability only. A weak U010, a negative
rung, or a joint path worse than M0 still allows all registered descendants to
run. Invalid lineage, corrupt artifacts, nonfinite required values, source
drift, or a failed Slurm parent stops the corresponding `afterok` closure.

The factorized, joint, D100-stationary, and HLT-stationary chains start in
parallel after their shared prerequisites. Each remains sequential internally.
Assignment/coupling arrays are uncapped by default unless measured filesystem
or site policy records an operational cap.

## 16. Versioned artifact families

New scientific semantics receive new identities. Existing high-coverage
assignment v2, Shell Exact v1, dense graphs, and HCWDL campaign v7 are not
broadened.

The implementation introduces at least:

```text
HCWDL_RESIDUAL_SHELL_COUPLING_CONFIG/v1
HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1
HCWDL_RESIDUAL_SHELL_STAGED_SHARD/v1
HCWDL_RESIDUAL_SHELL_SWITCH_CALIBRATION/v1
HCWDL_RESIDUAL_SHELL_COUPLING_SHARD/v1
HCWDL_RESIDUAL_SHELL_COUPLING_MANIFEST/v1
HCWDL_RESIDUAL_SHELL_COUPLING_AUDIT/v1
HCWDL_RESIDUAL_SHELL_COUPLING_LOCK/v1

HCWDL_STRUCTURAL_FEATURE_COORDINATE/v1
HCWDL_STRUCTURAL_FEATURE_NODE_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_GRAPH/v1
HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_COMMAND_PLAN/v1
HCWDL_STRUCTURAL_FEATURE_TRAINING_REPORT/v1
HCWDL_STRUCTURAL_FEATURE_AGGREGATE/v1

HCWDL_STRUCTURAL_FEATURE_RECOVERY_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_RECOVERY_COMMAND_PLAN/v1
```

The generic `HCWDL_RECIPE/v4`, PMARD training/resume v6 behavior, checkpoint
selection v1, high-coverage assignment manifests v2, Shell Exact v1, generic
submission ledger v2, monitor v1, and task-attestation v1 may be reused only
through their existing validators and exact parent hashes.

Any change to P0 membership, endpoint projection, coupling groups or weights,
scale floors, solver objective, dummy semantics, edit mass, CDF binning,
carrier order, U/J coordinate table, graph, teacher edge, seed alias, or
stationary controls is a scientific contract change.

## 17. Proposed durable layout

```text
campaign_root/
  campaign_spec.json
  imported_parent.json
  coupling/
    config.json
    train_calibration.json
    train/*.npz
    validation/*.npz
    train_manifest.json
    validation_manifest.json
    full_role_audit.json
  locks/
    coupling_lock.json
    endpoint_equality_lock.json
    graph_recipe_lock.json
  controls/
    P0CE/
    P0KD/
    direct/{D100direct,D0direct,M0self}/
    stationary_d100/S100_*/
    stationary_hlt/S0_*/
  training/
    factorized/{U010,...,U100,D90F,...,D0F,M1F}/
    joint/{J005,...,J100,M1J}/
  reports/
    validation_aggregate.json
    campaign_complete.json
  recovery/
  submission_ledger.json
```

Every node keeps immutable spec/report/selection files, selected and pass-60
model-only checkpoints, at most one rolling exact-resume checkpoint, compact
loss/validation history, and logs. Repaired U/J arrays and epoch copies are
forbidden durable artifacts.

## 18. Implementation map

The preferred reusable boundaries are:

| Path | Responsibility |
|---|---|
| `src/hlt_classification/scouting/repair.py` | expose the existing all-21-field offline projection and Shell Exact per-slot state without changing v1 semantics |
| `src/hlt_classification/scouting/hcwdl_upper_coupling.py` | A/B/C/O/R construction, cost groups, Hungarian coupling, edit mass, train CDF, carrier records |
| `src/hlt_classification/scouting/hcwdl_upper_cache.py` | coupling shards, manifests, full-role audit, ROOT recomputation, lock inputs |
| `src/hlt_classification/scouting/hcwdl_homotopy.py` | generalized `V(s,f)` raw endpoint builder, carrier compaction, endpoint equality |
| `src/hlt_classification/scouting/hcwdl_homotopy_stream.py` | bounded ROOT/assignment/coupling streaming into canonical ParticleInputs |
| `src/hlt_classification/scouting/hcwdl_homotopy_graph.py` | exact factorized, joint, stationary, adapter, and direct-control registries |
| `src/hlt_classification/scouting/hcwdl_homotopy_contracts.py` | new contract constants, payload builders, and fail-closed validators |
| `src/hlt_classification/scouting/hcwdl_homotopy_runner.py` | one node: views once, teacher logits once, generic training reuse |
| `src/hlt_classification/scouting/hcwdl_homotopy_campaign.py` | 300k DAG, resources, command plan, authorization, task identities |
| `src/hlt_classification/scouting/hcwdl_homotopy_workflow.py` | thin task dispatch and exact artifact consumption |
| `src/hlt_classification/scouting/hcwdl_homotopy_reporting.py` | rung, path, recovery, stationary-control, and transition diagnostics |
| `src/hlt_classification/scouting/hcwdl_homotopy_recovery.py` | source-pinned failed-closure recovery |

Reuse rather than duplicate:

- `DenseAssignmentStore` for imported fixed-skeleton assignments;
- `EphemeralPmardViewCache` replay/memory principles, extended through a new
  lineage-aware homotopy wrapper if necessary;
- `precompute_teacher_targets()` and identity-keyed FP32 targets;
- `train_hcwdl_node()` with an explicitly supplied custom registry/domain
  table;
- existing metric, checkpoint-selection, Slurm journal, monitor, exact-ID
  cancellation, and task-attestation code.

`iterate_pmard_batches()` remains the fixed-HLT-skeleton stream. It is not
silently overloaded to construct variable-support U views. The new homotopy
stream has a distinct contract and calls shared projection/repair primitives.

Thin command surfaces should include:

```text
scripts/build_hcwdl_upper_calibration.py
scripts/build_hcwdl_upper_coupling_shard.py
scripts/finalize_hcwdl_upper_coupling.py
scripts/create_hcwdl_homotopy_pilot.py
scripts/run_hcwdl_homotopy_task.py
scripts/submit_hcwdl_homotopy_pilot.py
scripts/monitor_hcwdl_homotopy.py
scripts/resume_hcwdl_homotopy.py
scripts/cancel_hcwdl_homotopy.py
sbatch/run_hcwdl_homotopy_task.sh
```

Mechanical naming changes are acceptable only if the contracts, tests,
runbook, and handoff are updated together.

## 19. Slurm, RAM, storage, and source pinning

Every campaign uses a dedicated clean detached worktree at the exact pushed
commit. The immutable spec stores its absolute `PROJECT_DIR`; a shared mutable
main checkout is forbidden for workers.

Every worker:

- uses account `reu-aisocial` and partition `tigris`;
- activates `atlas_kd_tigris` with `PYTHONNOUSERSITE=1`;
- prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`;
- receives absolute project/spec paths;
- ends the batch shell with `exec python -s ...`;
- uses `--signal=B:USR1@120` for checkpointable GPU training;
- writes only below the declared campaign root or bounded temporary space.

U/J cardinality differs from fixed-skeleton D jobs, so the implementation must
run a new genuine Tigris cache/training miniature and measure RAM, GPU memory,
I/O, walltime, and storage. It cannot inherit 192/320/384-GiB or walltime
values without evidence. Requests keep explicit headroom.

The compact coupling is expected to be small. Process RAM holds at most one
student domain for train and validation plus temporary teacher views and a
small FP32 logit table. Durable storage is limited to coupling arrays,
checkpoints, reports, locks, logs, and optional compact validation/teacher
logits. No ROOT, NPZ, Parquet, or tensor file containing reconstructed U/J
particle views may be published.

## 20. Failure and recovery

Poor finite science completes. Execution failures include invalid endpoint
records, nonphysical p4, coupling conservation failure, active count above
200, hidden truncation, source/parent drift, corrupt hashes, nonfinite required
values, wrong teacher domain, checkpoint mismatch, forbidden test access, and
non-exact resume.

Source-pinned failed-closure recovery must exist before the real pilot launch.
It follows the established dense-recovery model:

1. bind the complete original ledger and immutable monitor report;
2. identify the exact failed task and downstream dependency closure;
3. reuse valid completed prefix artifacts in place;
4. resume a compatible interrupted node from its rolling checkpoint;
5. run corrected execution source from a new clean commit/worktree;
6. forbid changes to graph, recipe, coupling, coordinates, seeds, resources,
   output root, or scientific parents;
7. publish a separate recovery spec, command plan, attestations, and ledger;
8. support a later repeated recovery without discarding earlier completions.

Old pending jobs are cancelled only by exact IDs read from their campaign-
bound ledger. Broad name cancellation, `scontrol release` repair, deleting a
rolling checkpoint, or ordinary requeue under known-broken pinned source is
forbidden.

## 21. Required test matrix

### 21.1 Endpoint partition and coupling

- exact P0 90/60 membership, native indices, lost-track handling, and neutral
  offsets;
- unclassified HLT dustbin targets remain valid, exact HLT, and never acquire
  an offline correspondence claim;
- assigned D100 endpoints inside and outside P0 visibility;
- exact A/B/C/O/R partition and `R_hlt`/`R_off` typing;
- empty, O-only, R-only, equal, and rectangular residual sets;
- exact substitution/removal/insertion counts and unique endpoint use;
- hand-calculated cost groups, train quantiles, floors, weights, quantization,
  Hungarian optimum, and lexicographic ties;
- row/input/worker permutations leave the logical coupling unchanged;
- no label or final-test branch dependency;
- edit mass, train CDF, validation transform, hash tie, uint16 round trip, and
  nested thresholds;
- shard corruption, cross-role/source/selection/assignment/config rejection;
- 99,999-of-100,000 perfect observed rows fails completeness;
- sampled ROOT recomputation reproduces logical arrays.

### 21.2 Carrier and generalized views

- carrier capacity equals `max(|A|,|B|)` and never exceeds 200;
- U000 active multiset and all 21 projected fields equal P0;
- U100 features, vectors, mask, padding, lengths, and order are byte-identical
  to current D100;
- `V(1,f)` is byte-identical to existing Shell Exact at `alpha=1-f` for every
  registered f;
- D0F and J100 are byte-identical to canonical HLT;
- atomic O-to-R, O-to-padding, and padding-to-R edits;
- no fractional discrete values, hybrid residual records, duplicate particles,
  nonphysical p4, nonfinite active fields, or nonzero padding;
- relative fields and axes are not recomputed from the changing set;
- structural and discrete switches are nested and invariant to batch, chunk,
  worker, epoch, cache, and resume;
- stream and RAM replay emit identical identities, batches, tensors, labels,
  and tails.

### 21.3 Graph and training

- exact 21-node factorized and 21-node joint registries;
- exact 10 U, 10 D, 20 J coordinate tables with decimal and hexadecimal
  floats;
- U010 and J005 teacher is TOFF-native;
- U100 teaches D90F; D10F teaches D0F; D0F/J100 teach separate M1 nodes;
- every student is fresh and no cross-path teacher is allowed;
- exact D100 and HLT stationary-depth chains and direct controls;
- paired transition-index seeds and endpoint seed aliases;
- identical rows, 60 passes, 60 validations, update counts, and final partial
  batch across paired nodes;
- unweighted CE/KD, role-routed temperature, BF16 forward, and FP32 loss math;
- teacher evaluated on its own domain exactly once and student view built once;
- macro-AUC/CE/logR/earliest checkpoint tie order;
- uninterrupted and GPU-mapped interrupted/resumed equivalence;
- USR1 reaches Python and publishes compatible state;
- deliberately poor finite models complete every registered descendant.

### 21.4 Campaign and operational contracts

- every imported and new parent hash is validated before reuse;
- source drift, dirty worktree, mutable project path, and changed command bytes
  fail closed;
- dry run is nonmutating and renders exact tasks, dependencies, resources,
  coordinates, and paths;
- factorized/joint/stationary chains are parallel across chains and sequential
  within each chain;
- all smoke commands are bounded and no smoke result authorizes full coverage;
- no initial task can read final test;
- exact-ID monitor, cancellation, first recovery, and repeated recovery;
- task attestations and all-reused restart validation;
- no durable repaired-particle artifact and no matcher call after assignment
  authorization.

### 21.5 Acceptance ladder

Local acceptance requires focused tests, the complete repository suite,
Python compilation, CLI help/static checks, Markdown-link validation, and
`git diff --check`. Runtime acceptance additionally requires:

- a bounded local synthetic end-to-end graph;
- installed-Weaver FP32 parity for the unified and native teacher factories;
- a genuine Tigris production-worker miniature exercising coupling, U, J,
  stationary, cache, target, training, preemption, resume, and aggregate paths;
- measured resource and storage reports;
- a complete nonmutating 300k dry run from exact pushed source;
- explicit user authorization for live 300k submission.

## 22. Implementation blocks

1. **Contract and projection extraction:** create new contract constants,
   expose the existing 21-field endpoint projector, and prove no change to
   Shell Exact v1.
2. **Endpoint partition and coupling:** implement A/B/C/O/R, train-only scale
   calibration, cost, global solver, edit mass/CDF, compact shards, manifests,
   audit, and lock.
3. **Generalized view builder:** implement carrier layout and `V(s,f)` with
   exact P0/D100/HLT endpoint tests and shared Shell Exact semantics.
4. **Graph and training reuse:** register factorized, joint, adapter, direct,
   and stationary nodes; reuse one-time RAM views/targets, generic training,
   AUC selection, and exact resume.
5. **Campaign and recovery:** implement the validation-only 300k DAG, thin
   CLIs/worker, command plan, source pinning, monitoring, exact cancellation,
   and failed-closure recovery.
6. **Reporting and local closure:** implement transition/recovery/stationary
   reports, synthetic smoke, full tests, traceability, and handoff evidence.
7. **Real acceptance:** run installed-Weaver parity, Tigris miniature, measure
   resources, render the exact 300k dry run, and request separate launch
   authorization.

Each block receives focused tests before the next block consumes its output.
No live Slurm job, remote push, or final-test access is authorized merely by
this document.

## 23. Definition of done

Implementation is complete only when:

- no donor or exploratory checkout is a runtime dependency;
- P0, D100, and HLT endpoint meanings are exact and independently tested;
- coupling is generated once, label-free, compact, authenticated, and fully
  conserved for all selected train/validation rows;
- U100 is byte-identical to current D100 and D0F/J100 are byte-identical HLT;
- all primary and stationary graphs are immutable and executable end to end;
- every node uses the locked unweighted recipe, 60 passes, every-pass
  validation, macro-AUC-first selection, one-time views/targets, and exact
  resume;
- the initial DAG has no final-test access path;
- source-pinned repeated failed-closure recovery is available before launch;
- the genuine Tigris miniature passes and resource requests are measured;
- a complete 300k dry run emits exact commands without mutation;
- `docs/HANDOFF.md` records evidence and the remaining authorization boundary.

Files existing without those behaviors do not satisfy this plan.

## 24. Claims and limitations

Allowed claims include:

- projected-native-offline carrier endpoint;
- residual-shell structural coupling;
- minimum-disruption edit path;
- exact D100 and HLT endpoints;
- factorized versus joint knowledge-transfer path;
- HLT-only deployment for M1F/M1J.

Forbidden claims include:

- residual O-to-R truth matching or physical correspondence;
- U000 being tensor- or architecture-identical to TOFF-native;
- U100 being native offline;
- a switched HLT dustbin being reconstructed from its coupled O particle;
- equal GPU compute merely because edge/update counts match;
- success proving that unavailable offline particles are inferable at HLT;
- failure proving an information-theoretic ceiling.

The remaining gaps may arise from the native-to-unified adapter, TOFF capacity,
90/60 versus 200 visibility, wrong completion-shell assignments, omitted or
inserted particles, the operational coupling metric, field/support path,
optimization, KD temperature/weight, model capacity, or genuinely unavailable
HLT information.

A one-seed validation result is exploratory. Stationary-depth controls are
required before attributing a gain specifically to gradual view change.
Confirmation requires the registered paired seeds, and any future test claim
requires a separately authorized, scientifically valid holdout policy.

## 25. Authorization boundary

This plan authorizes local implementation and verification when the user asks
for it. It does not authorize:

- changing completed HCWDL/dense artifacts;
- rerunning the high-coverage matcher;
- live Tigris submission;
- final-test branch access;
- remote push or cancellation of existing jobs;
- replacing the fixed coordinate or controls after validation inspection.

A live 300k HCWDL-UJ screen requires completed implementation, exact clean
pushed source, authenticated parent and coupling locks, genuine Tigris
miniature, measured resources, complete dry run, source-pinned recovery
surface, and an explicit user launch instruction.
