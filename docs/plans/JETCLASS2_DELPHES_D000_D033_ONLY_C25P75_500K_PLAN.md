# JetClass2 D000 from D033-only C25/P75 endpoint ablation

## Question

The completed `TRAIN_500K` salience MT20 coarse ladder retains substantial
validation recovery at D033 but loses it at exact-HLT D000. The MT20 endpoint
is not a single-parent test: its objective is 20% CE, 50% immediate D033 KD,
and 30% KD from older, more privileged ancestors. This isolated fit tests
whether that historical-teacher objective causes the endpoint collapse.

## Matched intervention

Train one fresh deployable D000 Particle Transformer with:

- the exact selected `JC2SMT20_COARSE_D033_from_D066` T=2 train probability
  bank as its only teacher;
- 25% ground-truth cross entropy and 75% forward-KL logit KD at T=2;
- the exact `TRAIN_500K` membership and 1,000,000-jet validation role;
- the selected `SALIENCE_PT_LINEAR` persistent-HLT foundation;
- the exact D000 initialization and sampler seeds, architecture, batch size,
  optimizer, learning-rate schedule, checkpoint selection, and exact-HLT view
  used by the MT20 D000 endpoint.

No D066, U100, U050, or U000 probability enters the new loss. There is no
warm start, model ensemble, representation loss, final-test access, rolling
resume, or modification of the source campaign.

## Authentication and execution

Creation authenticates the complete source MT20 campaign contract and requires
durably completed, checksummed M0HLT, U000, coarse D033, coarse MT20 D000, and
the D033 reducer bank. The new root is disjoint from source data, matching,
campaign, and source worktrees.

The study is one independent SPORC A100 job. It may be routed at creation to
`debug` or `tier3`; routing changes no scientific setting. The worker first
performs a real C25/P75 one-batch finite backward check using the authenticated
D033 bank, discards that model, restores the paired D000 RNG seed, and starts
the full fit from scratch. Particle views and teacher probabilities remain in
RAM. Durable outputs are only the selected checkpoint and compact JSON
evidence/results.

## Interpretation

The validation report prints M0HLT, persistent-HLT U000, the D033 teacher, the
existing MT20 D000 endpoint, and the new D033-only endpoint using the same
recovery convention. A substantial improvement over MT20 D000 supports the
hypothesis that historical privileged teachers and/or C20/P80 caused the
collapse. Another collapse localizes the problem to the D033-to-D000 endpoint,
new data/views, or their optimization rather than the MT20 ancestry mixture.

Poor performance remains a valid result and must not fail completion.
