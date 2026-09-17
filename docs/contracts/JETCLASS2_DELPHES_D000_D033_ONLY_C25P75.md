# JetClass2 D000 from D033-only C25/P75 contract

`JETCLASS2_DELPHES_D000_D033_ONLY_*/v1` artifacts identify one isolated
`TRAIN_500K` endpoint ablation. The only teacher is the exact selected
`JC2SMT20_COARSE_D033_from_D066` T=2 train probability bank from the bound
salience MT20 campaign. The student is a fresh exact-HLT D000 model trained
with 1/4 CE and 3/4 forward-KL KD at T=2.

The source campaign, foundation, split memberships, reduced 17-input/11-output
model, D000 view, initialization seed, sampler seed, optimizer, batch size,
schedule, and selection rule are immutable parents. Historical teachers,
cross-spine teachers, warm starts, ensembles, representation targets, rolling
state, and final-test access are forbidden.

Source task reports, training reports, bank manifest, and all probability
shards must validate by content hash and exact ordered train identities before
training. Particle views and loaded probabilities are RAM-only. The worker
must pass a real finite C25/P75 backward check, discard that model, reseed, and
then train from scratch. Only selected weights and compact reports persist.

Execution is one source-pinned SPORC A100 Slurm job with an immutable dry-run
ledger and exact live-submission receipt. The job may use `debug` or `tier3`
as recorded at creation. It never mutates, cancels, or joins the source MT20
DAG. Poor validation performance is scientific output, not workflow failure.
