# Luka all-eligible regular DIRECT / COARSE development experiment

User authorization: 2026-10-14. For this NEW campaign only, supersede the
100k/50k population and no-former-test-reading rules of LUKA_FULLSIM_REGULAR_PLAN.
Never mutate historical manifests, jobs, caches or worktrees. The earlier
149553-row holdout was already evaluated; it is explicitly retired for this
exploratory experiment. No independent final-test result is claimed.

Reauthenticate the same 890 ROOT files and original scalar inventory
0832a577f7bcd0b9a9b719a7f62ef2a8e1b5b24108beec5bd7b3946a09642d17.
Replay the original 100000/50000 split
79ac6e577b92a5d2ed00d848c996812b388cab57f9d97d6b519f2bc10060c3b2.
Keep every original validation FILE in validation, now including all its
eligible entries (74895). All eligible entries in original train and former
test FILES become train (149681 + 149553 = 299234). There are zero test and
zero unused eligible rows. Do not reshuffle files, rebalance classes, relax
cuts, sample with replacement or leak former training rows into validation.
Retain natural per-role class proportions; whole-file ratios need not equal
the exact old sampled quotas. File disjointness is established; cross-file
event independence remains unverified without event identifiers.

Freeze the same nine fresh single-encoder fits and five TRAIN-only reducers:
M0HLT, OFFLINE, U000, DIRECT_D000 from U000, and U000 -> COARSE_U050 ->
COARSE_U100 -> COARSE_D066 -> COARSE_D033 -> COARSE_D000. Aggregate/complete
make 16 science jobs. Identical endpoint views, pairing, feature kernels,
provisional mm, error-only zero masking, support 512/no truncation, coordinate
seeds, batch 256, BF16, optimizer, 60--100-pass schedule, early stopping and
25% CE/75% T=2 KL are inherited unchanged. No fusion or imported weights.

All 74895 validation rows select checkpoints. After selection, record metrics
on both the full validation set and the fixed ORIGINAL 50000-row subset.
The subset never chooses a checkpoint or recipe. Save ordered validation
probabilities/labels/identities for auditable subsequent diagnostics. Both
sets are development results, not blind tests; comparisons remain single-seed
and direct/ladder have unequal training compute. Poor metrics never gate jobs.

Use a separate ALLJETS/v1 artifact family and source-pinned namespace. Legacy
split/particle readers continue rejecting this new role assignment. Reuse the
bounded eight-process durable-input implementation via explicit family dispatch;
do not rebuild ROOT views during GPU science. Reauthenticate scalar particle
counts to bound disk use instead of assuming every row has 512 particles.
Keep a storage reserve and fail without deletion if insufficient space.

Three gate jobs: new foundation, durable preparation, genuine SPORC A100
preflight. Same 6 CPUs/90000 MiB/8h GPU envelope, separate TRAIN backward and
validation inference-only stress for seven coordinates. Load only the current
coordinate's TRAIN/validation caches during this larger preflight. Installed
Weaver parity, full-population technical passes, bank/checkpoint round trips,
resource headroom and measured debug <=24h requests remain mandatory. Technical
metrics never select methods. Real SPORC acceptance, not local CPU fixtures,
admits science. Old smaller-data evidence cannot substitute for this gate.

Dry-review and exact-hash authorization are separate for gate and science.
No automatic follow-up, retries, cancellation, commit, push or live submission
is authorized by implementation alone. Preserve failed/ambiguous work.
