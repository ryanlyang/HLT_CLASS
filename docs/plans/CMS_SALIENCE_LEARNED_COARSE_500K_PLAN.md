# Native CMS Strategy-B coarse replacement

Authorized 2026-09-19 as a time-saving replacement for the pending dense
ladder. This plan supersedes only the graph of the native CMS dense plan;
its data, matching, model, optimization, validation and sealed-test rules
remain unchanged. Do not edit an existing dense spec or source worktree.

## Registered science

`U000 -> U050 -> U100 -> D066 -> D033 -> D000`

U050 is exactly one half of the upper transition. D066 and D033 retain
exactly two thirds and one third of offline content, respectively (not
rounded 0.66/0.33). Each arrow owns cold acquisition, acquisition probability
reduction, warm withdrawal with fresh optimizer, exact single-view extraction,
and a carrier reducer when another arrow follows. Keep the original three
CE references and the direct U000-to-D000 KD control. This is 14 fits,
5 extractions, 10 reducers and aggregate/completion: 31 logical science tasks.
Use the same per-coordinate seed aliases for shared nodes; do not silently
reseed reference models. There is no guarantee of matching dense performance.

## Versioning and read-only reuse

Native CAMPAIGN_SPEC/v4 explicitly registers `ladder=coarse` and GRAPH/v2;
v1-v3 specs and GRAPH/v1 remain dense. The historical family namespace and
seed domain are retained deliberately; version and graph identity distinguish
the experiments. Execution acceptance remains v2 with 85% CPU / 90% CUDA
limits, batch 256 and the 30 repeated worst-length probes. Require a fresh
source-bound gate for the new source; the measured 7bb17138 gate is supporting
evidence, not an acceptance artifact to relabel.

Preparation may be imported from the original completed native producer.
PREPARATION_IMPORT/v2 permits only the reviewed addition of U050/D066/D033:
all preparation file/tree Git objects must match and the coordinate ASTs must
belong to the exact reviewed pair. Dense coordinate meanings, view config,
matching/coupling payloads, identities, split and budgets must match. Existing
v1 imports retain their original strict rules.

An optional SHARED_SOURCE/v1 binds the exact accepted dense campaign and its
authenticated science ledger. Reuse only train_M0HLT, train_OFFLINE,
train_U000, reduce_U000 and train_DIRECT_D000. Their node identities, recipe,
data/preparation identity and model/training source must agree. Keep these
source jobs untouched, including pending/running jobs. New import tasks wait
on exact source job IDs where necessary and validate completed receipts and
reports before copying compact checkpoints/probability banks into the new
root. Imported reports name their original source hashes explicitly. Dense
fusion carriers never substitute for coarse carriers.

Do not attach dependencies to purged completed jobs: authenticated completed
outputs need no scheduler dependency. Resolve source dependencies at actual
submission time and journal the exact resulting commands. Existing journaled
jobs are never resubmitted; unresolved/failed sources fail closed.

## Safe operations

Create a fresh source-pinned coarse root and full dry plan before retirement.
No command cancels as a side effect of creation, preparation, gating or
submission. An explicit retirement operation targets only the dense ledger's
41 nonshared tasks, never job-name globs or another campaign. Default is a
read-only preview. Executing retirement requires an accepted replacement gate
and explicit authorization; preserve shared jobs and all source artifacts.
The science stage queues the full 31-task coarse DAG (five imports and ten
fresh fits when all shared work is reused). Final test stays sealed.

Tests must cover unchanged legacy dense graphs, exact coarse coordinates,
preparation byte parity, incompatible/tampered source rejection, partial and
completed shared work, stale scheduler dependencies, crash-safe submission,
exact cancellation boundaries, and the production import/train/withdraw/
extract chain. Local tests do not replace installed-Weaver/A100 acceptance.

Internal implementation donor is `7bb171382b7206013bc5d9308a4c22b2929bc7f4`;
original preparation producer is `f2e8a374f522a39c7f3a6331f0ec77ae12cabaea`.
`scripts/switch_cms_salience_coarse.sh prepare DENSE_SPEC NEW_ROOT` creates
the campaign, verifies the read-only preparation import, and queues only the
fresh gate. `preview COARSE_SPEC` reports exact retirement targets without
mutation. `finish COARSE_SPEC` checks the new gate and shared-source health,
cancels the dense-only jobs, and submits the complete coarse science graph.
Run with bash, not by sourcing: failures must not terminate the SSH shell.
The new scientific source must be committed, pushed and pinned first. Never
stage unrelated dirty matching/scouting/model changes into this migration;
the conservative Git-object compatibility checks intentionally reject them.

Local verification (2026-09-19): 88 passed, 5 installed-Weaver-dependent skips
across native, coarse and temporary-memory regression tests. Shell syntax and
scoped whitespace checks pass. The miniature spans all five coarse transitions
through final single-view D000 extraction; it is not GPU acceptance evidence.
The staged helper reruns focused tests on SPORC before creation and requires a
fresh genuine gate before retirement/science. No remote action was performed
by the implementation agent.
