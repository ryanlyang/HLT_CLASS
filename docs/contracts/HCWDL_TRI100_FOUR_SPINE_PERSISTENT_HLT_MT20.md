# HCWDL TRI100 persistent-HLT MT20 contracts

The `HCWDL_TRI100_FOUR_SPINE_FULLCARD_PERSISTENT_HLT_MT20_* /v1`
contract family defines an isolated four-spine experiment over the
full-cardinality bottleneck foundation and persistent-HLT support policy.

Every non-anchor fit uses exactly 20% CE and 80% temperature-2 probability
KD.  Its teacher set is every earlier node on the same spine, nearest first.
With two or more teachers the immediate predecessor receives 50 percentage
points of the total loss and the remaining 30 points are distributed over
older ancestors using a nearest-first geometric ratio of one half.  A node
with one teacher assigns all 80 KD points to that teacher.  Exact rational
weights, teacher order, probability-bank locks, manifests, identities, and
selected-checkpoint lineage are recorded in the graph and per-fit registry.

Each selected model publishes one reusable train/validation probability bank.
The weighted target for a fit is accumulated once in float64 RAM, normalized,
converted to float32, and never published as an array.  Only the small
content-hashed mixture registry is durable. Particle views, hidden states,
optimizer state, and rolling-resume state are not durable campaign artifacts.

The campaign retrains its own persistent-HLT U000 anchor, has no scheduler
dependency on or write path into existing campaigns, never uses cross-spine
teachers, never performs weight continuation or ensembling, and cannot access
final test.  Missing immediate-parent control reports remain pending in the
aggregate and do not control completion.  Poor metrics are scientific output,
not an operational failure.

Production execution is single-node, single-task, single-GH200.  The in-DAG
preflight must execute the installed production model on a real persistent-HLT
view, complete a backward pass, and exercise a two-or-more-teacher RAM mixture
without publishing it.  Recovery is exact-ledger and restart-from-zero only.
