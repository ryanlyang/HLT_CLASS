# LUKA_ALTERNATING_GATE / v1

Authority: [active plan](../plans/LUKA_ALTERNATING_GATE_PLAN.md).

All package artifacts use `LUKA_ALTERNATING_GATE_<KIND>/v1`, schema1,
canonical content hashes, parent hashes and inherited ALLJETS evaluation flags.
No old warm/cold gate artifact is interchangeable with these artifacts.

Kinds: RECIPE, ANCHOR, GATE, SPEC, PLAN, SUBMISSION_CLAIM, BANK,
TRAINING_REPORT, SELECTED_OUTPUTS, TASK, PREFLIGHT, RESULTS, OWNER,
RETIREMENT_INTENT, RETIREMENT. Immutable publication and file SHA256/size checks
are required before reuse; path existence does not authenticate an artifact.

Nodes bind arm, stage, coordinate, teacher identity/coordinate, context coordinate,
gate mode/weight, export-factors flag, initialization/sampler seeds, fresh state
and deployability. Exact 18-fit registration is replayed against the original
fixed30 parent. Each arm is an independent six-node chain; no cross-arm teachers.

BANK binds the complete registered teacher node, checkpoint/report SHA256,
temperature 2, TRAIN ordered identities/labels/counts/prepared hash and exact
factor coverage. Factor export is permitted only for GATE_TEACH_D066/D033/D000.
Ragged 9-channel FP32 factors reconstruct the bounded correction on demand;
read the linear-size bank once into RAM, never repeatedly reopen shuffled shards.
No external context is accepted by SELF models, even at mixed-view coordinates.

PREFLIGHT binds successful exact submitted job accounting, installed environment,
GPU, all 18 one-pass receipts, 36 role-specific stress records, FP32/BF16 parity,
initial-state digests, population alignment, measured fit walltimes and storage.
Science SPEC binds that evidence; dry-review/explicit plan-hash authorization
is required independently for gate and science. All allocations are debug/A100,
6 CPUs/128000 MiB; gate 8h, fit <=24h, aggregate 1 CPU/16384 MiB/1h.

Selected outputs retain one checkpoint, every epoch metric, validation
probabilities/identities/labels/counts/original50k membership and the teacher
manifest. Target retirement authenticates the entire exact inventory and durable
consumer proof before unlink; reject symlinks, hardlinks, escape paths and unknown
files. Deletion records remain valid after node-local scratch disappears.
Do not remove any anchor, shared input, prior campaign or Fresh_check artifact.

Results authenticate retained outputs without live accounting or retired target
payloads. Every registered row is reported, including pending and poor results.
Full validation selects; original50k only reports. Flags explicitly state the
historical test was consumed and repurposed; no independent test is claimed.
