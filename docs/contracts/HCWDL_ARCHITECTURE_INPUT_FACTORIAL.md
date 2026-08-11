# HCWDL Architecture–Input Factorial Contract

Version 1 freezes the validation-only common-schema 2x2 study in the
[implementation plan](../plans/HCWDL_ARCHITECTURE_INPUT_FACTORIAL_PLAN.md).

The versioned artifact family is:

- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_SPEC/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_GRAPH/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_NODE/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_CHECK/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_TRAINING_REPORT/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_AGGREGATE/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMPLETION/v1`;
- `HCWDL_ARCHITECTURE_INPUT_FACTORIAL_COMMAND_PLAN/v1`.

The graph contains exactly `H_U`, `H_S`, `O_U`, and `O_S`; native `TOFF` is
reference-only. Semantic changes to partition policy, stream depth, fusion,
P0 projection, cells, loss, pairing, selection, or effects require a new
version. Operational source recovery may not alter those semantics.

Every artifact is content-hashed and parent-bound. A clean source-pinned
worktree, exact canonical campaign spec, explicit creation and submission
phrases, installed architecture check, immutable parent lineage, zero
final-test rows, and exact Slurm job IDs are required. Persistent repaired P0
or split datasets are forbidden. The installed architecture check must run
both architectures through finite forward and backward passes under the exact
BF16 autocast policy used by training; an FP32-only check is insufficient.
