# Literature ladder conditional science submission, v1

## Authority and unchanged science

Execution amendment authorized by the user on 2026-10-03: automatically queue
the full literature V3 200k train / 50k validation DIRECT + COARSE campaign
when its existing preflight succeeds. This replaces the need for a second
interactive science approval for the bound campaign only. It does not waive
the measured gate, exact-source check, full dry plan or site/resource checks
in [the scientific plan](../plans/JETCLASS2_LITERATURE_V3_200K_DIRECT_COARSE_PLAN.md).
No matching, data, model, KD, seeds, stopping rule or scientific schema changes.

The operational controller runs from a separate checkout and invokes the
original gate worktree's Python/CLI in fresh subprocesses. Never update the
active gate worktree or run its scientific artifacts under the new checkout.
Initial intended binding: source `99a1e231e38fe5a07ce5a31a4f4a74bb9ec40539`,
preflight `21796580`, gate hash
`d61180d02f69951bb4d73959f19b8764a78cafec5f680a1c31dd2285ff4d3b9e`.
Other bindings require explicit caller-supplied full source/hash/job values.

## State machine

1. Authenticate the saved GATE_SPEC/v7 under the original clean pushed source.
   Recompute the gate plan and validate its complete live ledger and per-job
   journal. Require the supplied exact preflight ID and the SPORC cluster.
2. Without `--execute`, print the binding/policy and exit read-only. With it,
   obtain an exclusive Linux file lock under `STUDY/followup/`, and publish
   immutable authorization including gate/ledger parents, source, job IDs,
   controller-code SHA-256 and the bounded science policy. Print `ARMED`.
3. Poll accounting for only the three authenticated gate IDs every 60 seconds,
   for at most 14 days. Verify owner, account and full job names. Require all
   three to be COMPLETED with exit 0:0. Missing records are not success;
   failure/cancellation/unknown terminal states stop. Accounting errors stop
   rather than treating an unavailable scheduler as a successful gate.
4. Reauthenticate the identical gate/ledger/source. Through the pinned CLI,
   create or validate its sibling `science/campaign_spec.json`. This requires
   authentic gate_complete, installed-Weaver parity, real full-population
   measurements and a profile fitting debug's 24h ceiling. Slurm completion
   alone is insufficient. Existing incompatible/partial roots are preserved.
5. Materialize the complete canonical dry plan and ledger through that CLI.
   Enforce nine fits, five reducers and aggregate/completion: sixteen jobs;
   DIRECT/COARSE only; SPORC debug, reu-aisocial, qos_tier3. Save review evidence
   binding the actual plan hash to the advance authorization. There is no
   second prompt: the caller approved this conditional, measured envelope.
6. Invoke the original live submit with that exact plan hash and phrase.
   Its read-only site feasibility checks, exact dependency DAG, sanitized
   environment, durable exclusive submit claim and journals still apply.
   Authenticate the resulting ledger and save `followup/complete.json`.
   This means submitted, NOT trained. No automatic retries of science jobs.

`JC2_LITERATURE_LADDER_FOLLOWUP/v1` has kinds authorization, review, submitted;
each carries schema_version=1, canonical content_hash, parent hashes and
final_test_accessed=false. Publication is atomic and refuses overwrites.
Scientific artifacts retain their existing contracts and original source pin.

## Disconnects, duplicate controllers and failures

`scripts/start_jetclass2_literature_ladder_followup.sh` activates the SPORC
environment and starts the thin CLI with nohup, unbuffered output, detached
stdin and a unique log under the study. Closing SSH or stopping `tail` does
not deliberately terminate it; host failure/administrative process cleanup
can still stop it. It is a polling process, not a scheduler job, so it is not
itself listed by squeue. Initial source validation can take several minutes.
Do not call it armed until the log says ARMED.

Only one process holds the kernel lock. Restarting after a controller crash
can reuse identical metadata and a completed science ledger, but never
automatically bypasses a science submit claim without a completed ledger.
Any ambiguous partial live submission requires manual exact-ID inspection.
No cancellation, partition changes, artifact deletion, or final-test access.
The controller clears inherited SBATCH/SLURM settings for its subprocesses;
the launcher sets PYTHONNOUSERSITE and the Conda library path.

Launcher arguments (all on one command):

```text
start_jetclass2_literature_ladder_followup.sh GATE_SPEC GATE_HASH SCIENCE_COMMIT PREFLIGHT_JOB
```

The launcher explicitly arms live conditional submission. For a read-only
review, run `jetclass2_literature_ladder_followup.py --gate-spec ... --gate-hash
... --source-commit ... --preflight-job ...` without `--execute` instead.
