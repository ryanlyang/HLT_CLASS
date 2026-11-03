# U100 particle/pair corrections on Brown CS Hydra

This additive execution port implements the user's request to move the four
fits in [the scientific plan](LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md) to Brown
CS Hydra. It does not change that experiment's losses, models, data, seeds,
30-epoch schedule, selection, or HLT-only inference. No new reference fit or
teacher fit is authorized. Existing SPORC and ALBERT jobs remain untouched.

The user verified the 2,287-file, 2.913 GiB export on Brown. Archive SHA256:
`2fbb97036df805967c651b9a15a4a14f89ef3a9a75fab1f9e96f1819ecc986f5`.
Manifest: `b87e76450441c78892af092a1cb67d6abeb23963b4c983eaf756cd4c4535116c`.
These are the only admitted inputs. Original manifests remain byte-identical;
old absolute paths are provenance, never runtime paths on Brown. Missing files
fail without raw-data reconstruction or new target inference.

Hydra uses `gpus`, account `default`, job QoS `normal`, one `nvidia_l40` GPU,
six CPUs and 120000 MiB per job. This is a site change from SPORC/debug, not an
assumption that Hydra has a debug partition. Two fit lanes preserve C/A/D/B
ordering, with `afterany` ordering dependencies. One shared setup precedes the
four fits and report. No separate preflight job. Setup measures all four real
production miniatures and support extremes before releasing fits, as before.

Create a separate Python 3.10.21 environment with the recorded eight packages,
PyTorch 2.5.1+cu118, cuDNN90100, and exact installed Weaver Python-source hashes.
Do not modify system Python or unrelated environments. Environment equality is
required, including on workers. The new execution namespace owns site/source
validation; original scientific modules are imported unchanged.

Additionally reload the transferred native control checkpoint and evaluate the
same 74,895 validation rows in batch128 using the original BF16 prediction
routine. Require exact row/label order, finite normalized probabilities,
maximum absolute probability deviation <=0.005, RMS deviation <=0.0002 and
absolute macro-AUC difference <=0.00005 against the saved SPORC predictions.
These fixed, symmetric numerical portability tolerances are not scientific
quality thresholds or evidence of bitwise equality. Record deviations before
checking them. Failure requires diagnosis; never silently widen tolerances or
rebase the control. Native geometry, zero-correction forward and deployment
reload checks remain part of genuine setup.

The extracted bundle is shared immutable input, with an atomic completion
receipt. Its archive, manifest and each file are authenticated. Campaign outputs
live separately and reserve eight GiB plus one GiB filesystem headroom; extraction
and the environment consume additional space. No automatic deletion or retries.
Final-test access remains sealed; original50k overlaps validation and is report-only.

Queue only clean pushed source after a full dry review. The detached dispatcher
may submit the four fits automatically within the reviewed resource envelope
after successful setup. It may reduce measured walltimes, not alter the science
or expand resources. All submission and worker claims bind exact job IDs.

Implementation status and measured evidence belong in HANDOFF. Local tests do
not establish Hydra feasibility; actual environment creation and shared setup
must succeed on Brown.
