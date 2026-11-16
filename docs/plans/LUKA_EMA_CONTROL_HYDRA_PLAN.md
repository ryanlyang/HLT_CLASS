# Nine-fit A6000 EMA control campaign

Approved scope: three objectives at seeds 7/8/9, nine independent scientific
fits. This is separate from the 42-fit stability campaign and SOFT recovery.

| Arm | Training objective | Comparison |
| --- | --- | --- |
| CE | Native HLT cross entropy only | Establish the CE-only EMA baseline |
| U | CE + 5 SmoothL1 to frozen U100 embeddings | U minus same-seed CE |
| R24 | U + full OFFLINE particle reconstruction | R24 minus same-seed U |

Each fit uses one RTX A6000, FP32 with TF32 disabled, physical batch 128,
30 complete passes through the authenticated 299234 TRAIN jets, and full 74895
row development validation. Preserve the original AdamW, three-epoch warmup,
3e-4 peak and cosine decay to 1.5e-5; no optimizer reset or LR scaling. R24
uses the unchanged hard assignment, complete 298-slot OFFLINE support,
block 8 tokens and pooled HLT representation, coefficient
`0.3 * clip((position - 3) / 3, 0, 1)` on the end-of-update example clock.
Decoder VJPs use chunks of 8. There is no soft assignment or gradient intervention.

Every trajectory records RAW, RAW_BN, EMA05 and EMA10. **EMA05 is primary**;
EMA10 is secondary and RAW/RAW_BN are diagnostics. Select each stream's best
checkpoint separately using the original validation selection rule. Never
report the maximum across streams as the predeclared primary method. EMA
averages native parameters only, starts at position 3 and uses half-lives of
0.5/1 TRAIN epoch. RAW_BN and both EMA copies recalibrate BatchNorm on the
same 8192 identity-selected TRAIN rows, with dropout disabled. Evaluation must
leave the live model's buffers and training RNG unchanged.

Setup creates one **untrained**, seeded canonical model state per seed. All
three arms load identical native and predictor tensors; only R24 loads the
decoder. Record component and named-tensor hashes. This prevents host-dependent
initialization differences from confounding paired comparisons. Training uses
matched full-TRAIN permutations and starting dropout streams. Decoder and
readout RNG are isolated. CE has no embedding/reconstruction target access in
its training loader and no predictor/decoder gradients or AdamW updates. Its
passive predictor is retained solely to keep construction and initialization
identical; it is absent from all native-only deployment exports.

Seeds7/8/9 were used in the earlier stability campaign. This is a controlled
development comparison, **not fresh unseen-seed confirmation**. Historical
M0HLT/OFFLINE recovery anchors remain fixed. Report AUC, recovery, each seed,
mean/min/sample SD, U-CE and R24-U with the same seed/readout, and fixed late
epochs 21-30 separately from selected results. Original 50k overlaps validation;
no independent test or final-test inference is authorized.

Queue seed-major triplets: CE_s7, U_s7, R24_s7, then seed 8, then seed 9.
Science fits have no dependencies on each other. One GPU setup measures all
three objective costs, complete support and all four readouts, verifies
independent one-step objective parity and all nine production miniatures,
exports/reloads native models and replays the historical checkpoint. A durable
CPU release waits afterok, authenticates setup and submits all nine fits plus
an afterany report. Twelve jobs total. Poor finite AUC is never a gate.

Resources: 6 CPUs / 32000 MiB per GPU fit/setup, setup 6h ceiling, fit 48h ceiling,
2x measured timing safety plus 30 min overhead. CPU release/report 1 CPU / 4000 MiB,
1h. Registration measures serialized output allowance; setup checks actual
memory, storage and timings before release. No four-GPU performance estimate is
used to qualify A6000. Reuse authenticated local inputs; no new transfer is
needed. New outputs and source are immutable and separate from existing jobs.

Implementation must provide focused local tests, actual installed-Weaver CUDA
miniatures where available, source/donor lineage, exact Slurm submission
journals, low-space read-only results/progress and a copyable Hydra runbook.
Real A6000/full-population qualification remains an on-site setup requirement.
