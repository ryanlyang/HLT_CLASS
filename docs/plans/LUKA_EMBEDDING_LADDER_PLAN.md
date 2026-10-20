# Fixed-OFFLINE embedding prediction plus logit ladder (v1)

Authorized 2026-10-20. This additive experiment supersedes the single-stage
restriction only in its NEW namespace; old embedding/attention artifacts and
scientific meanings remain immutable. Use trial ALLJETS, not Oscar FullStat:
299234 TRAIN / 74895 validation, no independent test, original50k report-only.

Reuse the original fixed30 selected U000 as the starting classification teacher.
Train five fresh students U050 -> U100 -> D066 -> D033 -> D000. Each gets T=2
probabilities from its own immediately preceding selected teacher, and the SAME
frozen pure OFFLINE 128-vector targets authenticated from the completed single-
stage embedding campaign. U000 is not pure OFFLINE. Do not change target axes,
normalization, teacher or target precision across rungs.

Loss: 0.25 CE + 0.75 * 4 KL(q_T || p_T) + 0.3 mean SmoothL1(head(h), z_OFFLINE).
The last term is identical to the single-stage auxiliary kernel (TRAIN-only
standardization, FP32, beta=1), but CE now has coefficient 0.25, not 1. The small
128-GELU-128 head has isolated RNG and never feeds the classification head.
Native backbone is fresh at every rung, with original coordinate seeds, true
batch 128, AdamW, fixed 30-epoch schedule and full-validation AUC-first selection.
No attention loss, warm start, fusion or additional inference input.

Primary causal comparison is a matched LOGIT-only five-rung ladder. The existing
fixed30 campaign was cancelled before its D side; the later three-rung LOGIT
campaign is not silently relabelled. Register a fresh LOGIT control alongside
EMBED, same shared U000 and same per-coordinate seeds. Both complete all rows,
regardless of scientific quality. Existing M0HLT/OFFLINE and the single-stage
embedding result are supplementary references. No final-test operation exists.

One real debug A100 preflight measures both objectives at every coordinate,
native FP32/BF16 parity, longest TRAIN backward/validation inference support,
full-population epochs and selected-state/temperature2 export. Science is two
separately reviewed single-node sequential ladders. Each measured whole-chain
request must fit 24h, using less than 85% of host and 90% of GPU memory;
no schedule reduction to pass.
If this additional matched control is not desired, do not silently substitute
an unmatched historical run: the operator may queue only EMBED and label the
comparison historical via the separately reviewed plan.

Reuse compact TRAIN embedding targets read-only, without regenerating them.
All requested view caches are authenticated; deleted U000/U050 views are
rebuilt in RAM only with existing assignments and exact saved fingerprints.
Corrupt existing bytes fail, rather than triggering reconstruction. No new
durable particle caches, matching rerun, attention banks or data copies.

Compact logit targets remain in RAM, with content/identity/teacher descriptors
in durable records. Selected intermediate model files are roundtrip-checked,
used to export next TRAIN probabilities, then individually retired with receipts.
Final D000 native checkpoint, every epoch history, selected validation predictions,
IDs/labels, original50k indices, comparisons and provenance remain. Shared parent
models, embedding targets, old campaigns and Fresh_check are never deleted.
Reserve 2 GiB new persistent storage. No automatic retries or follow-up.

Acceptance: focused baseline and new tests, exact zero-aux native KD parity,
all-rung identity/teacher/target guards, saved-input reconstruction, native
deployment parity, no offline validation targets, independent-arm teacher
chains, finite resource measurements, immutable exact submission journals,
safe owned retirement and replayable compact results. Installed local Weaver
fixtures are not remote acceptance; real debug preflight remains mandatory.
