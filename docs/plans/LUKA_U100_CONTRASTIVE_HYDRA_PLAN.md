# Paired OFFLINE identification on Hydra: independent remaining fits

Date: 2026-11-03. User requests moving the contrastive campaign to Hydra and
explicitly wants all students queued together with no student dependencies or
campaign concurrency limit. This is a separate execution of the original
[twenty-row scientific plan](LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md), retaining
30 epochs. The separate serial/patience execution is not imported as training
policy. Original jobs and results remain untouched.

## Science and reuse

Keep CE weight 1 + U100 embedding weight 5, the independent OFFLINE identification head, all
twenty coefficient/seed/sampler registrations, full 299234/74895 populations,
batch 128, BF16 backbone/FP32 auxiliary losses, original optimizer and schedule.
Native HLT inference only; final test remains sealed. No new teachers or targets.

Import the successful `431341db5eced8eaa3e52b37ba047f8cd4beed50` SPORC campaign
through original-source replay, checking its clean source snapshot first.
Authenticate setup, accepted jobs, target banks and completed fits. Each reused
student requires COMPLETED/0:0 accounting and a valid completed receipt. A
COMPLETED student without a receipt is inconsistent and stops export. Other
rows are new Hydra attempts; partial outputs never become completed results.
Existing SPORC pending/running jobs are not cancelled, held or modified by this
adapter. Record their snapshot states; later SPORC completions do not silently
change this immutable Hydra execution.

Reuse the Brown `bundle_65952ed3` HLT views, U100 targets and historical references
in place. Transfer only a new directory containing the OFFLINE TRAIN keys and
manifest, original metadata/source/accounting proof, and completed student
reports, receipts and native validation predictions. Do not transfer raw inputs,
particle caches, teacher/student weights or another U100 bank. Authenticate the
transfer manifest hash explicitly at import; preserve original JSON bytes and
absolute paths as evidence only. Do not dereference SPORC paths on Brown.

Reuse completed Hydra precision diagnostic 131484 under its existing v2 policy
and frozen matched control; no repeat control inference or reference fit. This
qualifies numerical portability, not contrastive runtime resources. Shared setup
loads/join-checks transferred TRAIN keys and existing HLT/U100 arrays, then runs
the original six production sampler-family miniatures, native zero parity,
support stress and measured runtime checks on one L40. No separate preflight.
No teacher forward passes or target re-export take place.

## Execution

Use Hydra `gpus/default/normal`, one `gpu:nvidia_l40:1`, six CPUs, 120000 MiB per
job and the existing pinned environment. Setup cap 12h, fit cap 24h, report 2h.
Reserve 8 GiB outputs plus 1 GiB filesystem headroom, separately from the delta
transfer. Actual measured resources must fit these original limits.

Save the full dry plan before setup submission. A detached dispatcher waits for
exact successful setup accounting and authenticated measured acceptance, then
submits every remaining student in the original registry order. **Every student
has an empty dependency list and no `--dependency` option. No lanes, array `%`
limit, or campaign concurrency throttle.** Ordering is submission order only;
Hydra decides actual starts. The report alone uses `afterany` on all newly
submitted fits. Setup failure releases no students. Failed students do not
block other students. No automatic retries or duplicate submissions.

## Reporting and limitations

Combine authenticated SPORC completions and Hydra attempts with explicit site,
job, epoch and completion labels; preserve all twenty rows and incomplete rows.
Use the same full validation identities/labels; original50k overlaps and is
report-only. Report historical SPORC M0HLT/OFFLINE gap recovery. Provide matched
seed-control deltas, labelling the site's control source, and a separate delta
against the saved Hydra replay of the seed0 checkpoint. Replayed inference does
not imply equivalent training trajectories across GPUs. Do not label the
historical SPORC controls as newly Hydra-trained models.

Implementation must remain additive under `luka_u100_contrastive_hydra`, reusing
the original model/sampler/training/resources unchanged. Validate source in its
own checkout, immutable artifacts, exact job ownership, transfer joins and
independent dependency graphs. Local tests do not claim real L40 acceptance.
