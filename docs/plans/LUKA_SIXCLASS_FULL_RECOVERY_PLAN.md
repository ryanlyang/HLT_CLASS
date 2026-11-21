# Full-validation DE000 epoch comparison with fixed recovery anchors

Authorized analysis, 2026-11-21. Compare original balanced DE000 (job9198212)
with lower-LR CE-only-main-head DE000. Use old M0HLT (9198208) and new OFFLINE
(9442678) as fixed anchors. Both anchors retain their published primary
in-range macro-AUC-selected epochs for both metrics. Do not use exploratory
Hbb-selected anchors or change training selection.

Use all566032 authenticated, naturally distributed validation jets:
328589 in range plus237443 outside, without a pT cut. Classes are QCD, X_bb,
X_cc, X_ss, X_qq and X_gg; counts432708,26937,26592,26483,26449,26863.
Report two epoch tables in order: six-class macro OVR ROC AUC on all rows,
then binary Hbb/QCD AUC on26937Hbb+432708QCD using p(bb)/(p(bb)+p(QCD)).
Each table contains old/new DE000 AUC, recovery and new-minus-old AUC.
Both columns use100*(DE000-oldM0HLT)/(newOFFLINE-oldM0HLT), separately for
each metric. A nonpositive anchor gap is unresolved; negative recovery is valid.

Full macro AUC was not saved per epoch. Replay completed states on validation
only and pool predictions before computing metrics; do not average regional
AUCs or relabel in-range AUC as full. Read both anchors' authenticated saved
predictions, avoiding additional OFFLINE inference or OFFLINE particle caches.

One separate L40S allocation shares one HLT validation view between both DE000
histories, preserving original128-row batching, padding, BF16 and runtime.
Replay original-selected DE000 first and require exact published probability
agreement. Every epoch must reproduce saved in-range metrics; new DE000 must
also reproduce its saved full Hbb diagnostics. No TRAIN or test particle spans
are read. Raw checksums and shared scalar metadata may authenticate populations.

Snapshot the contiguous committed new epoch prefix when analysis starts;
completion markers determine availability, not log lines. Training may continue.
Resume preserves that snapshot; later epochs need a separate analysis output.
No original source, job, checkpoint, report, teacher bank or claim is changed.

Use the persistent HOME runtime,6CPUs,192000MiB,oneL40S,24-hour ceiling (not
a measured duration estimate). Save only small metric receipts and a report.
Replay qualification occurs inside this evaluation allocation, with no separate
preflight fit. Final test remains sealed. This is development validation and
the new campaign remains a combined LR/objective/teacher experiment.
