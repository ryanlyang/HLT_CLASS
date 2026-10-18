# Cold-start privileged-gate ladders on tier3

Authorized 2026-10-18: implement until queue-ready, not submit. This new
campaign changes no existing artifact, job, source worktree, raw dataset,
prepared cache, U100 anchor, or Fresh_check file. No retrospective cleanup.

## Frozen scientific design

Reuse the exact fixed30 COARSE_U100 anchor from commit
74aa58c582ff6f497a2c6d91c503f5489fac3e9a and ALLJETS 299234 TRAIN / 74895
validation. Original50000 is report-only. Historical test is consumed and
retired into TRAIN; there is NO independent test.

Three chains: LOGIT, SELF, PRIV, each U100 -> D066 -> D033 -> D000.
Every student has fresh seeded native/gate weights, buffers and optimizer;
no teacher parameter is copied. Teacher weights load only into a separate
evaluation-only target producer. Each chain supplies its own selected teacher
to its next rung. Seeds, fixed30, batch128, accumulation1, AdamW warmup3/cosine,
C25/P75 T2, AUC-first checkpoint selector and aligned views are unchanged.

LOGIT has no gate. SELF has own/own gate inputs. PRIV has own/immediately
richer inputs (U100, D066, D033). Reuse the original rank4, hidden32, head-shared
bounded additive log gate in blocks6/7; native physics bias remains unchanged.
PRIV is an offline-dependent oracle, including at D000.

From the new selected PRIV_D000 teacher, fit fresh REPLACE_LOGIT and
REPLACE_GATE students with own/own inputs; the latter adds 0.1 centered log-gate
MSE. No staged freezing of a random backbone. SELF_REDISTILL is a fresh
own/own student of SELF_D000, matching the extra fit stage. All three final
students are HLT-only. Twelve fits total; no scientific row is quality-gated.
Matched thirty epochs measure fixed-budget performance, not convergence parity.

## Bounded persistent storage

One job per fit, plus aggregate. Each fit creates one private, verified
node-local scratch directory, exports its teacher's TRAIN probability bank
there (plus nine FP32 factors only for the two PRIV_D000 replacements),
authenticates it, fits and publishes the selected checkpoint, validation
predictions/IDs/labels/original50k mask, all epoch metrics, and provenance.
After these authenticate, retire ONLY the exact owned target files, with
intent and completion receipts. No dense attention bank, optimizer snapshot,
per-epoch checkpoint, particle-cache copy or persistent teacher bank.

Selected checkpoints are deliberately retained: they are compact and permit
downstream target rebuilding and future inference. The retained validation
probabilities support future R30/R50 diagnostics without inference. Scratch
manifests are historical evidence, NOT reusable target payloads. Missing
persistent outputs remain corruption; explicit scratch retirement is normal.
No silent fallback to home. Shared input caches and old campaigns are read-only.
Failed jobs preserve partial outputs; there is no automatic retry or deletion
of unknown/failed directories. Node-local scratch may be purged by the site.

## Execution and acceptance

Both gate and science use SPORC tier3 / reu-aisocial / qos_tier3, one A100,
6 CPUs,128000MiB. Preflight requests8h. Each fit has its own measured bound
(including teacher export and authentication); ceiling24h. Aggregate1CPU/16GiB/1h.
The gate runs native/gate FP32/BF16 parity, cold initialization witnesses,
maximum-support batch128 stresses for both roles at every node, and the full
one-pass graph using the production workers and owned-scratch retirement.
All technical artifacts are isolated and never become scientific teachers.
Budget host/GPU headroom, scratch independently and a conservative2GiB
persistent reserve for all technical/scientific selected outputs. Verify actual
output size. No shortening/truncation to force resource admission.

New LUKA_COLD_GATE_* /v1 contracts separate this science and storage lifecycle
from old warm campaigns. Exact pushed clean source, original-source anchor
replay, immutable specs, dry/live plan hashes, exact DAG journals and measured
real A100 acceptance are required. An explicitly authorized optional automatic
follow-up may submit the thirteen science jobs only after successful accounting
AND validated gate evidence. No old preflight acceptance is reused.

Compare cold endpoints against warm counterparts only on identical validation
identities and clearly identified initializations. Existing warm runs are not
required for execution and are not copied into the new campaign.
