# JetClass2 CMS-proxy Oscar portability plan

## Objective

Run the already specified 200k-train/100k-validation CMS-proxy three-spine
comparison on Brown Oscar without changing its selected jet identities,
particle matching, views, controls, seeds, or task graph.

The scientific graph remains:

- DIRECT: `U000 -> D000`;
- COARSE: `U000 -> U050 -> U100 -> D066 -> D033 -> D000`;
- DENSE: `U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000`;
- controls: `M0HLT`, pure `OFFLINE`, and `U000`.

## Why a portable artifact is required

The authenticated SPORC release and matching foundation contain absolute
SPORC paths. Oscar cannot reproduce `/home/ryreu/...`, and editing those JSON
files in place would invalidate their content hashes. The transfer therefore
has two explicit layers:

1. a root-independent `PORTABLE_BUNDLE/v1` containing only the exact ordinary
   train/validation files, proxy blocks, release bank, matching bank, study
   metadata, and inventory required by the frozen release;
2. an Oscar-local `PORTABLE_MATERIALIZATION/v1` that republishes location-
   bearing release/foundation wrappers while retaining the original hashes as
   parents and reusing the release and assignment banks byte-for-byte.

No matching is recomputed. No selected identity changes. No final-test file is
copied or opened.

## Oscar execution site

The registered production site is:

- cluster `slurmctld`;
- account `default`;
- partition `gpu`, QOS `norm-gpu`;
- one L40S GPU, one node/task;
- at most 12 CPUs and 192000 MiB;
- x86-64 environment prefix
  `/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar`.

The gate requests 12 CPUs, 160000 MiB, one L40S, and 12 hours. It performs the
same full-population one-pass preflight used to measure cache, training, and
reducer bounds. The resulting runtime profile is Oscar-native; SPORC A100
timings and GPU identity are not transferred.

## Stages

1. Export and validate the portable bundle on SPORC from the completed
   matching-foundation gate.
2. Archive and transfer the bundle to Oscar scratch; verify the archive and
   every payload hash.
3. Materialize the bundle on Oscar without recomputing release membership or
   assignments.
4. Create and dry-run the Oscar v4 preflight gate from exact pushed source.
5. Run the full-population Oscar preflight.
6. Only after the gate completes, create and dry-run the unchanged science
   campaign. Live science submission remains separately authorized.

Interrupted exports, materializations, or destinations fail closed and require
a fresh target. Existing Tigris and SPORC artifacts and jobs are not mutated.

