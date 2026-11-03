# U100 correction campaign transfer contract v1

This contract covers **data transfer only**, following the user's request to
move only files needed by the four correction fits to Brown CS Hydra. It adds
`LUKA_U100_PAIR_TRANSFER/v1`; it does not change the
[scientific correction contract](LUKA_U100_PAIR_CORRECTION.md), original execution
specification, source guard, model, or Slurm graph. A Hydra execution adapter and
site-specific acceptance are still required before training.

The exporter runs with the existing SPORC environment. It replays the source
registered by the original correction spec, calls its campaign validation with
storage checking disabled, reimports the completed `u100_w5_s0` control from its
original source, and validates prepared metadata. The resulting reference must
equal the registered reference, including exact job21993056, population,
environment and parent hashes. Origin replay can read validation matching maps
to authenticate preparation, but those maps are not transferred. It performs
no new inference, training, raw particle reconstruction or scheduler operations.

## Exact payload scope

- Prepared D000 TRAIN and validation input blocks and U100 TRAIN input blocks.
- TRAIN matching assignments only.
- Existing standardized U100 TRAIN embedding bank and its original manifest.
- Cache/prepared manifests and foundation, inventory and split JSON.
- Original correction, matched-control, historical-U100 and fixed30 specs.
- Matched-control receipts, report, predictions, setup metadata and the **one
  selected native control checkpoint**, reserved for cross-site parity.
- Historical U100-w5 saved predictions, report, receipts, original50k indices,
  bank metadata, teacher-readback proof and saved results.
- M0HLT/OFFLINE saved validation predictions, reports and receipts for recovery.
- U100 teacher report/receipt; its identity and checkpoint hash remain in the
  original evidence. No teacher weights are needed to reuse the existing bank.

Selection uses explicit registered records, never a recursive directory copy or
a recursive traversal of every path mentioned in historical JSON. No raw ROOT,
final-test arrays, validation U100 arrays, validation matching maps, unrelated
campaign payloads, teacher weights, old worktrees, environment binaries or logs
are transferred. Existing manifests remain byte-identical, including historical
absolute paths and references to deliberately omitted files. They are evidence;
this bundle is **not** a complete historical-worktree replay on Brown.

## Integrity, storage and publication

The manifest contains original source paths, relative archive destinations,
purposes, sizes, SHA256 hashes, parent replay evidence, exporter module SHA256,
the fixed transfer policy and a canonical content hash. It retains the original
299,234 TRAIN /74,895 validation population. Historical ALLJETS evaluation is
exploratory; original50k overlaps validation. The bundle creates no independent
test role or new test access.

`inspect` is read-only. It authenticates the origin and checks existence and
sizes, but does not claim to hash all prepared input arrays. `export` streams
each allowlisted source into one uncompressed tar, comparing the bytes actually
copied with the registered size/hash. It rejects missing, corrupt, changed or
redirected inputs. It never rebuilds missing arrays or exports new targets.

The output directory must be new and outside all selected input directories.
The payload ceiling is6GiB; free-space checking reserves actual payload size,
archive/manifest overhead and1GiB headroom. This is a filesystem check, not a
claim about personal quota. An exclusive partial tar is retained on failure.
The final name is published through an atomic non-overwriting hard link after
all checks succeed; the partial link is then removed. SHA256SUMS is written for
the completed archive. Source files are neither deleted nor changed.

`verify` uses standard-library Python only. It requires the archive SHA256
obtained from the origin, then checks the manifest and every listed member.
Missing, duplicate, unexpected, nonregular or unsafe member paths fail. It does
not extract files, load models/arrays or follow original absolute paths. Hashes
detect corruption; they are not cryptographic signatures. Keep the origin
digest in the trusted terminal/session when transferring.

No network upload, SSH key copying, package installation, Slurm submission or
cancellation is implemented by this transfer tool. The Hydra consumer must bind
the verified bundle to a new site-specific spec, validate array joins and native
checkpoint parity, and perform the real shared setup before any scientific fit.
