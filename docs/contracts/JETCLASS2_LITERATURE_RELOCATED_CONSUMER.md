# Frozen NOISE_V3 relocated consumer v1

Authority: the frozen NOISE_V3 production plan and the user's authorization to
copy the complete dataset and paired offline snapshot to Oscar. This is an
additive read-only consumer, not a new scientific population or response model.
Original production, pilot, split and classifier contracts are unchanged.

`JC2_LITERATURE_RELOCATED_READER/v1` describes a runtime inspection report with
canonical content hash, original manifest/study parents, resolved local paths,
counts, units and truthful access flags. No producer JSON is rewritten, no new
dataset manifest is published, and no architecture-specific generator is run.
The saved report is informative; readers authenticate the actual inputs again.

The default trust anchor is the externally reviewed complete manifest hash
`26c902d04988b3f32be98a8d2ccbed6d48dd8c3f49c71ddc2c660da0f2ec5deb`.
An explicit different anchor permits a separately reviewed same-contract copy,
not silently accepting whichever hash is found on disk. Counts remain the
registered 1M train / 250k validation / 1M test. Recipe must be frozen NOISE_V3;
this is not the older CMS-calibrated proxy.

Paths are supplied separately as proxy root, offline ROOT root and provenance
root. Declared references within the original producer root resolve under the
local proxy root. Split references within the original offline snapshot resolve
under the local offline root's parent. The inventory and three direct pilot
references resolve under `provenance_root/<original-parent-name>/<filename>`.
Every reference is checked against its original size (when supplied) and SHA256.
Relative paths must remain within the corresponding root, without symlinks or
traversal. There is no fallback to an original absolute RIT path.

Construction authenticates the complete manifest, study, population, recipe
bundle, inventory, profiles and copied direct pilot references, opening JSON
only. It does **not** reopen pilot physical blocks or recreate pilot evidence;
the reviewed production manifest anchors the inherited provenance. It does
not claim fresh full-bank verification or full transitive pilot archival.

Only `train` and `validation` are consumer capabilities. Unknown/test roles and
cross-role shard subsets fail before opening receipts, ROOT or NPZ payloads.
A complete manifest's test-materialization flag is not evaluation permission.
Ordinary readers do not require opening test receipts, test banks or test ROOT
files; corrupt test payloads cannot cause an ordinary read to touch test data.

On each requested shard, authenticate the ordinary role release against the
complete manifest, receipt bytes/content, producer attempt and preflight, exact
shard registration and bank sequence/counts. Verify each bank's bytes before
and after decoding, physical array schema and **every canonical jet identity**
against inventory hash + original relative file + tree key/cycle + raw entry.
Only one bounded bank is held at once; original entries are reconstructed from
the authenticated mask, never compressed row ordinals.

Paired reads authenticate the original offline ROOT bytes before opening and
when closing, including early generator closure. Tree key/cycle and entry count
must match. Read only offline particle branches, in at most 512-entry windows;
optional labels additionally read `jet_label` and the inherited `hlt_matched`
eligibility scalar. Native `hlt_part_*` particles are never read. Labels use the
existing frozen 11-class map and are targets, not model inputs. Labels default
off. There is no membership resampling or balancing in this loader.

Return `PairedJet(identity, role, source_file, tree_key, entry, offline, proxy,
label)`; physical endpoints are existing `Particles` objects in GeV and mm.
Particle-index equality does not imply an offline/proxy correspondence. Stored
ancestry is absent; consumers must not invent it. Geometry/derived features must
come from each supplied view. Identity, raw entry, filenames, particle keys,
labels, and generation metadata are forbidden deployable inputs.

The companion CLI has metadata-only `inspect` and bounded `smoke` (1..128
ordinary paired jets), no fitting, Slurm submission, regeneration, matching,
final-test reader, or data-writing capability. This does not port a SPORC
training campaign or authorize GPU science; a new Oscar execution needs its
own source/environment/runtime acceptance.
