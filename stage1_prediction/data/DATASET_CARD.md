# MarsLikePro-40K dataset card

## Contents

- 40,000 unique protein sequences.
- Fixed splits: 36,000 train, 2,000 validation, 2,000 test.
- Six broad multi-label tasks: `cold`, `desiccation`, `oxidative`, `perchlorate`, `radiation`, and `salt`.
- Three-state labels: `1`, `0`, and `unknown`.
- `id50_cluster_id` supports homology-aware leakage checks.

## Label meaning

`1` means that the protein belongs to a source/proteome or evidence scope associated with the corresponding condition. `0` is retained from the original source rule or added from a documented source-sensitive contrast. `unknown` means the task is unobserved for that sequence. Neither `1` nor `0` should be rewritten as direct experimental functional proof without separate evidence.

## Final counts

Counts and source-zero rules are recorded in `总结.md`. Newly added source zeros were created only from unknown positions, never by overwriting an existing positive. No pseudo-zero labels are used in this release.

## Integrity

The split CSV hashes are regenerated in the repository root `MANIFEST.sha256`; the source package's original checksum file is retained as `checksums.sha256`.
