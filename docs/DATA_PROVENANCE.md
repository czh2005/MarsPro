# Data provenance

## Stage 1 resource

The public repository exposes the 40,000 records as fixed accession-first
`train.csv`, `validation.csv`, and `test.csv` files under
`stage1_prediction/data/public_accession_only/`. These tables omit raw sequence
strings and retain sequence hashes, accessions, labels, splits, and retrieval
fields. The complete sequence tables are preserved in the dated local release
package under `full_training_tables/` for authorized internal training.

The release also includes:

- `source_inventory_40k.csv`: every scientific source identifier represented in the 40k table and its split counts;
- `source_registry.csv`: source title, type, URL, access status, evidence meaning, and redistribution-review status;
- `field_dictionary.csv`: column definitions;
- `release_audit.json`: row counts, label-state counts, metadata join, and split-leakage checks;
- construction scripts and union-zero audit tables.

## Stage 2 structure corpus

`structure_corpus_index_10000.csv` contains 10,000 unique sequences with at least one positive source-association label. It records the expected relative PDB path and structure quality metadata when available. The PDB/CIF files are excluded from this repository; their absence is explicit in `structure_corpus_manifest.json`.

## License status

The release distinguishes auditability from redistribution permission. Empty historical license fields are normalized to `not_verified_for_redistribution` in the public source inventory. This status does not invalidate the scientific provenance, but it must be resolved before unrestricted redistribution of affected third-party sequence collections. See the dated source-by-source audit in `docs/SEQUENCE_REDISTRIBUTION_AUDIT_20260922.md`.
