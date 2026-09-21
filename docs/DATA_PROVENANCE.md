# Data provenance

## Stage 1 resource

The 40,000 records are released as fixed `train.csv`, `validation.csv`, and `test.csv` files. `metadata_sidecar_40k.csv` joins one-to-one by `sequence_id` and records sequence hashes, ID50 clusters, source identifiers, taxonomy, relationship components, Pfam fields, and evidence-layer fields.

The release also includes:

- `source_inventory_40k.csv`: every scientific source identifier represented in the 40k table and its split counts;
- `source_registry.csv`: source title, type, URL, access status, evidence meaning, and redistribution-review status;
- `field_dictionary.csv`: column definitions;
- `release_audit.json`: row counts, label-state counts, metadata join, and split-leakage checks;
- construction scripts and union-zero audit tables.

## Stage 2 structure corpus

`structure_corpus_index_10000.csv` contains 10,000 unique sequences with at least one positive source-association label. It records the expected relative PDB path and structure quality metadata when available. The PDB/CIF files are excluded from this repository; their absence is explicit in `structure_corpus_manifest.json`.

## License status

The release distinguishes auditability from redistribution permission. Empty historical license fields are normalized to `not_verified_for_redistribution` in the public source inventory. This status does not invalidate the scientific provenance, but it must be resolved before unrestricted redistribution of affected third-party sequence collections.
