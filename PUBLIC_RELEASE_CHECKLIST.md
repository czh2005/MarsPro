# Public release checklist

The repository is ready for a private GitHub backup and collaborator review.
Before changing repository visibility to public:

1. Review `stage1_prediction/data/provenance/source_inventory_40k.csv`.
2. Resolve every `not_verified_for_redistribution` sequence source, or publish
   a metadata-only release that excludes the affected sequence rows.
3. Confirm the upstream terms for pretrained models and third-party software;
   base-model weights are intentionally excluded here.
4. Run `python scripts/update_manifest.py` followed by
   `python scripts/verify_release.py`.
5. Confirm that no PDB/CIF structure archive, credential, key, or local cache
   has been added.
6. Compile both Overleaf projects and visually inspect the PDFs.

The presence of a source URL or an open-access paper does not by itself grant
permission to redistribute all associated protein sequences.
