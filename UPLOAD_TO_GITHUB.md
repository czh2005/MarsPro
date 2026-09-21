# GitHub upload checklist

Create the complete repository as **private** first. Its sequence tables include
sources whose public redistribution status is not yet fully verified. Do not
switch repository visibility to public until `PUBLIC_RELEASE_CHECKLIST.md` is
satisfied.

1. Run `python scripts/verify_release.py` and require `PASS`.
2. Install Git LFS, then run `git lfs install`; `.gitattributes` already tracks model and workbook binaries.
3. Review `DATA_LICENSE.md` and `stage1_prediction/data/provenance/source_inventory_40k.csv`. Code and author-created metadata have explicit licenses; third-party sequence collections with `not_verified_for_redistribution` still require a release decision.
4. Initialize and upload:

```bash
git init
git lfs install
git add .
git commit -m "Initial reproducible MarsPro release"
git branch -M main
git remote add origin <repository-url>
git push -u origin main
```

No individual file may exceed GitHub's 100 MB hard limit. Do not add adjacent backup/staging directories. PDB/CIF files and base-model weights are intentionally excluded.
