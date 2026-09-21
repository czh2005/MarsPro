# Sequence redistribution audit (2026-09-22)

## Decision

The 40,000-sequence release is scientifically traceable, but it is **not yet cleared as an unrestricted third-party sequence redistribution package**. The audit distinguishes:

1. the license of the article or code repository;
2. the terms of the upstream sequence database (for example UniProt or NCBI); and
3. the license of our author-created metadata, labels, mappings, and scripts.

An open-access article does not by itself grant permission to redistribute every sequence copied from an external database or every derived collection. PMC also states that license terms must be checked record by record and that material without an explicit license should be treated conservatively.

## Current release rule

- The repository may publish code, documentation, hashes, source identifiers, accession IDs, label mappings, and author-created derived statistics under the licenses stated in `DATA_LICENSE.md`.
- A sequence FASTA or a bulk sequence table may be redistributed only when the upstream record terms have been verified for that exact source and the required attribution is retained.
- Until that check is complete, use the accession-only release and a reconstruction script/API route. Do not describe the GitHub repository as granting redistribution rights for all 40,000 sequences.
- The 10,000 structure corpus remains index-only; PDB/CIF files are intentionally excluded.

## Source-level disposition

| Source ID | Main upstream | Evidence checked | Sequence redistribution disposition | Required action |
|---|---|---|---|---|
| CO001 | AFP-R database | Public browse/help pages; no explicit reuse license located | **Not confirmed** | Obtain AFP-R terms or release accession-only records |
| DE001 | PMC11312438 and its sequence sources | Article is accessible; sequence-level upstream terms not established from the current inventory | **Upstream terms required** | Record exact accession source and its license |
| DE003 | PMC9283220 / GEO-linked records | Article/data access is public; sequence-level terms not established | **Upstream terms required** | Verify GEO/NCBI and sequence-record terms |
| DE004 | PLOS ONE / PMC3306391 | Article states CC BY for the article | **Article reuse confirmed; sequence bundle pending** | Verify whether each sequence is article-created or copied from UniProt/NCBI |
| OX002 | PMC2676504 | Public full text; exact sequence redistribution basis not recorded | **Upstream terms required** | Verify sequence accession source and terms |
| OX005 | PMC3219624 | Public full text; exact sequence redistribution basis not recorded | **Upstream terms required** | Verify sequence accession source and terms |
| PE002 | PMC4438318 | Public full text; exact sequence redistribution basis not recorded | **Upstream terms required** | Verify sequence accession source and terms |
| RA001 | PMC4401554 | Public full text; exact sequence redistribution basis not recorded | **Upstream terms required** | Verify sequence accession source and terms |
| MX001 | GEO GSE58325-derived mapping | Public NCBI/GEO record; mapping bundle terms not frozen | **Public access, redistribution not confirmed** | Preserve GEO accession and rebuild mapping from source records |
| RA006 | GEO GSE3876-derived mapping | Public NCBI/GEO record; mapping bundle terms not frozen | **Public access, redistribution not confirmed** | Preserve GEO accession and rebuild mapping from source records |
| SA006 | GEO GSE4447-derived mapping | Public NCBI/GEO record; mapping bundle terms not frozen | **Public access, redistribution not confirmed** | Preserve GEO accession and rebuild mapping from source records |
| ISS_SA_HALOCLASS | UniProt-derived collection plus project repository | Paper reports UniProt sequences; project code repository has MIT license | **Code confirmed; sequence terms follow UniProt** | Retain UniProt attribution and exact accessions |
| ISS_SA_HPCLAS | UniProt/NCBI-derived collection plus project repository | Repository has no visible LICENSE; README identifies train/test FASTA files | **Not confirmed** | Obtain author permission or publish accession-only data |
| ISS_SA_HALOMPNN | HaloMPNN preprint and derived collection | Preprint states “All rights reserved. No reuse allowed without permission.” | **Permission required for preprint-derived data** | Exclude derived sequences unless independently sourced and cleared |
| ISS_CO_ESMPSYPRED | ESM-PsyPred repository/paper | Public repository README found; no repository license located | **Not confirmed** | Obtain author permission or use accession-only references |
| ISS_CO_AAC | Psychrophilic-enzyme benchmark | Internal source label has no frozen public license record | **Internal only until verified** | Identify the original sequence source and terms |
| ISS_OX_AOP2020 | Antioxidant benchmark | Internal source label has no frozen public license record | **Internal only until verified** | Identify the original sequence source and terms |
| ISS_RD_DNAREPAIR2009 | DNA-repair benchmark | Internal source label has no frozen public license record | **Internal only until verified** | Identify the original sequence source and terms |

## Confirmed facts and non-facts

**Confirmed:** UniProt documents CC BY 4.0 for copyrightable parts of its databases. This supports redistribution of UniProt-derived records when attribution and the applicable record-level conditions are preserved. The HaloClass source repository publishes its code under MIT.

**Confirmed:** the HaloMPNN preprint is not a safe source for redistributing its derived sequence collection without permission.

**Not confirmed:** that all sequences in the 40k union came from a single upstream license; that all paper tables can be repackaged as FASTA; or that a public webpage implies redistribution permission.

## Public release correction

The current public repository therefore uses an **accession-first release**: source IDs, accession identifiers, hashes, labels, splits, mappings, scripts, and derived statistics are public; third-party sequence content remains conditional on its upstream terms. A future `sequence_release_manifest.csv` may promote a source only after recording its exact license URL, access date, accession coverage, attribution text, and checksum.

## Evidence links

- UniProt API support-data license: https://www.uniprot.org/api-documentation/support-data
- PMC copyright and license guidance: https://pmc.ncbi.nlm.nih.gov/about/copyright/
- DE004 article record: https://pmc.ncbi.nlm.nih.gov/articles/PMC3306391/
- HaloClass source repository license: https://raw.githubusercontent.com/kushnarang/haloclass-source/main/LICENSE
- HaloClass paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11543744/
- HPClas repository: https://github.com/Showmake2/HPClas
- ESM-PsyPred repository: https://github.com/tust-lamee/ESM-PsyPred
- HaloMPNN preprint: https://www.biorxiv.org/content/10.64898/2026.08.02.742362v1
- AFP-R public help page: https://bio-comp.ucas.ac.cn/AFP/help/

