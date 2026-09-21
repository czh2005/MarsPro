MarsPro English BIBM manuscript (final integrated version, 2026-09-19)

Compile entry: main.tex
Template: IEEEtran conference, two columns
Verified length: exactly 8 pages, including references and appendices
Submission mode: BIBM 2026 main-conference double blind; author block intentionally empty
Recommended engine: pdfLaTeX on Overleaf (Tectonic was used for local verification)

Contents:
- main.tex and IEEEtran.cls
- figures/: four main-text figures; Figs. 2--4 use one sans-serif typography system and the manuscript palette. Fig. 3 reports the matched-seed surface, sequence-recovery, confidence, and US-align analysis on 798 registered backbones
- data/: source tables, aggregate statistics, sealed locked-test metrics, complete-source holdout results, matched-seed design statistics, and fixed-test redesign/rescoring summaries

Reference numbering follows IEEE order of first appearance. The Chinese and English manuscripts use the same 32-item numbering map.

Scientific boundary:
Model selection and ablations use validation. The selected shared ProtT5-LoRA checkpoint was evaluated once on the fixed test split with frozen thresholds (macro AP 0.97529). Four complete scientific-source retrainings evaluate source transfer; all rank above prevalence, while threshold transfer is source dependent. The matched-seed structural study contains 798 backbones, three seeds, and 4,788 completed refolded designs. Surface STNQDE and Met/Cys reference distances decrease, whereas US-align RMSD and pLDDT remain similar and TM-score/GDT-TS decline slightly; the study does not support structural improvement. A separate prediction-blind redesign challenge uses 1,000 distinct fixed-test ID50 clusters and three matched seeds; frozen rescoring rises from 0.61854 to 0.67251, with 731/1,000 backbone-level comparisons improving. Frozen rescoring is a model-mediated consistency endpoint, not experimental functional validation.
