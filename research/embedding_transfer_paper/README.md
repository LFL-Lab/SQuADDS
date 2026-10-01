# Physics-aware layout embeddings for transferable device models

This directory is a self-contained, Overleaf-ready snapshot of the working
paper package. The editable working copy also lives outside the SQuADDS source
tree at `../embedding-transfer-paper/`; this in-repository snapshot keeps the
manuscript, generated figures, evidence ledger, and compiled PDF reviewable in
the same pull request as the exact encoder and tutorial code that produced them.

## Target and format

The working target is a **Nature Machine Intelligence Article**. As checked on
2026-09-09, that format allows a 150-word abstract, up to 3,500 words of main
text (excluding Methods, references, and figure legends), and up to six display
items. Initial submissions may be supplied as TeX/LaTeX plus a compiled PDF.

Official guidance:

- https://www.nature.com/natmachintell/content
- https://www.nature.com/natmachintell/submission-guidelines/initial-formatting

The journal choice should be revisited after the pre-registered representation
and few-shot results are complete. A physics-facing alternative is *Physical
Review Applied*; a methods/data-facing alternative is *Scientific Data* plus a
separate physics/ML paper.

## Files

- `main.tex` — working manuscript.
- `references.bib` — verified references only; placeholders are marked TODO.
- `claims-and-evidence.md` — claim gate that prevents the prose from outrunning
  the experiments.
- `figures/README.md` — transcript-driven evidence/figure map and export naming;
  final panels will be consolidated only after the results are stable.

## Build

Run `latexmk -pdf main.tex` when a TeX distribution is available. All generated
figures should be copied into `figures/` from the executed acceptance notebook.
