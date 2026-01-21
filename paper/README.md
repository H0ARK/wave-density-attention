# WDA Paper (Legacy Location)

This folder contains the LaTeX source for the Wave-Density Attention paper. The behavioral distillation paper now lives in `behavioral-distillation/`.

## Build

If you have LaTeX installed:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

(If you use BibTeX later: run `bibtex main` between the pdflatex passes.)

## Files

- `main.tex`: paper source
- `refs.bib`: bibliography (optional / currently minimal)
