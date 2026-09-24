# Springer Nature upload-ready source package

Use `main.tex` as the primary manuscript file. It contains the complete Abstract, Introduction, Related Work and Positioning, Methods, Results and Discussion. `supplementary.tex` is the supplementary information source.

## Included source files

- `main.tex`
- `supplementary.tex`
- `references.bib`
- `sn-jnl.cls`
- `sn-nature.bst`
- all figures referenced by the two TeX files
- `main.pdf` and `supplementary.pdf` for visual checking

## Local build

```bash
tectonic main.tex
tectonic supplementary.tex
```

or, with a full TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplementary.tex
```

## Before journal submission

Replace every `AUTHOR INPUT REQUIRED` marker with the final author, affiliation, ethics, data availability, code availability, funding and contribution information. The frozen scientific boundaries must remain unchanged: PREPARE does not outperform SpeechCARE, Agent corrections did not alter prediction, and automated report scores are not physician validation.
