# Supplementary materials

This repository contains reproducibility files for the manuscript:

**A Dimensionless Criterion for Evaluating Flexible Foundations under Earthquake Excitation**

Target journal: *Bulletin of Earthquake Engineering*.

## Contents

- `evidence_buee_20260509/`: benchmark inputs, outputs, validation summaries, generated figures, and record files.
- `run ssi dimensionless benchmark.py`: dimensionless response sweep.
- `run ssi spectrum validation.py`: direct spectral and record-based validation.
- `run ssi expanded validation.py`: expanded spectral stress-test validation.
- `make verified publication figures.py`: verified publication figure generator used after screening AI-generated figures.

## Key validation results

- Dimensionless sweep: 1,764 cases.
- Direct spectral validation: 7,680 cases, 82.1% classification agreement.
- Expanded validation: 30,720 cases, 93.6% classification agreement.
- Mean absolute force-ratio error in expanded validation: 0.063.

The AI-generated figures were screened before use. The final manuscript uses verified local figures generated from the reproducible model because the AI-generated force-ratio heatmap inverted the validated benchmark pattern.
