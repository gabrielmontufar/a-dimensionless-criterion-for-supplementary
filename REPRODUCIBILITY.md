# Reproducibility Protocol

Package version: `demand_polarity_map_v40_submission_reproducible`

The official workflow has three levels.

## R0 Fast Audit

```bash
python scripts/quick_check.py
```

R0 verifies required files, static file hashes, published FoRDy/FoRCy metrics, strict force-beneficial screening metrics, FLIQ false-safe audit metrics, decision-value false-safe metrics, failure-mode guardrail metrics, absence of local paths or legacy labels, and the reproducibility report.

## R1 Core Reproduction

```bash
python scripts/reproduce_core.py
```

R1 regenerates the FoRDy/FoRCy proxy-class validation tables, the independent predictive FoRDy validation outputs, the OpenSeesPy 3D solver-backed class-prediction outputs, the FLIQ liquefaction-oriented false-safe audit, the decision-value false-safe audit, the failure-mode guardrail audit, and the experimental figures from the redistributed mastersheets in `data/raw_redistributed/`.

The independent predictive layer also runs a grouped uncertainty audit. It resamples complete FoRDy study groups, not individual rows, and regenerates `grouped_metric_by_study.csv`, `grouped_bootstrap_uncertainty.csv`, `paired_group_baseline_comparison.csv`, and `grouped_uncertainty_summary.json`.

Reproduction scripts write regenerated artifacts only under `outputs/regenerated/`. The `outputs/published/` directory and validation snapshot files are treated as frozen package evidence and should remain hash-stable before and after R1/R2 reproduction.

## R2 Optional Full Reproduction

```bash
python scripts/reproduce_full.py
```

R2 calls R1 and reports that online metadata and historical field-data audits are optional. These optional checks are not required for the main manuscript claims.

## Expected Metrics

| Metric | Expected value |
| --- | ---: |
| FoRDy/FoRCy valid rows | 591 |
| Combined proxy-class accuracy | 0.8781725888324873 |
| Combined false-safe rate | 0.021996615905245348 |
| Combined safe precision | 0.6904761904761905 |
| Combined force-beneficial false-safe rate | 0.04710144927536232 |
| Decision-value proxy false-safe rate | 0.02 |
| Decision-value false-safe reduction vs optimistic force rule | 0.9409090909090909 |
| Predictive accuracy grouped bootstrap 95% CI | 0.410714285714 to 0.660606060606 |
| Predictive false-safe rate grouped bootstrap 95% CI | 0.00558659217877 to 0.14 |
| Predictive mixed-recall grouped bootstrap 95% CI | 0.767123287671 to 0.987951807229 |
| Solver-backed class-prediction rows | 194 |
| Solver-backed accuracy | 0.5103092783505154 |
| Solver-backed balanced accuracy | 0.5033766233766234 |
| Solver-backed false-safe rate | 0.05154639175257732 |
| Solver-backed mixed recall | 0.8701298701298701 |
| Solver-backed safe precision | 0.7619047619047619 |
| FLIQ usable structural rows | 241 |
| FLIQ pre-response guardrail false-safe rate | 0.08713692946058091 |
| Guardrail false-safe releases after low-margin rule | 0 |
| Guardrail abstention/escalation rate | 0.038461538461538464 |
| Theory identity maximum error | 8.881784197001252e-16 |

## Offline Guarantee

R0 and R1 do not require internet access. The principal experimental validation depends on files included in `data/raw_redistributed/` and on deterministic scripts included in `scripts/` and `validation/`.
