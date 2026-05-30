# Independent Predictive Validation

This module tests whether FoRDy polarity classes can be predicted using pre-response descriptors only.

Official reproduction is included in R1 through:

```bash
python scripts/reproduce_core.py
```

The module withholds observed force, drift, rotation and settlement until scoring. It reports a predictive screening layer only. It does not claim nonlinear 3D field validation or independent response-history prediction.

The additional grouped uncertainty audit is reproduced with:

```bash
python validation/independent_predictive_validation/reproduce_group_uncertainty_audit.py
```

It writes `grouped_metric_by_study.csv`, `grouped_bootstrap_uncertainty.csv`, `paired_group_baseline_comparison.csv`, and `grouped_uncertainty_summary.json`.

## Independence Control

The independent unit is the FoRDy study/project group, not an individual table row. The leave-study-out split is therefore the operational leakage control: predictor descriptors and frozen class rules are evaluated against held-out study groups, while observed response quantities and observed classes remain unavailable until scoring. This keeps the claim bounded to pre-response class screening and avoids interpreting grouped experimental records as independent row-wise samples.

Cluster bootstrap intervals also use the FoRDy study group as the resampling unit. The reported confidence ranges are therefore deliberately wider than row-wise intervals and are used as a robustness audit, not as a claim that the predictor is calibrated for response-history prediction.
