# Decision-Value Validation

This is a secondary validation layer in the official v40 reproducibility
workflow.

Purpose:

- Test a more disruptive validation framing: the SSI Demand-Polarity Map is
  evaluated as a risk-aware decision tool, not as a response-history simulator.
- Quantify whether the criterion reduces dangerous "false-safe" decisions under
  asymmetric engineering loss.
- Evaluate selective prediction: the tool may flag/abstain on low-margin cases
  instead of forcing a potentially unsafe beneficial decision.

Official command for this experiment:

```bash
python validation/decision_value_validation/reproduce_decision_value_validation.py
```

Outputs:

- `outputs/regenerated/tables/proxy_decision_value_summary.csv`
- `outputs/regenerated/tables/predictive_decision_value_summary.csv`
- `outputs/regenerated/tables/loss_ratio_sweep.csv`
- `outputs/regenerated/tables/selective_abstention_curve.csv`
- `outputs/regenerated/summaries/decision_value_validation_summary.json`
- `outputs/regenerated/figures/loss_ratio_sweep.png`
- `outputs/regenerated/figures/selective_abstention_curve.png`

The same files are mirrored inside this folder for inspection.

Claim boundary:

This experiment tests whether the criterion has decision value as a
pre-analysis screening layer. It does not claim nonlinear 3D SSI validation or
independent response-history prediction.
