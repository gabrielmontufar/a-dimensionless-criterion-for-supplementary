# Failure-Mode Atlas and Guardrail Discovery

This is a secondary validation layer in the official v40 reproducibility
workflow.

Purpose:

- Map where the SSI Demand-Polarity Map fails in FoRDy/FoRCy proxy validation.
- Identify false-safe, false-mixed, false-detrimental, robust-correct and
  marginal-correct regions.
- Derive simple empirical guardrails that tell the screening tool when to
  abstain/escalate instead of releasing a case as beneficial.
- Measure the false-safe reduction and the abstention/escalation cost.

Official command:

```bash
python validation/failure_mode_guardrails/reproduce_failure_mode_guardrails.py
```

Outputs:

- `outputs/regenerated/tables/failure_mode_cases.csv`
- `outputs/regenerated/tables/failure_mode_summary.csv`
- `outputs/regenerated/tables/guardrail_candidate_summary.csv`
- `outputs/regenerated/tables/guardrail_application_summary.csv`
- `outputs/regenerated/summaries/failure_mode_guardrail_summary.json`
- `outputs/regenerated/figures/failure_mode_map_lambda_rv.png`
- `outputs/regenerated/figures/guardrail_tradeoff.png`

The same outputs are mirrored in this folder for inspection.

Claim boundary:

This experiment explores empirical guardrails for safer screening. It does not
claim nonlinear 3D SSI validation or response-history prediction.
