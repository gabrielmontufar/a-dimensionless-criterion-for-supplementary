# FLIQ False-Safe Audit

Purpose: external liquefaction-oriented false-safe audit for the SSI
Demand-Polarity screening framework.

This layer uses the public FLIQ foundation and ground performance in
liquefaction experiments dataset to test whether apparent acceleration-side
benefit can coincide with settlement or rotation penalties.

Inputs:

- `data/raw_redistributed/FLIQ/FLIQ_MainSpreadsheet_unformatted.csv`
- `data/raw_redistributed/FLIQ/FLIQ_ColumnDefinitions_unformatted.csv`
- `data/raw_redistributed/FLIQ/FLIQ_Soil_Properties_unformatted.csv`

Outputs:

- `fliq_false_safe_audit_rows.csv`
- `fliq_guardrail_threshold_sweep.csv`
- `fliq_false_safe_probe_summary.json`

Claim boundary:

This is an external false-safe audit, not nonlinear 3D response-history
validation and not an independent force-history prediction.

