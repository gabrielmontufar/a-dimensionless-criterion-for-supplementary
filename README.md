# Demand-Polarity Map Reproducibility Package v40

This package reproduces the analytical checks, screening-evidence summaries, FoRDy/FoRCy experimental proxy-class validation, the FoRDy independent predictive screening layer, the OpenSeesPy 3D solver-backed class-prediction layer, the grouped uncertainty audit for the predictive layer, the FLIQ liquefaction-oriented false-safe audit, the decision-value false-safe audit, and the failure-mode guardrail audit used in the manuscript.

The main reproducibility path is fully offline. Online metadata refreshes and historical field-data audits are optional and are not required to verify the reported conclusions.

Reproduction is non-destructive: R1/R2 scripts write regenerated outputs only to `outputs/regenerated/`, leaving `outputs/published/` and validation snapshot files unchanged for hash verification.

## Official Commands

1. Fast audit:

```bash
python scripts/quick_check.py
```

Expected terminal marker:

```text
REPRODUCIBILITY_FAST_CHECK: PASS
```

2. Core reproduction:

```bash
python scripts/reproduce_core.py
```

Expected terminal marker:

```text
REPRODUCIBILITY_CORE: PASS
```

3. Optional full reproduction:

```bash
python scripts/reproduce_full.py
```

Expected terminal marker:

```text
REPRODUCIBILITY_FULL: PASS_WITH_OPTIONAL_WARNINGS
```

## Supported Claims

- Closed-form SSI Demand-Polarity Map and mixed-regime boundary.
- Secant spectral-slope consistency as a screening quantity.
- Experimental proxy class validation against FoRDy/FoRCy rocking-foundation databases.
- Independent predictive proxy screening on withheld FoRDy cases using only pre-response descriptors.
- OpenSeesPy 3D solver-backed predictive class validation on withheld FoRDy study groups.
- Grouped uncertainty and paired-baseline audit using complete FoRDy study groups as the resampling unit.
- External liquefaction-oriented false-safe audit using FLIQ settlement and rotation quantities.
- Risk-aware decision-value validation of false-safe reduction under asymmetric engineering loss.
- Failure-mode atlas and margin guardrails for safer screening releases.
- Garner Valley retained only as a validation-falsification and instrumentation-audit case.

## Claims Not Made

- Full nonlinear 3D response-history validation.
- Independent prediction of force histories.
- Completed site-specific SFSI validation.
- Successful Garner Valley predictive validation.

## Reproducibility Levels

| Level | Purpose | Required command | Internet |
| --- | --- | --- | --- |
| R0 | Fast audit of hashes, files, metrics, and claim boundaries | `python scripts/quick_check.py` | No |
| R1 | Regenerate core FoRDy/FoRCy tables, grouped uncertainty tables, and the experimental map figure | `python scripts/reproduce_core.py` | No |
| R2 | Run optional extended checks after the core workflow | `python scripts/reproduce_full.py` | Optional |

The package manifest is `MANIFEST_SHA256.csv`. Table and figure traceability is documented in `TABLE_FIGURE_TRACEABILITY.csv`. Claim boundaries are documented in `CLAIM_EVIDENCE_MATRIX.csv`.
