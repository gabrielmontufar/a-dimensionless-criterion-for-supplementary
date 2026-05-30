# Demand-polarity map external literature consistency audit

Date: 2026-05-30

Target package folder:
`demand_polarity_map_v40_submission_reproducible/supplementary_material/literature_external_consistency_audit_20260530/`

## Purpose

This experiment tests whether external scientific articles can strengthen the `Validacion/comparacion` score for the demand-polarity map without overstating the evidence. The answer is yes, but only if the evidence is framed as an external consistency benchmark. It should not be presented as nonlinear 3D site-specific validation or independent response-history prediction.

## Evidence classes

| Class | Meaning | Editorial value for the demand-polarity map |
|---|---|---|
| Quantitative external range check | Published numerical ratios or coefficient ranges can be compared with the demand-polarity map's force/displacement polarity logic. | Strongest available improvement short of new experiments. |
| Quantitative consistency mapping | Published period and force-ratio relationships can be transformed into lambda, eta or RV-like quantities. | Strong, if table/figure extraction is reproducible. |
| Mechanism consistency | External papers independently report the same mechanism: period shift, force reduction, drift/displacement amplification or detrimental resonance. | Useful, but weaker than case-level numerical comparison. |
| Field mechanics support | Field tests support stiffness, damping, impedance or instrumentation assumptions. | Important for plausibility, but not class-level validation by itself. |

## Main finding

The external literature can honestly improve the validation/comparison score from **13.5/15** to approximately **14.0/15** immediately, because the current article already has FoRDy/FoRCy proxy validation, FLIQ false-safe audit, Garner falsification, decision-value audit and guardrails. The literature layer adds independent agreement that:

- SSI can reduce force/base-shear demand while increasing drift or displacement-sensitive demand.
- Period/frequency shift is a control variable, not an automatic benefit.
- Soft-site and long-period conditions make the displacement penalty more likely.
- Foundation stiffness and damping are frequency dependent and field-measurable, supporting the reduced-model inputs but not replacing validation.

A score above **14.0-14.25/15** would require one more step: extracting case-level numerical points from Zhang and Far (2024) and Yang et al. (2024), then producing a reproducible table/figure with `published quantity`, `article-116 class`, `agreement`, and `reason for disagreement`.

## Strongest sources

### E01. Zhang and Far (2024)

Use: primary quantitative literature benchmark.

Why it helps: the study uses a finite-element soil-foundation-structure model verified by shaking-table tests, then evaluates 72 rigid-base cases and 720 flexible-base cases. It reports normalized base-shear and drift ratios and explicitly distinguishes beneficial and detrimental SSI effects. This is the closest external support for the the demand-polarity map demand-polarity premise.

How to use it:

- Extract natural periods from their tables to compute `lambda = T_flexible/T_fixed`.
- Extract base-shear ratio `V_fle/V_fix` as an external `RV` analog.
- Treat inter-storey drift ratio as the displacement-penalty analog.
- Classify each extractable case as force-beneficial, displacement-detrimental, both-beneficial, or both-detrimental.

Boundary: this is consistency mapping, not independent prediction by the demand-polarity map unless the demand-polarity map's rule is applied before seeing the response ratios.

### E02. Yang et al. (2024)

Use: quantitative external range check.

Why it helps: the paper reports that, under far-field long-period motions, the base-shear SSI coefficient is generally below 1, while top displacement and maximum inter-story drift coefficients are above 1. The reported ranges are directly aligned with the demand-polarity map's mixed polarity: `RV < 1` with displacement-sensitive response `> 1`.

How to use it:

- Add a range-consistency table:
  - published shear coefficient: 0.5 to 1;
  - published displacement/drift coefficient: greater than 1 and less than 3;
  - article-116 interpretation: mixed force-beneficial/displacement-detrimental domain.
- Include the caveat that bimodal long-period spectra can also produce shear amplification, which supports the article's spectral-slope warning.

Boundary: this supports the mechanism and range, not exact prediction accuracy.

### E03. Tao, Fu and Li (2024)

Use: mechanism and spectral-shift support.

Why it helps: the study specifically examines detrimental SSI due to shifts in system frequency, including time-domain building/site cases. This supports the demand-polarity map's central idea that period elongation must be interpreted through spectral position and cannot be treated as automatically beneficial.

How to use it:

- Use it in the validation discussion as an independent warning that frequency shift can move a structure into a worse spectral region.
- Pair it with the demand-polarity map's secant-slope formulation.

Boundary: not enough as a case-level validation table unless specific response ratios are extracted.

### E04. Mylonakis and Gazetas (2000)

Use: foundational theoretical and recorded-motion support.

Why it helps: it is a canonical source for the beneficial/detrimental SSI framing and supports the need to avoid design-code overgeneralization.

Boundary: this is not new validation evidence; it strengthens the intellectual framing and claim discipline.

### E05. Tileylioglu, Stewart and Nigbor (2011)

Use: field-mechanics support for foundation impedance and Garner Valley relevance.

Why it helps: it provides field forced-vibration evidence that stiffness and damping are frequency dependent and that measured impedance can be compared to numerical models at Garner Valley.

Boundary: it supports foundation mechanics and instrumentation provenance, not the demand-polarity map's class predictions.

## Recommended manuscript insertion

Add this as a short subsection after the existing FLIQ audit, or as a supplement subsection if the manuscript is already long.

Suggested title:

`8.13 External literature consistency benchmark`

Suggested text:

> As an additional literature-based consistency benchmark, recent and classical SSI studies were screened for independently reported combinations of force-side and displacement-side response. Zhang and Far (2024) provide the strongest quantitative mapping candidate because their finite-element soil-foundation-structure study was verified against shaking-table tests and reports normalized base-shear and inter-storey drift ratios for a large high-rise parametric matrix. Their reported mechanism, in which foundation rotation can reduce base shear while increasing inter-storey drift, is consistent with the mixed demand-polarity region of the present map. Yang et al. (2024) provide an independent range check under far-field long-period motions: reported SSI base-shear coefficients are generally below unity, while displacement and drift coefficients exceed unity, with the caveat that bimodal spectra can still produce shear amplification. Tao et al. (2024) further supports the spectral-shift interpretation by showing that SSI can be detrimental when period/frequency shift moves the system into an unfavorable demand region. These external studies are used only as literature-based consistency and range checks; they do not constitute independent nonlinear 3D response-history validation of the proposed screening criterion.

## Score impact

| Scenario | Validation/comparison score | Reason |
|---|---:|---|
| Current v27 package | 13.5/15 | Strong class-level screening evidence, but not nonlinear 3D validation or independent response-history prediction. |
| Add literature audit as text/table only | 14.0/15 | Adds independent external mechanism/range support without changing claim boundary. |
| Add reproducible digitized case-level extraction from Zhang/Far and Yang | 14.25/15 | Converts literature support into traceable numerical comparison. |
| Claim full validation based on literature only | Do not do this | Would be overclaiming and likely harms reviewer trust. |

## Next reproducible step

Create a small table named `literature_case_mapping.csv` with these columns:

`source_id, case_id, T_fixed, T_flexible, lambda, eta_from_lambda, published_force_ratio, published_displacement_or_drift_ratio, demand_polarity_map_class, published_class, agreement, extraction_method, source_table_or_figure`

The first extraction target should be Zhang and Far (2024) because their natural-period tables are already available in text form and their normalized response figures/tables are the closest to the the demand-polarity map variables.
