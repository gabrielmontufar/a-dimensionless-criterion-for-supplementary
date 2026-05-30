from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLES = ROOT / "outputs" / "regenerated" / "tables"
OUT_SUMMARIES = ROOT / "outputs" / "regenerated" / "summaries"

RESULTS = OUT_TABLES / "predictive_validation_results.csv"

CLASSES = ["both_beneficial", "mixed", "both_detrimental"]
PROPOSED = "proposed_predictive_centroid"
BOOTSTRAP_REPS = 10000
SEED = 20260530


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_block(rows: list[dict[str, str]]) -> dict[str, float | int]:
    usable = [r for r in rows if r["observed_class"] in CLASSES and r["predicted_class"] in CLASSES]
    if not usable:
        return {
            "tests_used": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "false_safe_rate": float("nan"),
            "mixed_recall": float("nan"),
            "safe_precision": float("nan"),
        }
    correct = sum(r["observed_class"] == r["predicted_class"] for r in usable)
    recalls = []
    for klass in CLASSES:
        den = sum(r["observed_class"] == klass for r in usable)
        if den:
            num = sum(r["observed_class"] == klass and r["predicted_class"] == klass for r in usable)
            recalls.append(num / den)
    false_safe = sum(
        r["predicted_class"] == "both_beneficial" and r["observed_class"] != "both_beneficial" for r in usable
    ) / len(usable)
    mixed_den = sum(r["observed_class"] == "mixed" for r in usable)
    mixed_recall = (
        sum(r["observed_class"] == "mixed" and r["predicted_class"] == "mixed" for r in usable) / mixed_den
        if mixed_den
        else float("nan")
    )
    safe_den = sum(r["predicted_class"] == "both_beneficial" for r in usable)
    safe_precision = (
        sum(
            r["predicted_class"] == "both_beneficial" and r["observed_class"] == "both_beneficial" for r in usable
        )
        / safe_den
        if safe_den
        else 1.0
    )
    return {
        "tests_used": len(usable),
        "accuracy": correct / len(usable),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "false_safe_rate": false_safe,
        "mixed_recall": mixed_recall,
        "safe_precision": safe_precision,
    }


def percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1 - weight) + clean[hi] * weight


def format_float(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def grouped_bootstrap(rows: list[dict[str, str]], model: str, groups: list[str]) -> list[dict[str, str]]:
    rng = random.Random(SEED + sum(ord(c) for c in model))
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["model"] == model:
            by_group[row["fold"]].append(row)

    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPS):
        selected = [rng.choice(groups) for _ in groups]
        sample_rows: list[dict[str, str]] = []
        for group in selected:
            sample_rows.extend(by_group[group])
        metrics = metric_block(sample_rows)
        for metric in ["accuracy", "balanced_accuracy", "false_safe_rate", "mixed_recall", "safe_precision"]:
            value = metrics[metric]
            if isinstance(value, float) and math.isfinite(value):
                samples[metric].append(value)

    observed = metric_block([r for r in rows if r["model"] == model])
    out = []
    for metric in ["accuracy", "balanced_accuracy", "false_safe_rate", "mixed_recall", "safe_precision"]:
        vals = samples[metric]
        out.append(
            {
                "model": model,
                "metric": metric,
                "estimate": format_float(float(observed[metric])),
                "cluster_bootstrap_ci95_low": format_float(percentile(vals, 0.025)),
                "cluster_bootstrap_ci95_high": format_float(percentile(vals, 0.975)),
                "bootstrap_reps": str(BOOTSTRAP_REPS),
                "resampling_unit": "FoRDy study_group",
            }
        )
    return out


def paired_group_comparisons(group_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_model_group = {(r["model"], r["fold"]): r for r in group_rows}
    groups = sorted({r["fold"] for r in group_rows})
    models = sorted({r["model"] for r in group_rows if r["model"] != PROPOSED})
    out = []
    for model in models:
        diffs = []
        false_safe_diffs = []
        for group in groups:
            proposed = by_model_group[(PROPOSED, group)]
            baseline = by_model_group[(model, group)]
            diffs.append(float(proposed["accuracy"]) - float(baseline["accuracy"]))
            false_safe_diffs.append(float(proposed["false_safe_rate"]) - float(baseline["false_safe_rate"]))
        out.append(
            {
                "comparison": f"{PROPOSED}_minus_{model}",
                "groups": str(len(groups)),
                "mean_accuracy_difference": format_float(sum(diffs) / len(diffs)),
                "median_accuracy_difference": format_float(percentile(diffs, 0.5)),
                "accuracy_wins": str(sum(d > 0 for d in diffs)),
                "accuracy_ties": str(sum(d == 0 for d in diffs)),
                "accuracy_losses": str(sum(d < 0 for d in diffs)),
                "mean_false_safe_rate_difference": format_float(sum(false_safe_diffs) / len(false_safe_diffs)),
                "false_safe_improvements": str(sum(d < 0 for d in false_safe_diffs)),
                "false_safe_ties": str(sum(d == 0 for d in false_safe_diffs)),
                "false_safe_worse": str(sum(d > 0 for d in false_safe_diffs)),
            }
        )
    return out


def main() -> None:
    rows = read_rows(RESULTS)
    groups = sorted({r["fold"] for r in rows})
    models = sorted({r["model"] for r in rows})

    group_metric_rows = []
    for model in models:
        for group in groups:
            sub = [r for r in rows if r["model"] == model and r["fold"] == group]
            metrics = metric_block(sub)
            group_metric_rows.append(
                {
                    "model": model,
                    "fold": group,
                    **{key: format_float(value) for key, value in metrics.items()},
                }
            )

    uncertainty_rows = []
    for model in models:
        uncertainty_rows.extend(grouped_bootstrap(rows, model, groups))

    comparison_rows = paired_group_comparisons(group_metric_rows)

    summary = {
        "package_version": "demand_polarity_map_v40_submission_reproducible",
        "audit_type": "grouped uncertainty and paired baseline audit",
        "resampling_unit": "FoRDy study_group",
        "groups": len(groups),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": SEED,
        "claim_boundary": "Uncertainty audit for pre-response predictive class screening only; not response-history prediction or nonlinear 3D validation.",
    }
    proposed_uncertainty = [r for r in uncertainty_rows if r["model"] == PROPOSED]
    for row in proposed_uncertainty:
        metric = row["metric"]
        summary[f"proposed_{metric}"] = float(row["estimate"])
        summary[f"proposed_{metric}_cluster_ci95_low"] = float(row["cluster_bootstrap_ci95_low"])
        summary[f"proposed_{metric}_cluster_ci95_high"] = float(row["cluster_bootstrap_ci95_high"])

    table_specs = [
        (
            "grouped_metric_by_study.csv",
            group_metric_rows,
            ["model", "fold", "tests_used", "accuracy", "balanced_accuracy", "false_safe_rate", "mixed_recall", "safe_precision"],
        ),
        (
            "grouped_bootstrap_uncertainty.csv",
            uncertainty_rows,
            [
                "model",
                "metric",
                "estimate",
                "cluster_bootstrap_ci95_low",
                "cluster_bootstrap_ci95_high",
                "bootstrap_reps",
                "resampling_unit",
            ],
        ),
        (
            "paired_group_baseline_comparison.csv",
            comparison_rows,
            [
                "comparison",
                "groups",
                "mean_accuracy_difference",
                "median_accuracy_difference",
                "accuracy_wins",
                "accuracy_ties",
                "accuracy_losses",
                "mean_false_safe_rate_difference",
                "false_safe_improvements",
                "false_safe_ties",
                "false_safe_worse",
            ],
        ),
    ]
    for name, table_rows, fields in table_specs:
        write_csv(OUT_TABLES / name, table_rows, fields)

    for target in [OUT_SUMMARIES / "grouped_uncertainty_summary.json"]:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
