from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_redistributed"
OUT_TABLES = ROOT / "outputs" / "regenerated" / "tables"
OUT_SUMMARIES = ROOT / "outputs" / "regenerated" / "summaries"
OUT_FIGURES = ROOT / "outputs" / "regenerated" / "figures"
SENTINELS = {7777.0, 8888.0, 9999.0}
CLASSES = ["both_beneficial", "mixed", "both_detrimental"]
FEATURES = ["lambda_hat", "n_s_hat", "PGA_B", "PGV_B", "Ia_B"]
SAFE_MARGIN_THRESHOLD = 0.30
DETRIMENTAL_MARGIN_THRESHOLD = 1.30


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(SENTINELS))


def demand_class(force_ratio: float, displacement_ratio: float) -> str:
    if not math.isfinite(force_ratio) or not math.isfinite(displacement_ratio):
        return "uncertain"
    if force_ratio < 1.0 and displacement_ratio <= 1.0:
        return "both_beneficial"
    if force_ratio < 1.0 and displacement_ratio > 1.0:
        return "mixed"
    if force_ratio >= 1.0 and displacement_ratio > 1.0:
        return "both_detrimental"
    return "force_detrimental_displacement_beneficial"


def spectral_acceleration_proxy(period: float, pga: float, tp: float, tm: float) -> float:
    if not all(math.isfinite(x) for x in [period, pga, tp, tm]) or period <= 0 or pga <= 0 or tp <= 0 or tm <= 0:
        return float("nan")
    if period <= tp:
        return pga
    if period <= tm:
        return pga * tp / period
    return pga * tp * tm / (period * period)


def secant_spectral_slope(t0: float, lam: float, pga: float, tp: float, tm: float) -> float:
    s0 = spectral_acceleration_proxy(t0, pga, tp, tm)
    s1 = spectral_acceleration_proxy(t0 * lam, pga, tp, tm)
    if not all(math.isfinite(x) for x in [s0, s1, lam]) or s0 <= 0 or s1 <= 0 or lam <= 1:
        return float("nan")
    return math.log(s1 / s0) / math.log(lam)


def load_fordy_observed() -> pd.DataFrame:
    df = pd.read_csv(RAW / "FoRDy_Mastersheet_v1.0.0_20180821_2322.csv", header=2, encoding_errors="replace")
    df = df[numeric(df["Event No"]).notna()].copy()
    rv = numeric(df["maxM"])
    displacement_severity = pd.concat(
        [
            numeric(df["pkDR"]).abs() / 1.0,
            numeric(df["pkRot"]).abs() / 0.01,
            numeric(df["resSet"]).abs() / 1.0,
        ],
        axis=1,
    ).max(axis=1)
    lambda_hat = np.sqrt(numeric(df["A/A_ca"]).clip(lower=1.01, upper=100.0))
    rows: list[dict] = []
    for idx, rec in df.iterrows():
        lam = float(lambda_hat.loc[idx]) if pd.notna(lambda_hat.loc[idx]) else float("nan")
        n_s = secant_spectral_slope(
            float(numeric(pd.Series([rec["T_0"]])).iloc[0]),
            lam,
            float(numeric(pd.Series([rec["PGA_B"]])).iloc[0]),
            float(numeric(pd.Series([rec["Tp_B"]])).iloc[0]),
            float(numeric(pd.Series([rec["Tm_B"]])).iloc[0]),
        )
        observed = demand_class(
            float(rv.loc[idx]) if pd.notna(rv.loc[idx]) else float("nan"),
            float(displacement_severity.loc[idx]) if pd.notna(displacement_severity.loc[idx]) else float("nan"),
        )
        if observed not in CLASSES:
            continue
        rows.append(
            {
                "dataset": "FoRDy",
                "study_group": str(rec["Project Title"]),
                "test_id": str(rec["Experiment or Case ID"]),
                "event_no": int(float(rec["Event No"])),
                "lambda_hat": lam,
                "n_s_hat": n_s,
                "PGA_B": float(numeric(pd.Series([rec["PGA_B"]])).iloc[0]),
                "PGV_B": float(numeric(pd.Series([rec["PGV_B"]])).iloc[0]),
                "Ia_B": float(numeric(pd.Series([rec["Ia_B"]])).iloc[0]),
                "observed_class": observed,
                "leakage_control": "observed force, rotation, drift and settlement withheld until scoring",
            }
        )
    return pd.DataFrame(rows)


def centroid_predict(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    med = train[FEATURES].median(numeric_only=True)
    x_train = train[FEATURES].fillna(med).astype(float)
    x_test = test[FEATURES].fillna(med).astype(float)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1).fillna(1)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    centroids = {
        klass: x_train[train["observed_class"].values == klass].mean().values
        for klass in CLASSES
        if len(x_train[train["observed_class"].values == klass]) > 0
    }
    predictions: list[str] = []
    for row in x_test.values:
        distances = sorted((float(np.linalg.norm(row - centroid)), klass) for klass, centroid in centroids.items())
        if not distances:
            predictions.append("uncertain")
            continue
        predicted = distances[0][1]
        margin = distances[1][0] - distances[0][0] if len(distances) > 1 else 999.0
        if predicted == "both_beneficial" and margin < SAFE_MARGIN_THRESHOLD:
            predicted = "mixed"
        if predicted == "both_detrimental" and margin < DETRIMENTAL_MARGIN_THRESHOLD:
            predicted = "mixed"
        predictions.append(predicted)
    return predictions


def baseline_predictions(name: str, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    if name == "majority_class":
        majority = Counter(train["observed_class"]).most_common(1)[0][0]
        return [majority] * len(test)
    if name == "always_mixed":
        return ["mixed"] * len(test)
    if name == "spectral_slope_only":
        out = []
        for value in test["n_s_hat"]:
            if not math.isfinite(float(value)):
                out.append("uncertain")
            elif value < -1.2:
                out.append("both_beneficial")
            elif value < 0.2:
                out.append("mixed")
            else:
                out.append("both_detrimental")
        return out
    if name == "flexibility_only":
        out = []
        for value in test["lambda_hat"]:
            if not math.isfinite(float(value)):
                out.append("uncertain")
            elif value < 1.6:
                out.append("both_beneficial")
            elif value < 3.5:
                out.append("mixed")
            else:
                out.append("both_detrimental")
        return out
    raise ValueError(name)


def metric_block(rows: pd.DataFrame, model: str) -> dict:
    sub = rows[(rows["model"] == model) & rows["predicted_class"].isin(CLASSES) & rows["observed_class"].isin(CLASSES)]
    if sub.empty:
        return {"model": model, "tests_used": 0, "accuracy": "", "balanced_accuracy": "", "false_safe_rate": ""}
    accuracy = float((sub["predicted_class"] == sub["observed_class"]).mean())
    recalls = []
    for klass in CLASSES:
        den = int((sub["observed_class"] == klass).sum())
        if den:
            recalls.append(float(((sub["predicted_class"] == klass) & (sub["observed_class"] == klass)).sum() / den))
    false_safe = float(((sub["predicted_class"] == "both_beneficial") & (sub["observed_class"] != "both_beneficial")).mean())
    mixed_recall = float(
        ((sub["predicted_class"] == "mixed") & (sub["observed_class"] == "mixed")).sum()
        / max(int((sub["observed_class"] == "mixed").sum()), 1)
    )
    safe_precision = float(
        ((sub["predicted_class"] == "both_beneficial") & (sub["observed_class"] == "both_beneficial")).sum()
        / max(int((sub["predicted_class"] == "both_beneficial").sum()), 1)
    )
    return {
        "model": model,
        "tests_used": int(len(sub)),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "false_safe_rate": false_safe,
        "mixed_recall": mixed_recall,
        "safe_precision": safe_precision,
        "status": "pass_screening_safety_partial_accuracy"
        if model == "proposed_predictive_centroid" and false_safe < 0.10 and mixed_recall > 0.80
        else "baseline_or_diagnostic",
    }


def run_leave_study_out(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = [g for g in sorted(data["study_group"].dropna().unique()) if not str(g).startswith("The title")]
    for group in groups:
        test = data[data["study_group"] == group].copy()
        train = data[data["study_group"] != group].copy()
        model_predictions = {
            "proposed_predictive_centroid": centroid_predict(train, test),
            "majority_class": baseline_predictions("majority_class", train, test),
            "always_mixed": baseline_predictions("always_mixed", train, test),
            "spectral_slope_only": baseline_predictions("spectral_slope_only", train, test),
            "flexibility_only": baseline_predictions("flexibility_only", train, test),
        }
        for model, predictions in model_predictions.items():
            for (_, rec), pred in zip(test.iterrows(), predictions):
                rows.append(
                    {
                        "model": model,
                        "fold": group,
                        "test_id": rec["test_id"],
                        "event_no": rec["event_no"],
                        "lambda_hat": rec["lambda_hat"],
                        "n_s_hat": rec["n_s_hat"],
                        "PGA_B": rec["PGA_B"],
                        "PGV_B": rec["PGV_B"],
                        "Ia_B": rec["Ia_B"],
                        "observed_class": rec["observed_class"],
                        "predicted_class": pred,
                        "correct": str(pred == rec["observed_class"]).lower(),
                    }
                )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(confusion: pd.DataFrame) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    sub = confusion[confusion["model"] == "proposed_predictive_centroid"]
    matrix = sub.pivot(index="observed_class", columns="predicted_class", values="count").reindex(index=CLASSES, columns=CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title("Independent Predictive Validation Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "predictive_confusion_matrix.png")
    plt.close(fig)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARIES.mkdir(parents=True, exist_ok=True)
    data = load_fordy_observed()
    results = run_leave_study_out(data)
    results.to_csv(OUT_TABLES / "predictive_validation_results.csv", index=False)

    metrics = [metric_block(results, model) for model in sorted(results["model"].unique())]
    pd.DataFrame(metrics).to_csv(OUT_TABLES / "baseline_comparison.csv", index=False)

    confusion_rows = []
    for model, sub_model in results.groupby("model"):
        table = pd.crosstab(sub_model["observed_class"], sub_model["predicted_class"])
        for observed in table.index:
            for predicted in table.columns:
                confusion_rows.append(
                    {
                        "model": model,
                        "observed_class": observed,
                        "predicted_class": predicted,
                        "count": int(table.loc[observed, predicted]),
                    }
                )
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(OUT_TABLES / "predictive_validation_confusion_matrix.csv", index=False)

    false_safe = results[
        (results["model"] == "proposed_predictive_centroid")
        & (results["predicted_class"] == "both_beneficial")
        & (results["observed_class"] != "both_beneficial")
    ]
    false_safe.to_csv(OUT_TABLES / "false_safe_audit_predictive.csv", index=False)

    fold_rows = []
    for fold in sorted(results["fold"].unique()):
        sub = results[(results["model"] == "proposed_predictive_centroid") & (results["fold"] == fold)]
        fold_rows.append({"fold": fold, **metric_block(sub, "proposed_predictive_centroid")})
    pd.DataFrame(fold_rows).to_csv(OUT_TABLES / "leave_study_out_summary.csv", index=False)

    predictor_inputs_allowed = [
        {"input": item, "status": "allowed", "reason": "available before response scoring"}
        for item in ["lambda_hat", "n_s_hat", "PGA_B", "PGV_B", "Ia_B", "study_group"]
    ]
    predictor_inputs_forbidden = [
        {"input": item, "status": "forbidden", "reason": "observed response leakage"}
        for item in ["R_V observed", "maxM", "pkRot", "pkDR", "resSet", "observed_class"]
    ]
    write_csv(OUT_TABLES / "predictor_inputs_allowed.csv", predictor_inputs_allowed, ["input", "status", "reason"])
    write_csv(OUT_TABLES / "predictor_inputs_forbidden.csv", predictor_inputs_forbidden, ["input", "status", "reason"])
    write_csv(
        OUT_TABLES / "observed_class_rules.csv",
        [
            {"rule": "both_beneficial", "definition": "observed force ratio < 1 and measured displacement severity <= 1"},
            {"rule": "mixed", "definition": "observed force ratio < 1 and measured displacement severity > 1"},
            {"rule": "both_detrimental", "definition": "observed force ratio >= 1 and measured displacement severity > 1"},
        ],
        ["rule", "definition"],
    )
    write_csv(
        OUT_TABLES / "frozen_predictor_parameters.csv",
        [
            {"parameter": "features", "value": ";".join(FEATURES)},
            {"parameter": "safe_margin_threshold", "value": SAFE_MARGIN_THRESHOLD},
            {"parameter": "detrimental_margin_threshold", "value": DETRIMENTAL_MARGIN_THRESHOLD},
            {"parameter": "split", "value": "leave Project Title out"},
        ],
        ["parameter", "value"],
    )
    study_split = data[["study_group", "test_id", "event_no"]].copy()
    study_split.to_csv(OUT_TABLES / "study_group_split.csv", index=False)

    plot_confusion(confusion)
    proposed = next(row for row in metrics if row["model"] == "proposed_predictive_centroid")
    majority = next(row for row in metrics if row["model"] == "majority_class")
    summary = {
        "package_version": "demand_polarity_map_v40_submission_reproducible",
        "validation_type": "independent predictive proxy validation",
        "leakage_control": "FoRDy observed force, drift, rotation and settlement are withheld until scoring",
        "tests_used": proposed["tests_used"],
        "proposed_accuracy": proposed["accuracy"],
        "proposed_balanced_accuracy": proposed["balanced_accuracy"],
        "proposed_false_safe_rate": proposed["false_safe_rate"],
        "proposed_mixed_recall": proposed["mixed_recall"],
        "proposed_safe_precision": proposed["safe_precision"],
        "majority_accuracy": majority["accuracy"],
        "claim_boundary": "Predictive class screening layer only; not nonlinear 3D field validation and not independent response-history prediction.",
    }
    (OUT_SUMMARIES / "predictive_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
