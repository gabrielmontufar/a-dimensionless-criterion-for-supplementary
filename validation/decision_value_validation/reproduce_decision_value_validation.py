from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "regenerated" / "tables"
SUMMARY_OUT = ROOT / "outputs" / "regenerated" / "summaries"
FIG = ROOT / "outputs" / "regenerated" / "figures"
PUBLISHED = ROOT / "outputs" / "published" / "tables"

CLASSES = ["both_beneficial", "mixed", "both_detrimental"]


def safe_decision(label: str) -> str:
    return "safe" if label == "both_beneficial" else "flag"


def decision_metrics(df: pd.DataFrame, model: str, loss_ratio: float = 10.0) -> dict:
    sub = df[(df["model"] == model) & df["observed_class"].isin(CLASSES) & df["predicted_class"].isin(CLASSES)].copy()
    if sub.empty:
        return {
            "model": model,
            "n": 0,
            "safe_rate": "",
            "false_safe_rate": "",
            "missed_safe_rate": "",
            "utility_per_case": "",
            "net_value_vs_always_flag": "",
        }
    obs_safe = sub["observed_class"].map(safe_decision) == "safe"
    pred_safe = sub["predicted_class"].map(safe_decision) == "safe"
    true_safe = pred_safe & obs_safe
    false_safe = pred_safe & ~obs_safe
    false_alarm = ~pred_safe & obs_safe
    # Decision utility: false-safe is the dominant engineering harm.
    # Correctly allowing a truly safe case has value +1; unnecessary flagging costs -1.
    # Correctly flagging unsafe cases has value 0 because it is conservative baseline behavior.
    utility = true_safe.astype(float) - false_alarm.astype(float) - loss_ratio * false_safe.astype(float)
    always_flag_utility = -obs_safe.astype(float)
    return {
        "model": model,
        "n": int(len(sub)),
        "safe_rate": float(pred_safe.mean()),
        "false_safe_count": int(false_safe.sum()),
        "false_safe_rate": float(false_safe.mean()),
        "missed_safe_count": int(false_alarm.sum()),
        "missed_safe_rate": float(false_alarm.mean()),
        "true_safe_count": int(true_safe.sum()),
        "utility_per_case": float(utility.mean()),
        "always_flag_utility_per_case": float(always_flag_utility.mean()),
        "net_value_vs_always_flag": float(utility.mean() - always_flag_utility.mean()),
        "loss_ratio_false_safe_to_false_alarm": float(loss_ratio),
    }


def load_proxy_decisions() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED / "fordy_forcy_master_validation_table.csv")
    df = df[df["sensitivity_label"] == "base"].copy()
    rows = []
    for model, pred_col in [
        ("proposed_proxy_map", "predicted_class"),
        ("always_flag_mixed", None),
        ("optimistic_force_reduction_rule", None),
        ("spectral_window_rule", None),
    ]:
        temp = df.copy()
        temp["model"] = model
        if model == "always_flag_mixed":
            temp["predicted_class"] = "mixed"
        elif model == "optimistic_force_reduction_rule":
            temp["predicted_class"] = np.where(temp["R_V_proxy"] < 1.0, "both_beneficial", "both_detrimental")
        elif model == "spectral_window_rule":
            temp["predicted_class"] = np.where(
                (temp["n_s_proxy"] > -2.0) & (temp["n_s_proxy"] < 0.0),
                "mixed",
                np.where(temp["n_s_proxy"] <= -2.0, "both_beneficial", "both_detrimental"),
            )
        rows.append(temp[["dataset", "test_id", "event_no", "model", "observed_class", "predicted_class", "lambda_proxy", "R_V_proxy"]])
    return pd.concat(rows, ignore_index=True)


def load_predictive_decisions() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED / "predictive_validation_results.csv")
    df = df.rename(columns={"lambda_hat": "lambda_proxy"})
    df["dataset"] = "FoRDy_predictive"
    df["R_V_proxy"] = np.nan
    return df[["dataset", "test_id", "event_no", "model", "observed_class", "predicted_class", "lambda_proxy", "R_V_proxy"]].copy()


def polarity_margin(row: pd.Series) -> float:
    rv = float(row["R_V_proxy"])
    lam = float(row["lambda_proxy"])
    if not math.isfinite(rv) or not math.isfinite(lam) or rv <= 0 or lam <= 1:
        return 0.0
    lower = 1.0 / (lam * lam)
    return float(min(abs(math.log(rv / lower)), abs(math.log(rv / 1.0))))


def selective_abstention(proxy_df: pd.DataFrame, loss_ratio: float = 10.0) -> pd.DataFrame:
    base = proxy_df[proxy_df["model"] == "proposed_proxy_map"].copy()
    base["margin"] = base.apply(polarity_margin, axis=1)
    rows = []
    for coverage in np.linspace(0.50, 1.00, 11):
        n_keep = max(1, int(round(len(base) * coverage)))
        keep = base.sort_values("margin", ascending=False).head(n_keep).copy()
        flagged = base.drop(keep.index).copy()
        scored = keep.copy()
        # Abstained cases are treated as flagged, which is conservative and decision-relevant.
        if not flagged.empty:
            flagged["predicted_class"] = "mixed"
            scored = pd.concat([scored, flagged], ignore_index=True)
        metrics = decision_metrics(scored.assign(model="selective_proxy_map"), "selective_proxy_map", loss_ratio)
        rows.append(
            {
                "coverage_scored_by_map": float(coverage),
                "abstention_rate_flagged": float(1.0 - coverage),
                "false_safe_rate": metrics["false_safe_rate"],
                "missed_safe_rate": metrics["missed_safe_rate"],
                "safe_rate": metrics["safe_rate"],
                "utility_per_case": metrics["utility_per_case"],
                "net_value_vs_always_flag": metrics["net_value_vs_always_flag"],
            }
        )
    return pd.DataFrame(rows)


def loss_sweep(proxy_df: pd.DataFrame, predictive_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loss_ratio in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        for model in sorted(proxy_df["model"].unique()):
            item = decision_metrics(proxy_df, model, float(loss_ratio))
            item["layer"] = "proxy_FoRDy_FoRCy"
            rows.append(item)
        for model in sorted(predictive_df["model"].unique()):
            item = decision_metrics(predictive_df, model, float(loss_ratio))
            item["layer"] = "predictive_FoRDy"
            rows.append(item)
    return pd.DataFrame(rows)


def plot_loss_sweep(sweep: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    show = sweep[
        (
            (sweep["layer"] == "proxy_FoRDy_FoRCy")
            & (sweep["model"].isin(["proposed_proxy_map", "always_flag_mixed", "optimistic_force_reduction_rule"]))
        )
        | (
            (sweep["layer"] == "predictive_FoRDy")
            & (sweep["model"].isin(["proposed_predictive_centroid", "always_mixed", "majority_class"]))
        )
    ]
    for (layer, model), group in show.groupby(["layer", "model"]):
        ax.plot(
            group["loss_ratio_false_safe_to_false_alarm"],
            group["net_value_vs_always_flag"],
            marker="o",
            linewidth=1.4,
            label=f"{layer}: {model}",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("False-safe loss / false-alarm loss")
    ax.set_ylabel("Net decision value vs always flag")
    ax.set_title("Decision value under asymmetric engineering loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "loss_ratio_sweep.png")
    plt.close(fig)


def plot_selective(curve: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax1.plot(curve["coverage_scored_by_map"], curve["false_safe_rate"], marker="o", label="False-safe rate")
    ax1.plot(curve["coverage_scored_by_map"], curve["missed_safe_rate"], marker="s", label="Missed-safe rate")
    ax1.set_xlabel("Coverage scored by map")
    ax1.set_ylabel("Rate")
    ax2 = ax1.twinx()
    ax2.plot(curve["coverage_scored_by_map"], curve["net_value_vs_always_flag"], color="tab:green", marker="^", label="Net value")
    ax2.set_ylabel("Net value vs always flag")
    ax1.set_title("Selective screening: abstain/flag low-margin cases")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "selective_abstention_curve.png")
    plt.close(fig)


def write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_both(name: str, df: pd.DataFrame) -> None:
    write(OUT / name, df)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    proxy = load_proxy_decisions()
    predictive = load_predictive_decisions()

    proxy_summary = pd.DataFrame([decision_metrics(proxy, model, 10.0) for model in sorted(proxy["model"].unique())])
    pred_summary = pd.DataFrame([decision_metrics(predictive, model, 10.0) for model in sorted(predictive["model"].unique())])
    sweep = loss_sweep(proxy, predictive)
    selective = selective_abstention(proxy, 10.0)

    write_both("proxy_decision_value_summary.csv", proxy_summary)
    write_both("predictive_decision_value_summary.csv", pred_summary)
    write_both("loss_ratio_sweep.csv", sweep)
    write_both("selective_abstention_curve.csv", selective)
    plot_loss_sweep(sweep)
    plot_selective(selective)

    proposed_proxy = proxy_summary[proxy_summary["model"] == "proposed_proxy_map"].iloc[0].to_dict()
    optimistic = proxy_summary[proxy_summary["model"] == "optimistic_force_reduction_rule"].iloc[0].to_dict()
    always_flag_proxy = proxy_summary[proxy_summary["model"] == "always_flag_mixed"].iloc[0].to_dict()
    proposed_pred = pred_summary[pred_summary["model"] == "proposed_predictive_centroid"].iloc[0].to_dict()
    always_mixed_pred = pred_summary[pred_summary["model"] == "always_mixed"].iloc[0].to_dict()
    best_selective = selective.sort_values("net_value_vs_always_flag", ascending=False).iloc[0].to_dict()
    proxy_false_safe_reduction_vs_optimistic = 1.0 - (
        float(proposed_proxy["false_safe_rate"]) / max(float(optimistic["false_safe_rate"]), 1e-12)
    )
    pred_false_safe_reduction_vs_spectral = np.nan
    spectral_pred = pred_summary[pred_summary["model"] == "spectral_slope_only"]
    if not spectral_pred.empty:
        pred_false_safe_reduction_vs_spectral = 1.0 - (
            float(proposed_pred["false_safe_rate"]) / max(float(spectral_pred.iloc[0]["false_safe_rate"]), 1e-12)
        )
    sweep_proxy = sweep[(sweep["layer"] == "proxy_FoRDy_FoRCy") & (sweep["model"] == "proposed_proxy_map")]
    positive_loss_ratios = sweep_proxy[sweep_proxy["net_value_vs_always_flag"] > 0][
        "loss_ratio_false_safe_to_false_alarm"
    ].tolist()

    report = {
        "experiment": "decision_value_validation",
        "isolated_from_submission_workflow": False,
        "interpretation": "risk-aware decision validation of the screening criterion",
        "proxy_layer": {
            "tests": int(proposed_proxy["n"]),
            "proposed_false_safe_rate": float(proposed_proxy["false_safe_rate"]),
            "proposed_net_value_vs_always_flag": float(proposed_proxy["net_value_vs_always_flag"]),
            "always_flag_false_safe_rate": float(always_flag_proxy["false_safe_rate"]),
            "always_flag_net_value_vs_always_flag": float(always_flag_proxy["net_value_vs_always_flag"]),
            "optimistic_force_rule_false_safe_rate": float(optimistic["false_safe_rate"]),
            "optimistic_force_rule_net_value_vs_always_flag": float(optimistic["net_value_vs_always_flag"]),
            "false_safe_reduction_vs_optimistic_force_rule": float(proxy_false_safe_reduction_vs_optimistic),
            "positive_net_value_vs_always_flag_loss_ratios": positive_loss_ratios,
        },
        "predictive_layer": {
            "tests": int(proposed_pred["n"]),
            "proposed_false_safe_rate": float(proposed_pred["false_safe_rate"]),
            "proposed_net_value_vs_always_flag": float(proposed_pred["net_value_vs_always_flag"]),
            "always_mixed_net_value_vs_always_flag": float(always_mixed_pred["net_value_vs_always_flag"]),
            "false_safe_reduction_vs_spectral_slope_only": float(pred_false_safe_reduction_vs_spectral),
        },
        "selective_screening": {
            "best_coverage_scored_by_map": float(best_selective["coverage_scored_by_map"]),
            "best_false_safe_rate": float(best_selective["false_safe_rate"]),
            "best_net_value_vs_always_flag": float(best_selective["net_value_vs_always_flag"]),
        },
        "decision_interpretation": "The proxy map strongly rejects unsafe optimistic force-reduction screening, but it does not dominate the fully conservative always-flag policy when false-safe loss is set very high.",
        "claim_boundary": "This validates risk reduction and decision tradeoffs under asymmetric loss, not nonlinear 3D response prediction.",
    }
    (SUMMARY_OUT / "decision_value_validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
