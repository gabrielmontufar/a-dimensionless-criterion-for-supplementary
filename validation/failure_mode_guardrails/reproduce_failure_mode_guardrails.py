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


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED / "fordy_forcy_master_validation_table.csv")
    df = df[(df["sensitivity_label"] == "base") & (df["observed_class"].isin(CLASSES))].copy()
    for col in [
        "lambda_proxy",
        "eta_proxy",
        "n_s_proxy",
        "R_V_proxy",
        "R_delta_total_predicted",
        "measured_displacement_severity",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["map_margin"] = [boundary_margin(rv, lam) for rv, lam in zip(df["R_V_proxy"], df["lambda_proxy"])]
    df["log_lambda"] = np.log(df["lambda_proxy"].clip(lower=1e-12))
    df["log_RV"] = np.log(df["R_V_proxy"].clip(lower=1e-12))
    df["log_Rdelta_pred"] = np.log(df["R_delta_total_predicted"].clip(lower=1e-12))
    df["failure_mode"] = df.apply(failure_mode, axis=1)
    df["release_by_map"] = df["predicted_class"].eq("both_beneficial")
    df["false_safe"] = df["release_by_map"] & df["observed_class"].ne("both_beneficial")
    return df.reset_index(drop=True)


def boundary_margin(rv: float, lam: float) -> float:
    if not math.isfinite(rv) or not math.isfinite(lam) or rv <= 0 or lam <= 1:
        return 0.0
    lower = 1.0 / (lam * lam)
    return float(min(abs(math.log(rv / lower)), abs(math.log(rv / 1.0))))


def failure_mode(row: pd.Series) -> str:
    obs = row["observed_class"]
    pred = row["predicted_class"]
    if pred == obs:
        return "correct_robust" if row.get("robust_marginal") == "robust" else "correct_marginal"
    if pred == "both_beneficial" and obs != "both_beneficial":
        return "false_safe"
    if pred == "mixed" and obs == "both_beneficial":
        return "false_conservative_mixed"
    if pred == "both_detrimental" and obs != "both_detrimental":
        return "false_detrimental"
    if pred != "mixed" and obs == "mixed":
        return "missed_mixed"
    return "other_error"


def summarize_failures(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in df.groupby("failure_mode"):
        rows.append(
            {
                "failure_mode": mode,
                "count": int(len(group)),
                "fraction": float(len(group) / len(df)),
                "median_lambda": float(group["lambda_proxy"].median()),
                "median_RV": float(group["R_V_proxy"].median()),
                "median_n_s": float(group["n_s_proxy"].median()),
                "median_margin": float(group["map_margin"].median()),
                "median_measured_displacement_severity": float(group["measured_displacement_severity"].median()),
                "dataset_mix": "; ".join(f"{k}:{v}" for k, v in group["dataset"].value_counts().to_dict().items()),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def apply_guardrail(df: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "none":
        return pd.Series(False, index=df.index)
    if rule == "low_margin":
        return df["map_margin"] < 0.50
    if rule == "high_lambda":
        return df["lambda_proxy"] > 4.0
    if rule == "low_margin_or_high_lambda":
        return (df["map_margin"] < 0.50) | (df["lambda_proxy"] > 4.0)
    if rule == "release_low_margin":
        return df["release_by_map"] & (df["map_margin"] < 1.00)
    if rule == "release_high_lambda":
        return df["release_by_map"] & (df["lambda_proxy"] > 4.0)
    if rule == "release_low_margin_or_high_lambda":
        return df["release_by_map"] & ((df["map_margin"] < 1.00) | (df["lambda_proxy"] > 4.0))
    if rule == "release_displacement_pred_high":
        return df["release_by_map"] & (df["R_delta_total_predicted"] > 1.0)
    raise ValueError(rule)


def guardrail_metrics(df: pd.DataFrame, rule: str) -> dict:
    abstain = apply_guardrail(df, rule)
    release_after = df["release_by_map"] & ~abstain
    false_safe_before = df["false_safe"]
    false_safe_after = release_after & df["observed_class"].ne("both_beneficial")
    true_safe_released_before = df["release_by_map"] & df["observed_class"].eq("both_beneficial")
    true_safe_released_after = release_after & df["observed_class"].eq("both_beneficial")
    return {
        "guardrail": rule,
        "abstained_or_escalated_cases": int(abstain.sum()),
        "abstention_rate": float(abstain.mean()),
        "released_before": int(df["release_by_map"].sum()),
        "released_after": int(release_after.sum()),
        "false_safe_before": int(false_safe_before.sum()),
        "false_safe_after": int(false_safe_after.sum()),
        "false_safe_rate_before_all_cases": float(false_safe_before.mean()),
        "false_safe_rate_after_all_cases": float(false_safe_after.mean()),
        "false_safe_reduction": float(1.0 - false_safe_after.sum() / max(false_safe_before.sum(), 1)),
        "true_safe_released_before": int(true_safe_released_before.sum()),
        "true_safe_released_after": int(true_safe_released_after.sum()),
        "safe_release_retention": float(true_safe_released_after.sum() / max(true_safe_released_before.sum(), 1)),
        "cost_per_false_safe_removed": float(abstain.sum() / max(false_safe_before.sum() - false_safe_after.sum(), 1)),
    }


def guardrail_summary(df: pd.DataFrame) -> pd.DataFrame:
    rules = [
        "none",
        "low_margin",
        "high_lambda",
        "low_margin_or_high_lambda",
        "release_low_margin",
        "release_high_lambda",
        "release_low_margin_or_high_lambda",
        "release_displacement_pred_high",
    ]
    return pd.DataFrame([guardrail_metrics(df, rule) for rule in rules]).sort_values(
        ["false_safe_after", "abstention_rate"], ascending=[True, True]
    )


def candidate_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    release = df[df["release_by_map"]].copy()
    false_safe = release[release["false_safe"]]
    true_safe = release[~release["false_safe"]]
    for feature in ["map_margin", "lambda_proxy", "R_delta_total_predicted", "n_s_proxy", "measured_displacement_severity"]:
        for q in np.linspace(0.1, 0.9, 9):
            threshold = float(release[feature].quantile(q))
            for direction in ["<=", ">="]:
                if direction == "<=":
                    flagged = release[feature] <= threshold
                else:
                    flagged = release[feature] >= threshold
                fs_caught = int((flagged & release["false_safe"]).sum())
                ts_flagged = int((flagged & ~release["false_safe"]).sum())
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "false_safe_caught": fs_caught,
                        "true_safe_flagged": ts_flagged,
                        "false_safe_recall": fs_caught / max(len(false_safe), 1),
                        "true_safe_penalty": ts_flagged / max(len(true_safe), 1),
                        "rule_score": fs_caught / max(len(false_safe), 1) - 0.5 * ts_flagged / max(len(true_safe), 1),
                    }
                )
    return pd.DataFrame(rows).sort_values("rule_score", ascending=False)


def plot_failure_map(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {
        "correct_robust": "#2ca02c",
        "correct_marginal": "#98df8a",
        "false_safe": "#d62728",
        "false_conservative_mixed": "#1f77b4",
        "false_detrimental": "#9467bd",
        "missed_mixed": "#ff7f0e",
        "other_error": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=200)
    for mode, group in df.groupby("failure_mode"):
        ax.scatter(
            group["lambda_proxy"],
            group["R_V_proxy"],
            s=18,
            alpha=0.75,
            label=mode,
            color=colors.get(mode, "#7f7f7f"),
            edgecolor="none",
        )
    lam = np.linspace(max(1.01, df["lambda_proxy"].min()), df["lambda_proxy"].quantile(0.98), 200)
    ax.plot(lam, 1.0 / (lam * lam), color="black", linestyle="--", linewidth=1.0, label="lower mixed boundary")
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.8, label="force-neutral boundary")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda proxy")
    ax.set_ylabel("R_V proxy")
    ax.set_title("Failure-mode atlas on Demand-Polarity Map")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "failure_mode_map_lambda_rv.png")
    plt.close(fig)


def plot_guardrail(summary: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sub = summary[summary["guardrail"] != "none"].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)
    ax.scatter(sub["abstention_rate"], sub["false_safe_rate_after_all_cases"], s=80)
    for _, row in sub.iterrows():
        ax.annotate(row["guardrail"], (row["abstention_rate"], row["false_safe_rate_after_all_cases"]), fontsize=7)
    ax.set_xlabel("Abstention/escalation rate")
    ax.set_ylabel("False-safe rate after guardrail")
    ax.set_title("Guardrail tradeoff for map releases")
    fig.tight_layout()
    fig.savefig(FIG / "guardrail_tradeoff.png")
    plt.close(fig)


def write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_both(name: str, df: pd.DataFrame) -> None:
    write(OUT / name, df)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    df = load_cases()
    failure_summary = summarize_failures(df)
    candidate_summary = candidate_rules(df)
    guardrails = guardrail_summary(df)
    plot_failure_map(df)
    plot_guardrail(guardrails)
    write_both("failure_mode_cases.csv", df)
    write_both("failure_mode_summary.csv", failure_summary)
    write_both("guardrail_candidate_summary.csv", candidate_summary)
    write_both("guardrail_application_summary.csv", guardrails)

    best = guardrails[(guardrails["false_safe_after"] == guardrails["false_safe_after"].min()) & (guardrails["guardrail"] != "none")]
    best = best.sort_values(["abstention_rate", "safe_release_retention"], ascending=[True, False]).iloc[0].to_dict()
    no_guardrail = guardrails[guardrails["guardrail"] == "none"].iloc[0].to_dict()
    false_safe_cases = df[df["failure_mode"] == "false_safe"]
    report = {
        "experiment": "failure_mode_guardrails",
        "isolated_from_submission_workflow": False,
        "n_cases": int(len(df)),
        "false_safe_cases": int(len(false_safe_cases)),
        "main_false_safe_region": {
            "median_lambda": float(false_safe_cases["lambda_proxy"].median()) if len(false_safe_cases) else None,
            "median_RV": float(false_safe_cases["R_V_proxy"].median()) if len(false_safe_cases) else None,
            "median_margin": float(false_safe_cases["map_margin"].median()) if len(false_safe_cases) else None,
            "dataset_mix": false_safe_cases["dataset"].value_counts().to_dict(),
        },
        "no_guardrail": {
            "released_cases": int(no_guardrail["released_after"]),
            "false_safe_rate_all_cases": float(no_guardrail["false_safe_rate_after_all_cases"]),
            "true_safe_released": int(no_guardrail["true_safe_released_after"]),
        },
        "best_guardrail": {
            "rule": best["guardrail"],
            "abstention_rate": float(best["abstention_rate"]),
            "released_after": int(best["released_after"]),
            "false_safe_after": int(best["false_safe_after"]),
            "false_safe_rate_after_all_cases": float(best["false_safe_rate_after_all_cases"]),
            "false_safe_reduction": float(best["false_safe_reduction"]),
            "safe_release_retention": float(best["safe_release_retention"]),
            "cost_per_false_safe_removed": float(best["cost_per_false_safe_removed"]),
        },
        "claim_boundary": "Empirical guardrail discovery for safer screening only; not nonlinear 3D SSI validation.",
    }
    (SUMMARY_OUT / "failure_mode_guardrail_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
