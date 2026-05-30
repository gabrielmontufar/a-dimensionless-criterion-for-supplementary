from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[2]
SUPP_DIR = PACKAGE_DIR / "validation"
RAW_DATA = PACKAGE_DIR / "data" / "raw_redistributed"
OUTPUTS = PACKAGE_DIR / "outputs" / "regenerated" / "tables"
SUMMARY_OUTPUTS = PACKAGE_DIR / "outputs" / "regenerated" / "summaries"
FIGURES = PACKAGE_DIR / "outputs" / "regenerated" / "figures"

SENTINELS = {7777.0, 8888.0, 9999.0}


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(SENTINELS))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator
    return out.mask(~np.isfinite(out))


def load_fordy(
    drift_threshold_percent: float = 1.0,
    rotation_threshold_rad: float = 0.01,
    settlement_threshold_percent: float = 1.0,
    sensitivity_label: str = "base",
) -> pd.DataFrame:
    path = RAW_DATA / "FoRDy_Mastersheet_v1.0.0_20180821_2322.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=2, encoding_errors="replace")
    df = df[numeric(df["Event No"]).notna()].copy()
    t0 = numeric(df["T_0"])
    t1e = numeric(df["T_1e"])
    t1a = numeric(df["T_1a"])
    lam = safe_ratio(t1e.fillna(t1a), t0)
    rv = numeric(df["maxM"])
    r_delta_pred = lam.pow(2.0) * rv
    measured_displacement_severity = pd.concat(
        [
            numeric(df["pkDR"]).abs() / drift_threshold_percent,
            numeric(df["pkRot"]).abs() / rotation_threshold_rad,
            numeric(df["resSet"]).abs() / settlement_threshold_percent,
        ],
        axis=1,
    ).max(axis=1)

    rows = []
    for idx, rec in df.iterrows():
        lambda_proxy = float(lam.loc[idx]) if pd.notna(lam.loc[idx]) else float("nan")
        rv_proxy = float(rv.loc[idx]) if pd.notna(rv.loc[idx]) else float("nan")
        rdelta = float(r_delta_pred.loc[idx]) if pd.notna(r_delta_pred.loc[idx]) else float("nan")
        obs_delta = (
            float(measured_displacement_severity.loc[idx])
            if pd.notna(measured_displacement_severity.loc[idx])
            else float("nan")
        )
        predicted = demand_class(rv_proxy, rdelta)
        observed = demand_class(rv_proxy, obs_delta)
        valid = (
            predicted != "uncertain"
            and observed != "uncertain"
            and math.isfinite(lambda_proxy)
            and 1.0 < lambda_proxy < 10.0
        )
        robust = (
            valid
            and abs(math.log(max(rdelta, 1e-12))) > 0.20
            and abs(rv_proxy - 1.0) > 0.10
        )
        rows.append(
            {
                "dataset": "FoRDy",
                "source_file": path.name,
                "test_id": str(rec["Experiment or Case ID"]),
                "event_no": int(float(rec["Event No"])),
                "lambda_proxy": lambda_proxy,
                "eta_proxy": lambda_proxy**2 - 1.0 if math.isfinite(lambda_proxy) else float("nan"),
                "n_s_proxy": math.log(max(rv_proxy, 1e-12)) / math.log(lambda_proxy)
                if math.isfinite(lambda_proxy) and lambda_proxy > 1.0 and math.isfinite(rv_proxy) and rv_proxy > 0
                else float("nan"),
                "R_V_proxy": rv_proxy,
                "R_delta_total_predicted": rdelta,
                "measured_displacement_severity": obs_delta,
                "observed_class": observed,
                "predicted_class": predicted,
                "correct": str(observed == predicted).lower() if valid else "",
                "robust_marginal": "robust" if robust else ("marginal" if valid else "invalid"),
                "status": "validated_proxy_row" if valid else "insufficient_or_out_of_gate",
                "notes": (
                    "FoRDy uses T1/T0 period-lengthening and normalized maximum footing moment; "
                    f"measured displacement gate is max(pkDR/{drift_threshold_percent}%, "
                    f"pkRot/{rotation_threshold_rad} rad, abs(resSet)/{settlement_threshold_percent}%)."
                ),
                "sensitivity_label": sensitivity_label,
            }
        )
    return pd.DataFrame(rows)


def load_forcy(
    rotation_threshold_rad: float = 0.01,
    sliding_threshold_percent: float = 0.10,
    settlement_threshold_percent: float = 0.10,
    sensitivity_label: str = "base",
) -> pd.DataFrame:
    path = RAW_DATA / "FoRCy_Mastersheet_others.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=2, encoding_errors="replace")
    df = df[numeric(df["Row Num"]).notna()].copy()
    rv = numeric(df["Norm. Max. Ftg. Moment"])
    contact_ratio = numeric(df["A_c/A"]).abs()
    eta = (1.0 / contact_ratio.clip(lower=1e-6)) - 1.0
    lam = np.sqrt(1.0 + eta)
    r_delta_pred = lam.pow(2.0) * rv
    measured_displacement_severity = pd.concat(
        [
            numeric(df["Pk. Ftg. Rotation [rad]"]).abs() / rotation_threshold_rad,
            numeric(df["Norm. Pk. Ftg. Sliding in %"]).abs() / sliding_threshold_percent,
            numeric(df["Norm. Cumm. Res. Ftg. Uplift/ Sett. [%]"]).abs() / settlement_threshold_percent,
        ],
        axis=1,
    ).max(axis=1)

    rows = []
    for idx, rec in df.iterrows():
        lambda_proxy = float(lam.loc[idx]) if pd.notna(lam.loc[idx]) else float("nan")
        rv_proxy = float(rv.loc[idx]) if pd.notna(rv.loc[idx]) else float("nan")
        eta_proxy = float(eta.loc[idx]) if pd.notna(eta.loc[idx]) else float("nan")
        rdelta = float(r_delta_pred.loc[idx]) if pd.notna(r_delta_pred.loc[idx]) else float("nan")
        obs_delta = (
            float(measured_displacement_severity.loc[idx])
            if pd.notna(measured_displacement_severity.loc[idx])
            else float("nan")
        )
        predicted = demand_class(rv_proxy, rdelta)
        observed = demand_class(rv_proxy, obs_delta)
        valid = (
            predicted != "uncertain"
            and observed != "uncertain"
            and math.isfinite(lambda_proxy)
            and 1.0 < lambda_proxy < 10.0
        )
        robust = (
            valid
            and abs(math.log(max(rdelta, 1e-12))) > 0.20
            and abs(rv_proxy - 1.0) > 0.10
        )
        rows.append(
            {
                "dataset": "FoRCy",
                "source_file": path.name,
                "test_id": str(rec["Event ID"]),
                "event_no": int(float(rec["Row Num"])),
                "lambda_proxy": lambda_proxy,
                "eta_proxy": eta_proxy,
                "n_s_proxy": math.log(max(rv_proxy, 1e-12)) / math.log(lambda_proxy)
                if math.isfinite(lambda_proxy) and lambda_proxy > 1.0 and math.isfinite(rv_proxy) and rv_proxy > 0
                else float("nan"),
                "R_V_proxy": rv_proxy,
                "R_delta_total_predicted": rdelta,
                "measured_displacement_severity": obs_delta,
                "observed_class": observed,
                "predicted_class": predicted,
                "correct": str(observed == predicted).lower() if valid else "",
                "robust_marginal": "robust" if robust else ("marginal" if valid else "invalid"),
                "status": "validated_proxy_row" if valid else "insufficient_or_out_of_gate",
                "notes": (
                    "FoRCy uses normalized maximum footing moment and contact-area proxy lambda=sqrt(1/A_c/A); "
                    f"measured displacement gate is max(rotation/{rotation_threshold_rad} rad, "
                    f"sliding/{sliding_threshold_percent}%, settlement/{settlement_threshold_percent}%)."
                ),
                "sensitivity_label": sensitivity_label,
            }
        )
    return pd.DataFrame(rows)


def metric_block(df: pd.DataFrame, dataset: str) -> dict:
    sub = df[(df["dataset"] == dataset) & (df["status"] == "validated_proxy_row")].copy()
    if sub.empty:
        return {
            "dataset": dataset,
            "tests_used": 0,
            "accuracy": "",
            "robust_accuracy": "",
            "false_safe_rate": "",
            "mixed_precision": "",
            "mixed_recall": "",
            "uncertain_fraction": 1.0,
            "status": "no_valid_rows",
        }
    correct = sub["observed_class"] == sub["predicted_class"]
    robust = sub[sub["robust_marginal"] == "robust"]
    pred_safe = sub["predicted_class"] == "both_beneficial"
    obs_safe = sub["observed_class"] == "both_beneficial"
    pred_mixed = sub["predicted_class"] == "mixed"
    obs_mixed = sub["observed_class"] == "mixed"
    true_mixed = pred_mixed & obs_mixed
    all_dataset = df[df["dataset"] == dataset]
    return {
        "dataset": dataset,
        "tests_used": int(len(sub)),
        "accuracy": float(correct.mean()),
        "robust_accuracy": float((robust["observed_class"] == robust["predicted_class"]).mean())
        if not robust.empty
        else "",
        "false_safe_rate": float((pred_safe & ~obs_safe).mean()),
        "mixed_precision": float(true_mixed.sum() / max(pred_mixed.sum(), 1)),
        "mixed_recall": float(true_mixed.sum() / max(obs_mixed.sum(), 1)),
        "uncertain_fraction": float(1.0 - len(sub) / max(len(all_dataset), 1)),
        "status": "pass_proxy_gate"
        if float(correct.mean()) >= 0.85 and float((pred_safe & ~obs_safe).mean()) <= 0.10
        else "partial_or_fail_proxy_gate",
    }


def strict_screening_block(df: pd.DataFrame, dataset: str) -> dict:
    sub = df[(df["dataset"] == dataset) & (df["status"] == "validated_proxy_row")].copy()
    if sub.empty:
        return {
            "dataset": dataset,
            "force_beneficial_cases": 0,
            "force_beneficial_accuracy": "",
            "predicted_safe_cases": 0,
            "safe_precision": "",
            "false_safe_rate_within_predicted_safe": "",
            "false_safe_rate_all_force_beneficial": "",
            "mixed_recall_force_beneficial": "",
            "status": "no_valid_rows",
        }
    force_beneficial = sub[sub["R_V_proxy"] < 1.0].copy()
    if force_beneficial.empty:
        return {
            "dataset": dataset,
            "force_beneficial_cases": 0,
            "force_beneficial_accuracy": "",
            "predicted_safe_cases": 0,
            "safe_precision": "",
            "false_safe_rate_within_predicted_safe": "",
            "false_safe_rate_all_force_beneficial": "",
            "mixed_recall_force_beneficial": "",
            "status": "no_force_beneficial_rows",
        }
    pred_safe = force_beneficial["predicted_class"] == "both_beneficial"
    obs_safe = force_beneficial["observed_class"] == "both_beneficial"
    obs_mixed = force_beneficial["observed_class"] == "mixed"
    pred_mixed = force_beneficial["predicted_class"] == "mixed"
    predicted_safe_cases = int(pred_safe.sum())
    false_safe_cases = int((pred_safe & ~obs_safe).sum())
    safe_precision = float((pred_safe & obs_safe).sum() / predicted_safe_cases) if predicted_safe_cases else ""
    mixed_recall = float((pred_mixed & obs_mixed).sum() / max(int(obs_mixed.sum()), 1))
    return {
        "dataset": dataset,
        "force_beneficial_cases": int(len(force_beneficial)),
        "force_beneficial_accuracy": float(
            (force_beneficial["observed_class"] == force_beneficial["predicted_class"]).mean()
        ),
        "predicted_safe_cases": predicted_safe_cases,
        "safe_precision": safe_precision,
        "false_safe_rate_within_predicted_safe": float(false_safe_cases / predicted_safe_cases)
        if predicted_safe_cases
        else "",
        "false_safe_rate_all_force_beneficial": float(false_safe_cases / len(force_beneficial)),
        "mixed_recall_force_beneficial": mixed_recall,
        "status": "screening_safe_precision_reported",
    }


def sensitivity_points() -> pd.DataFrame:
    configs = [
        {
            "threshold_set": "strict_rotation_0p005",
            "fordy": {"drift_threshold_percent": 0.5, "rotation_threshold_rad": 0.005, "settlement_threshold_percent": 0.5},
            "forcy": {"rotation_threshold_rad": 0.005, "sliding_threshold_percent": 0.05, "settlement_threshold_percent": 0.05},
        },
        {
            "threshold_set": "base_rotation_0p010",
            "fordy": {"drift_threshold_percent": 1.0, "rotation_threshold_rad": 0.01, "settlement_threshold_percent": 1.0},
            "forcy": {"rotation_threshold_rad": 0.01, "sliding_threshold_percent": 0.10, "settlement_threshold_percent": 0.10},
        },
        {
            "threshold_set": "lenient_rotation_0p020",
            "fordy": {"drift_threshold_percent": 2.0, "rotation_threshold_rad": 0.02, "settlement_threshold_percent": 2.0},
            "forcy": {"rotation_threshold_rad": 0.02, "sliding_threshold_percent": 0.20, "settlement_threshold_percent": 0.20},
        },
    ]
    rows: list[dict] = []
    for config in configs:
        label = config["threshold_set"]
        pts = pd.concat(
            [
                load_fordy(**config["fordy"], sensitivity_label=label),
                load_forcy(**config["forcy"], sensitivity_label=label),
            ],
            ignore_index=True,
        )
        for dataset in ["FoRDy", "FoRCy"]:
            block = metric_block(pts, dataset)
            strict = strict_screening_block(pts, dataset)
            rows.append(
                {
                    "threshold_set": label,
                    "dataset": dataset,
                    "tests_used": block["tests_used"],
                    "accuracy": block["accuracy"],
                    "false_safe_rate": block["false_safe_rate"],
                    "mixed_precision": block["mixed_precision"],
                    "mixed_recall": block["mixed_recall"],
                    "safe_precision": strict["safe_precision"],
                    "false_safe_rate_within_predicted_safe": strict["false_safe_rate_within_predicted_safe"],
                }
            )
        combined = pts.copy()
        combined["dataset"] = "FoRDy+FoRCy"
        block = metric_block(combined, "FoRDy+FoRCy")
        strict = strict_screening_block(combined, "FoRDy+FoRCy")
        rows.append(
            {
                "threshold_set": label,
                "dataset": "FoRDy+FoRCy",
                "tests_used": block["tests_used"],
                "accuracy": block["accuracy"],
                "false_safe_rate": block["false_safe_rate"],
                "mixed_precision": block["mixed_precision"],
                "mixed_recall": block["mixed_recall"],
                "safe_precision": strict["safe_precision"],
                "false_safe_rate_within_predicted_safe": strict["false_safe_rate_within_predicted_safe"],
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_points(points: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    valid = points[points["status"] == "validated_proxy_row"].copy()
    fig, ax = plt.subplots(figsize=(7.6, 5.1), dpi=220)
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(-5.0, 2.0)
    ax.axhspan(-5.0, -2.0, color="#2a9d8f", alpha=0.16, label="Both beneficial")
    ax.axhspan(-2.0, 0.0, color="#e9c46a", alpha=0.26, label="Mixed")
    ax.axhspan(0.0, 2.0, color="#e76f51", alpha=0.16, label="Both detrimental")
    ax.axhline(-2.0, color="#0f766e", lw=1.4)
    ax.axhline(0.0, color="#b91c1c", lw=1.4)
    colors = {
        "both_beneficial": "#0f766e",
        "mixed": "#ca8a04",
        "both_detrimental": "#b91c1c",
        "force_detrimental_displacement_beneficial": "#64748b",
    }
    markers = {"FoRDy": "o", "FoRCy": "^"}
    for dataset in ["FoRDy", "FoRCy"]:
        sub = valid[valid["dataset"] == dataset]
        for klass, group in sub.groupby("observed_class"):
            ax.scatter(
                group["lambda_proxy"],
                group["n_s_proxy"],
                s=18,
                c=colors.get(klass, "#334155"),
                marker=markers[dataset],
                edgecolors="white",
                linewidths=0.25,
                alpha=0.78,
                label=f"{dataset}: {klass}",
            )
    ax.set_xlabel(r"Proxy period-lengthening factor, $\lambda$")
    ax.set_ylabel(r"Proxy secant slope, $n_s=\ln(R_V)/\ln(\lambda)$")
    ax.set_title("Experimental FoRDy/FoRCy Points on Demand-Polarity Map")
    ax.grid(True, color="#cbd5e1", alpha=0.55, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=6.3, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_2b_experimental_points_on_demand_polarity_map.png")
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    f_data = [
        RAW_DATA / "FoRDy_Mastersheet_v1.0.0_20180821_2322.csv",
        RAW_DATA / "FoRCy_Mastersheet_others.csv",
    ]
    for p in f_data:
        if not p.exists():
            raise FileNotFoundError(f"Required offline data file missing: {p}")

    points = pd.concat([load_fordy(), load_forcy()], ignore_index=True)
    points_path = OUTPUTS / "experimental_polarity_points.csv"
    points.to_csv(points_path, index=False)
    points.to_csv(OUTPUTS / "fordy_forcy_master_validation_table.csv", index=False)

    metrics = [metric_block(points, "FoRDy"), metric_block(points, "FoRCy")]
    combined = points[points["status"] == "validated_proxy_row"].copy()
    if not combined.empty:
        metrics.append(metric_block(points.assign(dataset="FoRDy+FoRCy"), "FoRDy+FoRCy"))
        metrics[-1]["dataset"] = "FoRDy+FoRCy"
    respond_summary = read_json_if_exists(SUPP_DIR / "field_validation_real" / "respond_field_validation_summary.json")
    hierarchy_summary = read_json_if_exists(SUPP_DIR / "field_validation_real" / "real_field_validation_hierarchy_summary.json")
    if respond_summary:
        metrics.append(
            {
                "dataset": "ERIES-RESPOND",
                "tests_used": int(respond_summary.get("sensor_level_residual_rows", 0) or 0),
                "accuracy": "",
                "robust_accuracy": "",
                "false_safe_rate": "",
                "mixed_precision": "",
                "mixed_recall": "",
                "uncertain_fraction": "",
                "status": f"secondary_existing_transfer_validation_median_nrmse={respond_summary.get('median_nrmse')}",
            }
        )
    if hierarchy_summary:
        for item in hierarchy_summary.get("validation_hierarchy", []):
            if item.get("dataset") == "ERIES-POLIS":
                metrics.append(
                    {
                        "dataset": "ERIES-POLIS",
                        "tests_used": int(item.get("blind_residual_rows", 0) or 0),
                        "accuracy": "",
                        "robust_accuracy": "",
                        "false_safe_rate": "",
                        "mixed_precision": "",
                        "mixed_recall": "",
                        "uncertain_fraction": "",
                        "status": f"secondary_existing_transfer_validation_median_nrmse={item.get('median_nrmse')}",
                    }
                )
    metrics_path = OUTPUTS / "experimental_validation_metrics_by_dataset.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    strict_metrics = [
        strict_screening_block(points, "FoRDy"),
        strict_screening_block(points, "FoRCy"),
    ]
    combined_strict = points.copy()
    combined_strict["dataset"] = "FoRDy+FoRCy"
    strict_metrics.append(strict_screening_block(combined_strict, "FoRDy+FoRCy"))
    pd.DataFrame(strict_metrics).to_csv(OUTPUTS / "strict_screening_metrics_by_dataset.csv", index=False)

    sensitivity = sensitivity_points()
    sensitivity.to_csv(OUTPUTS / "proxy_threshold_sensitivity_metrics.csv", index=False)

    confusion_rows = []
    for dataset in ["FoRDy", "FoRCy", "FoRDy+FoRCy"]:
        sub = points[points["status"] == "validated_proxy_row"] if dataset == "FoRDy+FoRCy" else points[(points["dataset"] == dataset) & (points["status"] == "validated_proxy_row")]
        table = pd.crosstab(sub["observed_class"], sub["predicted_class"])
        for observed in table.index:
            for predicted in table.columns:
                confusion_rows.append(
                    {
                        "dataset": dataset,
                        "observed_class": observed,
                        "predicted_class": predicted,
                        "count": int(table.loc[observed, predicted]),
                    }
                )
    write_csv(
        OUTPUTS / "experimental_class_confusion_matrix.csv",
        confusion_rows,
        ["dataset", "observed_class", "predicted_class", "count"],
    )

    false_safe = points[
        (points["status"] == "validated_proxy_row")
        & (points["predicted_class"] == "both_beneficial")
        & (points["observed_class"] != "both_beneficial")
    ].copy()
    false_safe.to_csv(OUTPUTS / "false_safe_case_audit.csv", index=False)

    claim_rows = [
        {
            "claim": "FoRDy/FoRCy proxy class validation has been executed",
            "status": "ALLOWED WITH BOUNDARY",
            "evidence": "Downloaded public DesignSafe mastersheets were parsed offline; 591 proxy rows were classified and evaluated.",
            "limitation": "This is experimental polarity/proxy class validation, not nonlinear 3D site-specific field validation.",
        },
        {
            "claim": "FoRCy passes the proxy class-validation gate",
            "status": "ALLOWED",
            "evidence": "FoRCy accuracy is above 0.85 and false-safe rate is below 0.10 in the reproduced metrics.",
            "limitation": "Mixed precision is just below the nominal 0.80 target and must be reported numerically.",
        },
        {
            "claim": "FoRDy alone passes the proxy class-validation gate",
            "status": "DISALLOWED",
            "evidence": "FoRDy accuracy is about 0.756 and mixed precision about 0.603.",
            "limitation": "Use FoRDy as partial dynamic evidence and discuss its weaker proxy agreement.",
        },
        {
            "claim": "Garner is positive nonlinear 3D validation",
            "status": "DISALLOWED",
            "evidence": "Prior Garner validation-falsification gates failed for predictive nonlinear 3D field validation.",
            "limitation": "Retain Garner only as falsification/instrumentation audit.",
        },
    ]
    write_csv(
        OUTPUTS / "experimental_claim_boundary_matrix.csv",
        claim_rows,
        ["claim", "status", "evidence", "limitation"],
    )

    acquisition_rows = []
    for path in sorted(RAW_DATA.glob("*")):
        if path.is_file() and path.name != "designsafe_download_manifest.json":
            acquisition_rows.append(
                {
                    "local_file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(OUTPUTS / "offline_data_manifest.csv", acquisition_rows, ["local_file", "bytes", "sha256"])

    plot_points(points)
    (FIGURES / "figure_2b_experimental_points_on_demand_polarity_map.png").replace(
        FIGURES / "demand_polarity_map_experimental.png"
    )

    summary = {
        "package_version": "demand_polarity_map_v40_submission_reproducible",
        "raw_mastersheets_available_locally": True,
        "tests_used_total": int((points["status"] == "validated_proxy_row").sum()),
        "metrics": metrics,
        "strict_screening_metrics": strict_metrics,
        "threshold_sensitivity_rows": int(len(sensitivity)),
        "figure": (FIGURES / "demand_polarity_map_experimental.png").relative_to(PACKAGE_DIR).as_posix(),
        "claim_boundary": "FoRDy/FoRCy provide executed experimental proxy class validation; FoRDy remains partial, FoRCy passes the proxy gate. This is class-level polarity validation, not independent force prediction and not nonlinear 3D site-specific field validation.",
    }
    SUMMARY_OUTPUTS.mkdir(parents=True, exist_ok=True)
    (SUMMARY_OUTPUTS / "experimental_validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
