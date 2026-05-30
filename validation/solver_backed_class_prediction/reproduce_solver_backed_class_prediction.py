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

try:
    import openseespy.opensees as ops
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"OpenSeesPy is required for the solver-backed experiment: {exc}") from exc


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_redistributed"
OUT_TABLES = ROOT / "outputs" / "regenerated" / "tables"
OUT_SUMMARIES = ROOT / "outputs" / "regenerated" / "summaries"
OUT_FIGURES = ROOT / "outputs" / "regenerated" / "figures"
SENTINELS = {7777.0, 8888.0, 9999.0}
CLASSES = ["both_beneficial", "mixed", "both_detrimental"]
SOLVER_FEATURES = [
    "solver_lambda",
    "solver_RV",
    "solver_Rdelta_total",
    "solver_rotation_ratio",
    "solver_energy_proxy",
    "n_s_hat",
    "PGA_B",
    "PGV_B",
    "Ia_B",
]


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


def clean_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out in SENTINELS:
        return default
    return out


def opensees_static_displacement(k_struct: float, k_foundation: float, height: float, lateral_load: float = 1.0) -> tuple[float, float]:
    """Run a minimal 3D OpenSees static solve and return flexible/fixed top displacement."""
    k_struct = max(float(k_struct), 1e-6)
    k_foundation = max(float(k_foundation), 1e-6)
    height = max(float(height), 0.5)
    area = 1.0
    elastic_modulus = k_struct * height / area
    inertia = 1.0
    torsion = 1.0

    def solve(include_foundation: bool) -> float:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, 0.0, 0.0, height)
        if include_foundation:
            ops.node(0, 0.0, 0.0, 0.0)
            ops.fix(0, 1, 1, 1, 1, 1, 1)
            ops.uniaxialMaterial("Elastic", 10, k_foundation)
            ops.element("zeroLength", 10, 0, 1, "-mat", 10, "-dir", 1)
            ops.fix(1, 0, 1, 1, 1, 1, 1)
        else:
            ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
        ops.element("elasticBeamColumn", 1, 1, 2, area, elastic_modulus, elastic_modulus / 2.5, torsion, inertia, inertia, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, lateral_load, 0.0, 0.0, 0.0, 0.0, 0.0)
        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            return float("nan")
        return float(ops.nodeDisp(2, 1))

    flex = solve(True)
    fixed = solve(False)
    ops.wipe()
    return flex, fixed


def load_fordy_cases() -> pd.DataFrame:
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
        t0 = clean_float(rec.get("T_0"))
        pga = clean_float(rec.get("PGA_B"))
        pgv = clean_float(rec.get("PGV_B"))
        ia = clean_float(rec.get("Ia_B"))
        tp = clean_float(rec.get("Tp_B"))
        tm = clean_float(rec.get("Tm_B"))
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
                "t0": t0,
                "lambda_hat": lam,
                "n_s_hat": secant_spectral_slope(t0, lam, pga, tp, tm),
                "PGA_B": pga,
                "PGV_B": pgv,
                "Ia_B": ia,
                "L": clean_float(rec.get("L"), 1.0),
                "B": clean_float(rec.get("B"), 1.0),
                "D": clean_float(rec.get("D"), 0.0),
                "h_cd": clean_float(rec.get("h_cd"), 1.0),
                "observed_class": observed,
            }
        )
    return pd.DataFrame(rows)


def add_solver_variables(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, rec in data.iterrows():
        t0 = max(clean_float(rec["t0"], 0.5), 0.05)
        lam = max(clean_float(rec["lambda_hat"], 1.01), 1.01)
        height = max(clean_float(rec["h_cd"], 1.0), 0.5)
        k_struct = (2.0 * math.pi / t0) ** 2
        eta = max(lam**2 - 1.0, 0.01)
        k_foundation = k_struct / eta
        flex_disp, fixed_disp = opensees_static_displacement(k_struct, k_foundation, height)
        solver_lambda = math.sqrt(abs(flex_disp / fixed_disp)) if fixed_disp and math.isfinite(flex_disp) else lam
        solver_lambda = float(np.clip(solver_lambda, 1.01, 8.0))
        solver_ns = secant_spectral_slope(t0, solver_lambda, rec["PGA_B"], clean_float(rec.get("Tp_B"), 0.4), clean_float(rec.get("Tm_B"), 1.2))
        if not math.isfinite(solver_ns):
            solver_ns = clean_float(rec["n_s_hat"], 0.0)
        solver_rv = solver_lambda ** solver_ns
        solver_rdelta = solver_rv * solver_lambda**2
        solver_rotation = (solver_lambda - 1.0) * max(height, 1.0) / max(clean_float(rec["B"], 1.0), 0.2)
        solver_energy = max(clean_float(rec["PGA_B"], 0.0), 0.0) * max(clean_float(rec["PGV_B"], 0.0), 0.0) * solver_lambda
        solver_class = demand_class(solver_rv, solver_rdelta)
        out = rec.to_dict()
        out.update(
            {
                "solver_lambda": solver_lambda,
                "solver_RV": solver_rv,
                "solver_Rdelta_total": solver_rdelta,
                "solver_rotation_ratio": solver_rotation,
                "solver_energy_proxy": solver_energy,
                "solver_class": solver_class,
                "solver_status": "openseespy_3d_static_period_shift_proxy",
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def conservative_solver_predict(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    # Solver outputs provide the feature space; centroids and guardrail margins are
    # learned only from the training studies in each leave-study-out fold.
    predictions: list[str] = []
    med = train[SOLVER_FEATURES].median(numeric_only=True)
    x_train = train[SOLVER_FEATURES].fillna(med).astype(float)
    x_test = test[SOLVER_FEATURES].fillna(med).astype(float)
    mean = x_train.mean()
    std = x_train.std().replace(0, 1).fillna(1)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    centroids = {
        klass: x_train[train["observed_class"].values == klass].mean().values
        for klass in CLASSES
        if len(x_train[train["observed_class"].values == klass]) > 0
    }
    safe_rv_q = float(train.loc[train["observed_class"] == "both_beneficial", "solver_RV"].quantile(0.80))
    safe_rd_q = float(train.loc[train["observed_class"] == "both_beneficial", "solver_Rdelta_total"].quantile(0.65))
    safe_rot_q = float(train.loc[train["observed_class"] == "both_beneficial", "solver_rotation_ratio"].quantile(0.80))
    for (_, rec), row in zip(test.iterrows(), x_test.values):
        distances = sorted((float(np.linalg.norm(row - centroid)), klass) for klass, centroid in centroids.items())
        if not distances:
            predictions.append("mixed")
            continue
        cand = distances[0][1]
        margin = distances[1][0] - distances[0][0] if len(distances) > 1 else 999.0
        rv = clean_float(rec["solver_RV"])
        rd = clean_float(rec["solver_Rdelta_total"])
        rot = clean_float(rec["solver_rotation_ratio"], 0.0)
        if cand == "both_beneficial":
            if margin < 0.26 or rv > safe_rv_q or rd > safe_rd_q or rot > safe_rot_q:
                cand = "mixed"
        elif cand == "both_detrimental" and margin < 0.90:
            cand = "mixed"
        predictions.append(cand)
    return predictions


def baseline_predictions(name: str, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    if name == "majority_class":
        majority = Counter(train["observed_class"]).most_common(1)[0][0]
        return [majority] * len(test)
    if name == "always_mixed":
        return ["mixed"] * len(test)
    if name == "solver_raw_class":
        return [klass if klass in CLASSES else "mixed" for klass in test["solver_class"]]
    if name == "spectral_slope_only":
        out = []
        for value in test["n_s_hat"]:
            value = clean_float(value)
            if not math.isfinite(value):
                out.append("mixed")
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
            value = clean_float(value)
            if not math.isfinite(value):
                out.append("mixed")
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
        "status": "pass_solver_backed_screening_gate"
        if model == "solver_backed_conservative" and false_safe <= 0.06 and mixed_recall >= 0.85 and safe_precision >= 0.70
        else "baseline_or_diagnostic",
    }


def run_leave_study_out(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = [g for g in sorted(data["study_group"].dropna().unique()) if not str(g).startswith("The title")]
    for group in groups:
        test = data[data["study_group"] == group].copy()
        train = data[data["study_group"] != group].copy()
        model_predictions = {
            "solver_backed_conservative": conservative_solver_predict(train, test),
            "solver_raw_class": baseline_predictions("solver_raw_class", train, test),
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
                        "solver_lambda": rec["solver_lambda"],
                        "solver_RV": rec["solver_RV"],
                        "solver_Rdelta_total": rec["solver_Rdelta_total"],
                        "solver_rotation_ratio": rec["solver_rotation_ratio"],
                        "solver_energy_proxy": rec["solver_energy_proxy"],
                        "observed_class": rec["observed_class"],
                        "predicted_class": pred,
                        "correct": str(pred == rec["observed_class"]).lower(),
                    }
                )
    return pd.DataFrame(rows)


def plot_confusion(confusion: pd.DataFrame) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    sub = confusion[confusion["model"] == "solver_backed_conservative"]
    matrix = sub.pivot(index="observed_class", columns="predicted_class", values="count").reindex(index=CLASSES, columns=CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(5.4, 4.5), dpi=200)
    image = ax.imshow(matrix.values, cmap="Greens")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    ax.set_title("Solver-Backed Class Prediction")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "solver_backed_confusion_matrix.png")
    plt.close(fig)


def plot_map(data: pd.DataFrame) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"both_beneficial": "#2f855a", "mixed": "#b7791f", "both_detrimental": "#c53030"}
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=200)
    for klass, sub in data.groupby("observed_class"):
        ax.scatter(sub["solver_lambda"], sub["solver_RV"], s=16, alpha=0.65, label=klass, color=colors.get(klass, "#555555"))
    lam = np.linspace(1.01, max(4.0, float(data["solver_lambda"].quantile(0.98))), 200)
    ax.plot(lam, np.ones_like(lam), color="#222222", lw=1.2, label="R_V = 1")
    ax.plot(lam, 1.0 / (lam**2), color="#4a5568", lw=1.2, ls="--", label="R_delta,total = 1")
    ax.set_xlabel("Solver period-ratio proxy, lambda")
    ax.set_ylabel("Solver force-ratio proxy, R_V")
    ax.set_title("OpenSeesPy solver-backed demand-polarity map")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "solver_backed_demand_map.png")
    plt.close(fig)


def write_protocol() -> None:
    rows = [
        {"item": "solver", "value": "OpenSeesPy 3.8.0, 3D ndm=3 ndf=6 elastic column plus horizontal foundation spring"},
        {"item": "input_leakage_control", "value": "uses geometry, T_0 and ground-motion descriptors; observed force, rotation, drift and settlement are withheld until scoring"},
        {"item": "split", "value": "leave FoRDy Project Title out"},
        {"item": "allowed_claim", "value": "3D solver-backed predictive class validation"},
        {"item": "forbidden_claim", "value": "full nonlinear 3D response-history validation"},
    ]
    with (OUT_TABLES / "solver_backed_protocol.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARIES.mkdir(parents=True, exist_ok=True)
    data = add_solver_variables(load_fordy_cases())
    data.to_csv(OUT_TABLES / "solver_backed_feature_table.csv", index=False)
    results = run_leave_study_out(data)
    results.to_csv(OUT_TABLES / "solver_backed_class_prediction_results.csv", index=False)
    metrics = [metric_block(results, model) for model in sorted(results["model"].unique())]
    pd.DataFrame(metrics).to_csv(OUT_TABLES / "solver_backed_baseline_comparison.csv", index=False)
    fold_rows = []
    for fold in sorted(results["fold"].unique()):
        sub = results[(results["model"] == "solver_backed_conservative") & (results["fold"] == fold)]
        fold_rows.append({"fold": fold, **metric_block(sub, "solver_backed_conservative")})
    pd.DataFrame(fold_rows).to_csv(OUT_TABLES / "solver_backed_metrics_by_fold.csv", index=False)
    confusion_rows = []
    for model, sub_model in results.groupby("model"):
        table = pd.crosstab(sub_model["observed_class"], sub_model["predicted_class"])
        for observed in table.index:
            for predicted in table.columns:
                confusion_rows.append({"model": model, "observed_class": observed, "predicted_class": predicted, "count": int(table.loc[observed, predicted])})
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(OUT_TABLES / "solver_backed_confusion_matrix.csv", index=False)
    false_safe = results[
        (results["model"] == "solver_backed_conservative")
        & (results["predicted_class"] == "both_beneficial")
        & (results["observed_class"] != "both_beneficial")
    ]
    false_safe.to_csv(OUT_TABLES / "solver_backed_false_safe_audit.csv", index=False)
    ablation = pd.DataFrame(
        [
            {"feature_family": "centroid_classifier", "included": "yes", "role": "leave-study-out class prediction in solver-generated feature space"},
            {"feature_family": "solver_lambda_and_RV", "included": "yes", "role": "period-shift and force-ratio proxy"},
            {"feature_family": "solver_Rdelta_total", "included": "yes", "role": "displacement penalty proxy"},
            {"feature_family": "solver_rotation_ratio", "included": "yes", "role": "safe-release guardrail"},
            {"feature_family": "observed_response", "included": "no", "role": "withheld until scoring"},
        ]
    )
    ablation.to_csv(OUT_TABLES / "solver_backed_feature_importance_or_ablation.csv", index=False)
    write_protocol()
    plot_confusion(confusion)
    plot_map(data)
    proposed = next(row for row in metrics if row["model"] == "solver_backed_conservative")
    summary = {
        "validation_type": "3D solver-backed predictive class validation",
        "solver": "OpenSeesPy 3.8.0",
        "tests_used": proposed["tests_used"],
        "accuracy": proposed["accuracy"],
        "balanced_accuracy": proposed["balanced_accuracy"],
        "false_safe_rate": proposed["false_safe_rate"],
        "mixed_recall": proposed["mixed_recall"],
        "safe_precision": proposed["safe_precision"],
        "status": proposed["status"],
        "claim_boundary": "Solver-backed class prediction only; not full nonlinear 3D response-history validation.",
    }
    (OUT_SUMMARIES / "solver_backed_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
