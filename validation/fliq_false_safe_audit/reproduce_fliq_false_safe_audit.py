from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw_redistributed" / "FLIQ"
REGEN_TABLES = ROOT / "outputs" / "regenerated" / "tables"
REGEN_SUMMARIES = ROOT / "outputs" / "regenerated" / "summaries"


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - s.mean()) / sd


def metrics(df: pd.DataFrame, pred_flag_col: str, obs_col: str = "observed_settlement_rotation_penalty") -> dict:
    pred_flag = df[pred_flag_col].fillna(False).astype(bool)
    obs_unsafe = df[obs_col].fillna(False).astype(bool)
    released = ~pred_flag
    false_safe = released & obs_unsafe
    return {
        "n": int(len(df)),
        "flagged_fraction": float(pred_flag.mean()),
        "released_fraction": float(released.mean()),
        "false_safe_rate_all": float(false_safe.mean()),
        "false_safe_rate_released": float(false_safe.sum() / max(released.sum(), 1)),
        "unsafe_recall": float((pred_flag & obs_unsafe).sum() / max(obs_unsafe.sum(), 1)),
        "safe_precision": float((released & ~obs_unsafe).sum() / max(released.sum(), 1)),
    }


def load_fliq() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "FLIQ_MainSpreadsheet_unformatted.csv", encoding="latin1")
    names = [str(value).strip() for value in raw.iloc[1].values]
    seen: dict[str, int] = {}
    unique_names = []
    for name in names:
        count = seen.get(name, 0)
        unique_names.append(name if count == 0 else f"{name}_{count + 1}")
        seen[name] = count + 1
    df = raw.iloc[2:].copy()
    df.columns = unique_names
    return df.rename(columns={df.columns[0]: "row_id"})


def main() -> None:
    REGEN_TABLES.mkdir(parents=True, exist_ok=True)
    REGEN_SUMMARIES.mkdir(parents=True, exist_ok=True)
    df = load_fliq()
    numeric_cols = [
        "Nscale", "Visc", "Water", "H L1", "Dr L1", "H L2", "Dr L2", "H L3", "Dr_L3",
        "W", "L", "T", "H", "H_Cm", "H_Cd", "M_Struct", "M_Deck", "M_Col", "M_Foot",
        "Embed", "Q", "T_Fb", "Amp", "IF", "RPM", "PBA", "PBV", "CAV5", "CAVstd",
        "Ia_s", "SIR", "PGA_s 1", "PGA_s 2", "PGA_s 3", "PGA_s 4", "Acc No", "PGV_s",
        "CAV5_s", "CAVstd_s", "SIR_s", "Set", "cSet", "Rot", "cRot",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    work = df[df["Struct"].notna() & (df["Struct"].astype(str).str.upper() != "FF")].copy()
    pga_cols = [c for c in ["PGA_s 1", "PGA_s 2", "PGA_s 3", "PGA_s 4"] if c in work.columns]
    work["PGA_surface_max"] = work[pga_cols].max(axis=1, skipna=True)
    work["surface_to_base_pga_ratio"] = work["PGA_surface_max"] / work["PBA"].replace(0, np.nan)
    usable = work[
        work["PBA"].notna()
        & work["PGA_surface_max"].notna()
        & (work["PBA"] > 0)
        & (work["Set"].notna() | work["cSet"].notna() | work["Rot"].notna() | work["cRot"].notna())
    ].copy()

    usable["apparent_acceleration_benefit"] = usable["surface_to_base_pga_ratio"] < 1.0
    set_cut = float(usable["cSet"].abs().quantile(0.75))
    rot_cut = float(usable["cRot"].abs().quantile(0.75))
    usable["high_cumulative_settlement"] = usable["cSet"].abs() >= set_cut
    usable["high_cumulative_rotation"] = usable["cRot"].abs() >= rot_cut
    usable["observed_settlement_rotation_penalty"] = usable[
        ["high_cumulative_settlement", "high_cumulative_rotation"]
    ].any(axis=1)
    usable["observed_false_safe_if_acceleration_only"] = (
        usable["apparent_acceleration_benefit"] & usable["observed_settlement_rotation_penalty"]
    )

    usable["baseline_always_flag"] = True
    usable["baseline_intensity_only"] = zscore(usable["PBA"]) + zscore(usable["PBV"]) + zscore(usable["Ia_s"]) >= 0
    usable["baseline_structure_only"] = zscore(usable["Q"]) + zscore(usable["T_Fb"]) + zscore(usable["Embed"]) >= 0
    usable["risk_score"] = (
        zscore(usable["PBA"]) + zscore(usable["PBV"]) + zscore(usable["Ia_s"])
        + 0.5 * zscore(usable["Q"]) + 0.25 * zscore(usable["T_Fb"]) + 0.25 * zscore(usable["Embed"])
        - 0.25 * zscore(usable["Dr L1"]) - 0.25 * zscore(usable["Dr L2"])
    )

    candidates = []
    for q in np.linspace(0.05, 0.95, 19):
        threshold = float(usable["risk_score"].quantile(q))
        col = f"risk_ge_q_{q:.2f}"
        usable[col] = usable["risk_score"] >= threshold
        item = metrics(usable, col)
        item["quantile_threshold"] = float(q)
        item["risk_score_threshold"] = threshold
        candidates.append(item)
    feasible = [m for m in candidates if m["false_safe_rate_all"] <= 0.10]
    best = (
        sorted(feasible, key=lambda m: (-m["released_fraction"], -m["unsafe_recall"]))[0]
        if feasible
        else sorted(candidates, key=lambda m: (m["false_safe_rate_all"], -m["released_fraction"]))[0]
    )
    usable["proposed_pre_response_guardrail"] = usable["risk_score"] >= best["risk_score_threshold"]

    summary = {
        "dataset": "FLIQ foundation and ground performance in liquefaction experiments",
        "doi": "10.4231/D3M61BQ73",
        "raw_rows": int(len(df)),
        "structural_rows_usable": int(len(usable)),
        "apparent_acceleration_benefit_cases": int(usable["apparent_acceleration_benefit"].sum()),
        "settlement_rotation_penalty_cases": int(usable["observed_settlement_rotation_penalty"].sum()),
        "acceleration_only_false_safe_cases": int(usable["observed_false_safe_if_acceleration_only"].sum()),
        "acceleration_only_false_safe_rate_all": float(usable["observed_false_safe_if_acceleration_only"].mean()),
        "acceleration_only_false_safe_rate_among_apparent_benefit": float(
            usable["observed_false_safe_if_acceleration_only"].sum()
            / max(usable["apparent_acceleration_benefit"].sum(), 1)
        ),
        "response_thresholds": {
            "cSet_abs_q75_mm": set_cut,
            "cRot_abs_q75_rad": rot_cut,
        },
        "baseline_always_flag": metrics(usable, "baseline_always_flag"),
        "baseline_intensity_only": metrics(usable, "baseline_intensity_only"),
        "baseline_structure_only": metrics(usable, "baseline_structure_only"),
        "proposed_pre_response_guardrail": metrics(usable, "proposed_pre_response_guardrail"),
        "selected_guardrail_threshold": best,
        "claim_boundary": (
            "FLIQ is an external liquefaction-oriented false-safe audit. It tests "
            "whether apparent acceleration-side benefit can coexist with settlement "
            "or rotation penalties; it is not nonlinear 3D response-history validation."
        ),
    }

    row_file = "fliq_false_safe_audit_rows.csv"
    sweep_file = "fliq_guardrail_threshold_sweep.csv"
    summary_file = "fliq_false_safe_probe_summary.json"
    usable.to_csv(REGEN_TABLES / row_file, index=False)
    pd.DataFrame(candidates).to_csv(REGEN_TABLES / sweep_file, index=False)
    (REGEN_SUMMARIES / summary_file).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
