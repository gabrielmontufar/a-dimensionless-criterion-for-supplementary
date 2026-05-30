from __future__ import annotations

import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = ROOT / "validation" / "experimental_database_validation" / "reproduce_experimental_validation.py"
PREDICTIVE_SCRIPT = ROOT / "validation" / "independent_predictive_validation" / "reproduce_predictive_validation.py"
GROUP_UNCERTAINTY_SCRIPT = (
    ROOT / "validation" / "independent_predictive_validation" / "reproduce_group_uncertainty_audit.py"
)
DECISION_VALUE_SCRIPT = ROOT / "validation" / "decision_value_validation" / "reproduce_decision_value_validation.py"
GUARDRAIL_SCRIPT = ROOT / "validation" / "failure_mode_guardrails" / "reproduce_failure_mode_guardrails.py"
FLIQ_SCRIPT = ROOT / "validation" / "fliq_false_safe_audit" / "reproduce_fliq_false_safe_audit.py"
SOLVER_BACKED_SCRIPT = (
    ROOT / "validation" / "solver_backed_class_prediction" / "reproduce_solver_backed_class_prediction.py"
)
PUBLISHED = ROOT / "outputs" / "published"
REGENERATED = ROOT / "outputs" / "regenerated"


TABLES = [
    "experimental_claim_boundary_matrix.csv",
    "experimental_class_confusion_matrix.csv",
    "experimental_polarity_points.csv",
    "experimental_validation_metrics_by_dataset.csv",
    "false_safe_case_audit.csv",
    "fordy_forcy_master_validation_table.csv",
    "offline_data_manifest.csv",
    "proxy_threshold_sensitivity_metrics.csv",
    "strict_screening_metrics_by_dataset.csv",
    "baseline_comparison.csv",
    "false_safe_audit_predictive.csv",
    "grouped_bootstrap_uncertainty.csv",
    "grouped_metric_by_study.csv",
    "leave_study_out_summary.csv",
    "paired_group_baseline_comparison.csv",
    "predictive_validation_confusion_matrix.csv",
    "predictive_validation_results.csv",
    "proxy_decision_value_summary.csv",
    "predictive_decision_value_summary.csv",
    "loss_ratio_sweep.csv",
    "selective_abstention_curve.csv",
    "failure_mode_cases.csv",
    "failure_mode_summary.csv",
    "guardrail_candidate_summary.csv",
    "guardrail_application_summary.csv",
    "fliq_false_safe_audit_rows.csv",
    "fliq_guardrail_threshold_sweep.csv",
    "solver_backed_baseline_comparison.csv",
    "solver_backed_class_prediction_results.csv",
    "solver_backed_confusion_matrix.csv",
    "solver_backed_false_safe_audit.csv",
    "solver_backed_feature_importance_or_ablation.csv",
    "solver_backed_feature_table.csv",
    "solver_backed_metrics_by_fold.csv",
    "solver_backed_protocol.csv",
]


NUMERIC_TOL_ABS = 1e-9


def normalized_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def values_match(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= NUMERIC_TOL_ABS


def compare_csv(published: Path, regenerated: Path) -> str | None:
    pub_rows = normalized_rows(published)
    reg_rows = normalized_rows(regenerated)
    if len(pub_rows) != len(reg_rows):
        return f"row count mismatch {published.name}: {len(pub_rows)} != {len(reg_rows)}"
    if not pub_rows and not reg_rows:
        return None
    pub_fields = list(pub_rows[0].keys())
    reg_fields = list(reg_rows[0].keys())
    if pub_fields != reg_fields:
        return f"column mismatch {published.name}: {pub_fields} != {reg_fields}"
    for i, (pub, reg) in enumerate(zip(pub_rows, reg_rows), start=2):
        for field in pub_fields:
            if not values_match(pub[field], reg[field]):
                return f"value mismatch {published.name}:{i}:{field}: {pub[field]} != {reg[field]}"
    return None


def compare_required_tables() -> list[str]:
    mismatches: list[str] = []
    for name in TABLES:
        published = PUBLISHED / "tables" / name
        regenerated = REGENERATED / "tables" / name
        if not published.exists() or not regenerated.exists():
            mismatches.append(f"missing {name}")
        else:
            mismatch = compare_csv(published, regenerated)
            if mismatch:
                mismatches.append(mismatch)
    return mismatches


def main() -> None:
    if REGENERATED.exists():
        shutil.rmtree(REGENERATED)
    (REGENERATED / "tables").mkdir(parents=True, exist_ok=True)
    (REGENERATED / "figures").mkdir(parents=True, exist_ok=True)
    (REGENERATED / "summaries").mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, str(VALIDATION_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(PREDICTIVE_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(GROUP_UNCERTAINTY_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(DECISION_VALUE_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(GUARDRAIL_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(FLIQ_SCRIPT)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(SOLVER_BACKED_SCRIPT)], cwd=ROOT, check=True)
    mismatches = compare_required_tables()
    if mismatches:
        print("REPRODUCIBILITY_CORE: FAIL")
        for item in mismatches:
            print(item)
        raise SystemExit(1)
    if not (REGENERATED / "figures" / "demand_polarity_map_experimental.png").exists():
        raise SystemExit("Missing regenerated demand-polarity figure")
    if not (REGENERATED / "figures" / "predictive_confusion_matrix.png").exists():
        raise SystemExit("Missing regenerated predictive confusion matrix")
    if not (REGENERATED / "figures" / "loss_ratio_sweep.png").exists():
        raise SystemExit("Missing regenerated loss-ratio decision-value figure")
    if not (REGENERATED / "figures" / "selective_abstention_curve.png").exists():
        raise SystemExit("Missing regenerated selective-abstention decision-value figure")
    if not (REGENERATED / "figures" / "failure_mode_map_lambda_rv.png").exists():
        raise SystemExit("Missing regenerated failure-mode atlas figure")
    if not (REGENERATED / "figures" / "guardrail_tradeoff.png").exists():
        raise SystemExit("Missing regenerated guardrail tradeoff figure")
    if not (REGENERATED / "figures" / "solver_backed_confusion_matrix.png").exists():
        raise SystemExit("Missing regenerated solver-backed confusion matrix")
    if not (REGENERATED / "figures" / "solver_backed_demand_map.png").exists():
        raise SystemExit("Missing regenerated solver-backed demand map")
    print("REPRODUCIBILITY_CORE: PASS")


if __name__ == "__main__":
    main()
