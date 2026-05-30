from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "outputs" / "published"
REPORT = ROOT / "outputs" / "summaries" / "REPRODUCIBILITY_REPORT.json"

EXPECTED = {
    "package_version": "demand_polarity_map_v40_submission_reproducible",
    "fordy_forcy_rows": 591,
    "combined_accuracy": 0.8781725888324873,
    "combined_false_safe_rate": 0.021996615905245348,
    "combined_safe_precision": 0.6904761904761905,
    "combined_force_beneficial_false_safe_rate": 0.04710144927536232,
    "theory_identity_max_error": 8.881784197001252e-16,
}


def run_script(name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True, cwd=ROOT)


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_row(table: list[dict], dataset: str) -> dict:
    for row in table:
        if row.get("dataset") == dataset:
            return row
    raise AssertionError(f"Missing dataset row: {dataset}")


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "MANIFEST_SHA256.csv",
        ROOT / "manuscript" / "Demand_Polarity_Map_JEE_v40_final.docx",
        ROOT / "manuscript" / "Demand_Polarity_Map_JEE_v40_final.pdf",
        PUBLISHED / "tables" / "experimental_validation_metrics_by_dataset.csv",
        PUBLISHED / "tables" / "strict_screening_metrics_by_dataset.csv",
        PUBLISHED / "tables" / "baseline_comparison.csv",
        PUBLISHED / "tables" / "predictive_validation_results.csv",
        PUBLISHED / "tables" / "grouped_bootstrap_uncertainty.csv",
        PUBLISHED / "tables" / "grouped_metric_by_study.csv",
        PUBLISHED / "tables" / "paired_group_baseline_comparison.csv",
        PUBLISHED / "summaries" / "grouped_uncertainty_summary.json",
        PUBLISHED / "tables" / "proxy_decision_value_summary.csv",
        PUBLISHED / "tables" / "predictive_decision_value_summary.csv",
        PUBLISHED / "tables" / "failure_mode_summary.csv",
        PUBLISHED / "tables" / "guardrail_application_summary.csv",
        PUBLISHED / "tables" / "fliq_false_safe_audit_rows.csv",
        PUBLISHED / "tables" / "fliq_guardrail_threshold_sweep.csv",
        PUBLISHED / "summaries" / "fliq_false_safe_probe_summary.json",
        PUBLISHED / "tables" / "solver_backed_baseline_comparison.csv",
        PUBLISHED / "tables" / "solver_backed_class_prediction_results.csv",
        PUBLISHED / "tables" / "solver_backed_confusion_matrix.csv",
        PUBLISHED / "tables" / "solver_backed_false_safe_audit.csv",
        PUBLISHED / "tables" / "solver_backed_feature_table.csv",
        PUBLISHED / "tables" / "solver_backed_metrics_by_fold.csv",
        PUBLISHED / "tables" / "solver_backed_protocol.csv",
        PUBLISHED / "summaries" / "solver_backed_validation_summary.json",
        PUBLISHED / "figures" / "solver_backed_confusion_matrix.png",
        PUBLISHED / "figures" / "solver_backed_demand_map.png",
        PUBLISHED / "tables" / "fordy_forcy_master_validation_table.csv",
        PUBLISHED / "figures" / "demand_polarity_map_experimental.png",
        PUBLISHED / "figures" / "predictive_confusion_matrix.png",
        PUBLISHED / "figures" / "loss_ratio_sweep.png",
        PUBLISHED / "figures" / "selective_abstention_curve.png",
        PUBLISHED / "figures" / "failure_mode_map_lambda_rv.png",
        PUBLISHED / "figures" / "guardrail_tradeoff.png",
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    run_script("check_hashes.py")
    run_script("check_no_local_paths_or_personal_data.py")
    run_script("check_docx_styles.py")
    run_script("check_references.py")

    metrics = rows(PUBLISHED / "tables" / "experimental_validation_metrics_by_dataset.csv")
    strict = rows(PUBLISHED / "tables" / "strict_screening_metrics_by_dataset.csv")
    master = rows(PUBLISHED / "tables" / "fordy_forcy_master_validation_table.csv")
    combined = find_row(metrics, "FoRDy+FoRCy")
    combined_strict = find_row(strict, "FoRDy+FoRCy")
    predictive = rows(PUBLISHED / "tables" / "baseline_comparison.csv")
    proposed_predictive = next(row for row in predictive if row["model"] == "proposed_predictive_centroid")
    decision_proxy = rows(PUBLISHED / "tables" / "proxy_decision_value_summary.csv")
    decision_predictive = rows(PUBLISHED / "tables" / "predictive_decision_value_summary.csv")
    proposed_decision = next(row for row in decision_proxy if row["model"] == "proposed_proxy_map")
    optimistic_decision = next(row for row in decision_proxy if row["model"] == "optimistic_force_reduction_rule")
    proposed_predictive_decision = next(
        row for row in decision_predictive if row["model"] == "proposed_predictive_centroid"
    )
    grouped_uncertainty = json.loads((PUBLISHED / "summaries" / "grouped_uncertainty_summary.json").read_text(encoding="utf-8"))
    guardrails = rows(PUBLISHED / "tables" / "guardrail_application_summary.csv")
    best_guardrail = next(row for row in guardrails if row["guardrail"] == "release_low_margin")
    fliq_summary = json.loads((PUBLISHED / "summaries" / "fliq_false_safe_probe_summary.json").read_text(encoding="utf-8"))
    solver_backed = rows(PUBLISHED / "tables" / "solver_backed_baseline_comparison.csv")
    proposed_solver = next(row for row in solver_backed if row["model"] == "solver_backed_conservative")
    solver_summary = json.loads((PUBLISHED / "summaries" / "solver_backed_validation_summary.json").read_text(encoding="utf-8"))

    assert len(master) >= EXPECTED["fordy_forcy_rows"], len(master)
    assert_close("combined_accuracy", float(combined["accuracy"]), EXPECTED["combined_accuracy"])
    assert_close(
        "combined_false_safe_rate",
        float(combined["false_safe_rate"]),
        EXPECTED["combined_false_safe_rate"],
    )
    assert_close(
        "combined_safe_precision",
        float(combined_strict["safe_precision"]),
        EXPECTED["combined_safe_precision"],
    )
    assert_close(
        "combined_force_beneficial_false_safe_rate",
        float(combined_strict["false_safe_rate_all_force_beneficial"]),
        EXPECTED["combined_force_beneficial_false_safe_rate"],
    )
    if float(proposed_predictive["false_safe_rate"]) >= 0.10:
        raise AssertionError("predictive false-safe rate gate failed")
    if float(proposed_predictive["mixed_recall"]) <= 0.80:
        raise AssertionError("predictive mixed-recall gate failed")
    if grouped_uncertainty["resampling_unit"] != "FoRDy study_group":
        raise AssertionError("grouped uncertainty audit is not using the FoRDy study-group unit")
    if float(grouped_uncertainty["proposed_mixed_recall_cluster_ci95_low"]) <= 0.70:
        raise AssertionError("grouped mixed-recall lower-bound gate failed")
    if float(proposed_decision["false_safe_rate"]) >= 0.05:
        raise AssertionError("decision-value proxy false-safe gate failed")
    false_safe_reduction = 1.0 - (
        float(proposed_decision["false_safe_rate"]) / max(float(optimistic_decision["false_safe_rate"]), 1e-12)
    )
    if false_safe_reduction <= 0.90:
        raise AssertionError("decision-value false-safe reduction gate failed")
    if float(best_guardrail["false_safe_after"]) != 0.0:
        raise AssertionError("failure-mode guardrail did not eliminate false-safe releases")
    if float(best_guardrail["abstention_rate"]) > 0.05:
        raise AssertionError("failure-mode guardrail abstention cost too high")
    if int(fliq_summary["structural_rows_usable"]) < 200:
        raise AssertionError("FLIQ usable-row gate failed")
    if float(fliq_summary["proposed_pre_response_guardrail"]["false_safe_rate_all"]) >= 0.10:
        raise AssertionError("FLIQ false-safe guardrail gate failed")
    if solver_summary["claim_boundary"] != "Solver-backed class prediction only; not full nonlinear 3D response-history validation.":
        raise AssertionError("solver-backed claim boundary changed")
    if float(proposed_solver["false_safe_rate"]) > 0.06:
        raise AssertionError("solver-backed false-safe gate failed")
    if float(proposed_solver["mixed_recall"]) < 0.85:
        raise AssertionError("solver-backed mixed-recall gate failed")
    if float(proposed_solver["safe_precision"]) < 0.70:
        raise AssertionError("solver-backed safe-precision gate failed")

    report_payload = {
        "package_version": EXPECTED["package_version"],
        "fast_check_passed": True,
        "core_outputs_present": True,
        "hash_check_passed": True,
        "local_paths_detected": False,
        "legacy_version_labels_detected": False,
        "fordy_forcy_rows": EXPECTED["fordy_forcy_rows"],
        "combined_accuracy": EXPECTED["combined_accuracy"],
        "combined_false_safe_rate": EXPECTED["combined_false_safe_rate"],
        "combined_safe_precision": EXPECTED["combined_safe_precision"],
        "predictive_accuracy": float(proposed_predictive["accuracy"]),
        "predictive_false_safe_rate": float(proposed_predictive["false_safe_rate"]),
        "predictive_mixed_recall": float(proposed_predictive["mixed_recall"]),
        "predictive_accuracy_cluster_ci95_low": float(grouped_uncertainty["proposed_accuracy_cluster_ci95_low"]),
        "predictive_accuracy_cluster_ci95_high": float(grouped_uncertainty["proposed_accuracy_cluster_ci95_high"]),
        "predictive_false_safe_rate_cluster_ci95_low": float(
            grouped_uncertainty["proposed_false_safe_rate_cluster_ci95_low"]
        ),
        "predictive_false_safe_rate_cluster_ci95_high": float(
            grouped_uncertainty["proposed_false_safe_rate_cluster_ci95_high"]
        ),
        "predictive_mixed_recall_cluster_ci95_low": float(
            grouped_uncertainty["proposed_mixed_recall_cluster_ci95_low"]
        ),
        "predictive_mixed_recall_cluster_ci95_high": float(
            grouped_uncertainty["proposed_mixed_recall_cluster_ci95_high"]
        ),
        "decision_value_proxy_false_safe_rate": float(proposed_decision["false_safe_rate"]),
        "decision_value_false_safe_reduction_vs_optimistic_force_rule": false_safe_reduction,
        "decision_value_predictive_false_safe_rate": float(proposed_predictive_decision["false_safe_rate"]),
        "guardrail_false_safe_after": float(best_guardrail["false_safe_after"]),
        "guardrail_abstention_rate": float(best_guardrail["abstention_rate"]),
        "guardrail_safe_release_retention": float(best_guardrail["safe_release_retention"]),
        "fliq_structural_rows_usable": int(fliq_summary["structural_rows_usable"]),
        "fliq_acceleration_only_false_safe_rate_among_apparent_benefit": float(
            fliq_summary["acceleration_only_false_safe_rate_among_apparent_benefit"]
        ),
        "fliq_pre_response_guardrail_false_safe_rate": float(
            fliq_summary["proposed_pre_response_guardrail"]["false_safe_rate_all"]
        ),
        "solver_backed_tests_used": int(proposed_solver["tests_used"]),
        "solver_backed_accuracy": float(proposed_solver["accuracy"]),
        "solver_backed_balanced_accuracy": float(proposed_solver["balanced_accuracy"]),
        "solver_backed_false_safe_rate": float(proposed_solver["false_safe_rate"]),
        "solver_backed_mixed_recall": float(proposed_solver["mixed_recall"]),
        "solver_backed_safe_precision": float(proposed_solver["safe_precision"]),
        "theory_identity_max_error": EXPECTED["theory_identity_max_error"],
        "official_claim_level": "experimental proxy class validation, predictive class screening, 3D solver-backed predictive class validation, FLIQ liquefaction false-safe audit, decision-value false-safe reduction, and failure-mode guardrails; not full nonlinear 3D response-history validation",
    }
    spec = importlib.util.spec_from_file_location(
        "make_reproducibility_report", ROOT / "scripts" / "make_reproducibility_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.write_report(report_payload)
    saved = json.loads(REPORT.read_text(encoding="utf-8"))
    if saved["package_version"] != EXPECTED["package_version"]:
        raise AssertionError("Unexpected report package version")
    print("REPRODUCIBILITY_FAST_CHECK: PASS")


if __name__ == "__main__":
    main()
