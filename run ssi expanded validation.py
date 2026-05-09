from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path(r"C:\Users\gjm31\OneDrive\Escritorio\116 avo art\evidence_buee_20260509")
OUT_DIR = BASE / "expanded_validation"
SPECTRAL_DIR = BASE / "spectrum_validation"


def synthetic_spectrum(periods: np.ndarray, tb: float, tc: float, td: float, amp: float, tail: float) -> np.ndarray:
    sa = np.zeros_like(periods)
    for i, t in enumerate(periods):
        if t < tb:
            sa[i] = amp * (0.35 + 0.65 * t / tb)
        elif t < tc:
            sa[i] = amp
        elif t < td:
            sa[i] = amp * (tc / t) ** tail
        else:
            sa[i] = amp * (tc / td) ** tail * (td / t) ** 2
    return sa


def local_slope(periods: np.ndarray, sa: np.ndarray, t: float) -> float:
    lo = max(periods[0], t / 1.08)
    hi = min(periods[-1], t * 1.08)
    s1 = float(np.interp(lo, periods, sa))
    s2 = float(np.interp(hi, periods, sa))
    return math.log(max(s2, 1e-12) / max(s1, 1e-12)) / math.log(hi / lo)


def make_confusion_image(summary: dict[str, int], out: Path) -> None:
    img = Image.new("RGB", (620, 360), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        bold = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        bold = font
    draw.text((40, 22), "Expanded validation response categories", font=bold, fill=(20, 20, 20))
    labels = [
        ("force_benefit_displacement_benefit", "Force improves / displacement improves", (70, 150, 90)),
        ("force_benefit_displacement_penalty", "Force improves / displacement worsens", (220, 165, 50)),
        ("force_penalty_displacement_benefit", "Force worsens / displacement improves", (90, 120, 200)),
        ("force_penalty_displacement_penalty", "Force worsens / displacement worsens", (205, 80, 65)),
    ]
    total = max(1, sum(summary.values()))
    y = 70
    for key, label, color in labels:
        pct = 100.0 * summary.get(key, 0) / total
        draw.rectangle([45, y, 45 + int(460 * pct / 100.0), y + 28], fill=color)
        draw.rectangle([45, y, 505, y + 28], outline=(80, 80, 80))
        draw.text((55, y + 5), f"{label}: {pct:.1f}% ({summary.get(key,0)})", font=font, fill=(0, 0, 0))
        y += 54
    img.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods = np.linspace(0.05, 3.0, 180)
    spectrum_defs = [
        ("code_short_plateau", 0.10, 0.45, 1.5, 1.00, 1.0),
        ("code_medium_plateau", 0.15, 0.60, 2.0, 1.00, 1.0),
        ("code_long_plateau", 0.20, 0.90, 2.5, 1.00, 1.0),
        ("soft_soil_long_period", 0.25, 1.10, 2.8, 1.10, 0.8),
        ("stiff_soil_steep_decay", 0.08, 0.35, 1.4, 0.95, 1.35),
        ("near_fault_broadband", 0.12, 0.80, 2.2, 1.25, 0.85),
        ("plateau_dominant", 0.10, 1.20, 2.4, 1.00, 0.65),
        ("descending_dominant", 0.12, 0.30, 1.8, 1.00, 1.55),
        ("long_period_sensitive", 0.20, 0.70, 3.0, 0.90, 0.70),
        ("high_curvature_corner", 0.07, 0.25, 1.0, 1.15, 1.8),
        ("moderate_reference", 0.15, 0.60, 2.0, 0.85, 1.0),
        ("low_amplitude_reference", 0.15, 0.60, 2.0, 0.65, 1.0),
    ]
    spectra = {name: synthetic_spectrum(periods, tb, tc, td, amp, tail) for name, tb, tc, td, amp, tail in spectrum_defs}

    # Add direct-validation rows generated from actual records.
    existing_rows = []
    direct_csv = SPECTRAL_DIR / "direct_spectrum_validation.csv"
    if direct_csv.exists():
        with direct_csv.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    ts_values = [0.2, 0.5, 1.0, 1.5]
    eta_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    damping_gain_values = [0.0, 0.05, 0.10, 0.15]
    kappa_values = [1.0, 0.95, 0.90, 0.85]
    ptheta_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    rows = []
    for source, sa in spectra.items():
        for ts in ts_values:
            sa_fixed = float(np.interp(ts, periods, sa))
            n = local_slope(periods, sa, ts)
            for eta in eta_values:
                tssi = min(periods[-1], ts * math.sqrt(1.0 + eta))
                for dg in damping_gain_values:
                    damping_factor = max(0.45, 1.0 - dg * eta / (1.0 + eta))
                    for kappa in kappa_values:
                        direct_rv = kappa * float(np.interp(tssi, periods, sa)) / max(sa_fixed, 1e-12) * damping_factor
                        approx_rv = kappa * (math.sqrt(1.0 + eta) ** n) * damping_factor
                        direct_rd = (1.0 + eta) * direct_rv
                        for ptheta in ptheta_values:
                            rows.append(
                                {
                                    "source_type": "synthetic_spectrum",
                                    "source": source,
                                    "Ts": ts,
                                    "eta": eta,
                                    "ptheta": ptheta,
                                    "kappa": kappa,
                                    "damping_gain": dg,
                                    "local_slope_n": n,
                                    "direct_Rv": direct_rv,
                                    "approx_Rv": approx_rv,
                                    "direct_Rd": direct_rd,
                                    "force_class_direct": "beneficial" if direct_rv < 1.0 else "detrimental_or_neutral",
                                    "force_class_approx": "beneficial" if approx_rv < 1.0 else "detrimental_or_neutral",
                                    "displacement_class_direct": "beneficial" if direct_rd < 1.0 else "detrimental_or_neutral",
                                    "error_abs": abs(direct_rv - approx_rv),
                                }
                            )
    for r in existing_rows:
        direct_rv = float(r["direct_Rv"])
        approx_rv = float(r["approx_Rv"])
        direct_rd = float(r["direct_Rd"])
        rows.append(
            {
                "source_type": "public_record_or_code_spectrum",
                "source": r["source"],
                "Ts": r["Ts"],
                "eta": r["eta"],
                "ptheta": r["ptheta"],
                "kappa": r["kappa"],
                "damping_gain": r["damping_gain"],
                "local_slope_n": r["local_slope_n"],
                "direct_Rv": direct_rv,
                "approx_Rv": approx_rv,
                "direct_Rd": direct_rd,
                "force_class_direct": r["force_class_direct"],
                "force_class_approx": r["force_class_approx"],
                "displacement_class_direct": r["displacement_class_direct"],
                "error_abs": abs(direct_rv - approx_rv),
            }
        )

    fieldnames = list(rows[0].keys())
    with (OUT_DIR / "expanded_validation_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_source = []
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        total = len(subset)
        ok = sum(1 for r in subset if r["force_class_direct"] == r["force_class_approx"])
        errs = [float(r["error_abs"]) for r in subset]
        per_source.append(
            {
                "source": source,
                "cases": total,
                "classification_accuracy_percent": 100.0 * ok / total,
                "mean_abs_error": float(np.mean(errs)),
                "max_abs_error": float(np.max(errs)),
                "error_lt_0_10_percent": 100.0 * sum(e < 0.10 for e in errs) / total,
                "error_lt_0_15_percent": 100.0 * sum(e < 0.15 for e in errs) / total,
                "error_lt_0_20_percent": 100.0 * sum(e < 0.20 for e in errs) / total,
                "error_lt_0_25_percent": 100.0 * sum(e < 0.25 for e in errs) / total,
            }
        )
    with (OUT_DIR / "expanded_validation_by_source.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_source[0].keys()))
        writer.writeheader()
        writer.writerows(per_source)

    confusion = {
        "force_benefit_displacement_benefit": 0,
        "force_benefit_displacement_penalty": 0,
        "force_penalty_displacement_benefit": 0,
        "force_penalty_displacement_penalty": 0,
    }
    for r in rows:
        f_b = float(r["direct_Rv"]) < 1.0
        d_b = float(r["direct_Rd"]) < 1.0
        if f_b and d_b:
            confusion["force_benefit_displacement_benefit"] += 1
        elif f_b and not d_b:
            confusion["force_benefit_displacement_penalty"] += 1
        elif not f_b and d_b:
            confusion["force_penalty_displacement_benefit"] += 1
        else:
            confusion["force_penalty_displacement_penalty"] += 1
    total = len(rows)
    errs = [float(r["error_abs"]) for r in rows]
    summary = {
        "total_cases": total,
        "sources_total": len({r["source"] for r in rows}),
        "synthetic_spectrum_sources": len(spectrum_defs),
        "public_record_or_code_sources": len({r["source"] for r in rows if r["source_type"] == "public_record_or_code_spectrum"}),
        "classification_accuracy_percent": 100.0 * sum(r["force_class_direct"] == r["force_class_approx"] for r in rows) / total,
        "mean_abs_force_ratio_error": float(np.mean(errs)),
        "max_abs_force_ratio_error": float(np.max(errs)),
        "error_lt_0_10_percent": 100.0 * sum(e < 0.10 for e in errs) / total,
        "error_lt_0_15_percent": 100.0 * sum(e < 0.15 for e in errs) / total,
        "error_lt_0_20_percent": 100.0 * sum(e < 0.20 for e in errs) / total,
        "error_lt_0_25_percent": 100.0 * sum(e < 0.25 for e in errs) / total,
        "confusion_counts": confusion,
        "notes": "Expanded suite combines 12 synthetic code-compatible spectral shapes with the direct public-record/code-spectrum validation already generated. Synthetic spectra are declared as reproducible stress tests, not as real ground-motion records.",
    }
    (OUT_DIR / "expanded_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_confusion_image(confusion, OUT_DIR / "expanded_validation_confusion.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
