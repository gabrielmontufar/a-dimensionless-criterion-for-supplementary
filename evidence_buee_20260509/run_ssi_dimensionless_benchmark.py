from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(r"C:\Users\gjm31\OneDrive\Escritorio\116 avo art\evidence_buee_20260509")


def color_scale(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    value = max(vmin, min(vmax, value))
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    if t < 0.5:
        u = t / 0.5
        r = int(40 + 180 * u)
        g = int(90 + 130 * u)
        b = int(180 - 120 * u)
    else:
        u = (t - 0.5) / 0.5
        r = int(220 + 35 * u)
        g = int(220 - 130 * u)
        b = int(60 - 20 * u)
    return r, g, b


def draw_heatmap(matrix: np.ndarray, xs: np.ndarray, ys: np.ndarray, title: str, out: Path) -> None:
    cell = 18
    left = 82
    top = 52
    right = 30
    bottom = 70
    width = left + len(xs) * cell + right
    height = top + len(ys) * cell + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_bold = ImageFont.truetype("arialbd.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    draw.text((left, 16), title, fill=(20, 20, 20), font=font_bold)
    vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            value = float(matrix[iy, ix])
            x0 = left + ix * cell
            y0 = top + iy * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color_scale(value, vmin, vmax))
            if abs(value - 1.0) < 0.025:
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(0, 0, 0), width=2)

    for ix, x in enumerate(xs):
        if ix % 3 == 0:
            draw.text((left + ix * cell - 4, top + len(ys) * cell + 8), f"{x:.1f}", fill=(0, 0, 0), font=font)
    for iy, y in enumerate(ys):
        if iy % 3 == 0:
            draw.text((18, top + iy * cell + 2), f"{y:.1f}", fill=(0, 0, 0), font=font)

    draw.text((left + 60, height - 32), "total flexibility eta", fill=(0, 0, 0), font=font)
    draw.text((8, 30), "spectral slope n", fill=(0, 0, 0), font=font)
    draw.text((left, height - 52), "black boxes mark the transition region around response ratio = 1", fill=(60, 60, 60), font=font)
    img.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    etas = np.round(np.linspace(0.0, 2.0, 21), 3)
    slopes = np.round(np.linspace(-3.0, 1.0, 21), 3)
    damping_gain = [0.00, 0.05, 0.10, 0.15]
    kappa = 1.0
    zeta_s = 0.05
    rows = []
    for d in damping_gain:
        for n in slopes:
            for eta in etas:
                period_factor = math.sqrt(1.0 + eta)
                spectral_factor = period_factor**n
                damping_factor = max(0.45, 1.0 - d * eta / (1.0 + eta))
                rv = kappa * spectral_factor * damping_factor
                rd = (1.0 + eta) * rv
                rows.append(
                    {
                        "eta": eta,
                        "spectral_slope_n": n,
                        "damping_gain": d,
                        "period_factor": period_factor,
                        "force_ratio_Rv": rv,
                        "total_displacement_ratio_Rd": rd,
                        "force_class": "beneficial" if rv < 1.0 else "detrimental_or_neutral",
                        "displacement_class": "beneficial" if rd < 1.0 else "detrimental_or_neutral",
                    }
                )

    csv_path = OUT_DIR / "dimensionless_ssi_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for d in damping_gain:
        subset = [r for r in rows if r["damping_gain"] == d]
        total = len(subset)
        force_b = sum(1 for r in subset if r["force_ratio_Rv"] < 1.0)
        disp_b = sum(1 for r in subset if r["total_displacement_ratio_Rd"] < 1.0)
        mixed = sum(1 for r in subset if r["force_ratio_Rv"] < 1.0 and r["total_displacement_ratio_Rd"] >= 1.0)
        summary_rows.append(
            {
                "damping_gain": d,
                "grid_points": total,
                "force_beneficial_percent": 100.0 * force_b / total,
                "total_displacement_beneficial_percent": 100.0 * disp_b / total,
                "mixed_force_benefit_displacement_penalty_percent": 100.0 * mixed / total,
            }
        )
    summary_path = OUT_DIR / "dimensionless_ssi_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for d in [0.0, 0.1]:
        force = np.zeros((len(slopes), len(etas)))
        disp = np.zeros((len(slopes), len(etas)))
        for iy, n in enumerate(slopes):
            for ix, eta in enumerate(etas):
                period_factor = math.sqrt(1.0 + eta)
                damping_factor = max(0.45, 1.0 - d * eta / (1.0 + eta))
                force[iy, ix] = period_factor**n * damping_factor
                disp[iy, ix] = (1.0 + eta) * force[iy, ix]
        draw_heatmap(force, etas, slopes, f"Force demand ratio Rv, damping gain={d:.2f}", OUT_DIR / f"heatmap_force_ratio_d{d:.2f}.png")
        draw_heatmap(disp, etas, slopes, f"Total displacement ratio Rd, damping gain={d:.2f}", OUT_DIR / f"heatmap_displacement_ratio_d{d:.2f}.png")

    metadata = {
        "model": "dimensionless SDOF screening model",
        "eta_values": [float(etas.min()), float(etas.max()), len(etas)],
        "spectral_slope_values": [float(slopes.min()), float(slopes.max()), len(slopes)],
        "damping_gain_values": damping_gain,
        "grid_points": len(rows),
        "force_ratio": "Rv = (sqrt(1+eta))**n * max(0.45, 1 - damping_gain*eta/(1+eta))",
        "displacement_ratio": "Rd = (1+eta)*Rv",
    }
    (OUT_DIR / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT_DIR), "rows": len(rows), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
