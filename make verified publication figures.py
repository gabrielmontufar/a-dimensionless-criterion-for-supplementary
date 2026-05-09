from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(r"C:\Users\gjm31\OneDrive\Escritorio\116 avo art\evidence_buee_20260509")
FIG_DIR = OUT_DIR / "verified_publication_figures"


def force_ratio(eta: np.ndarray, n: np.ndarray, damping_gain: float = 0.0) -> np.ndarray:
    period_factor = np.sqrt(1.0 + eta)
    damping_factor = np.maximum(0.45, 1.0 - damping_gain * eta / (1.0 + eta))
    return np.power(period_factor, n) * damping_factor


def save_heatmap(
    values: np.ndarray,
    etas: np.ndarray,
    slopes: np.ndarray,
    label: str,
    title: str,
    output: Path,
    cmap: str,
    levels: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    mesh = ax.contourf(etas, slopes, values, levels=levels, cmap=cmap, extend="both")
    ax.contour(etas, slopes, values, levels=[1.0], colors="black", linewidths=1.35)
    ax.text(1.55, 0.72, "ratio > 1", ha="center", va="center", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.35"})
    ax.text(1.55, -2.25, "ratio < 1", ha="center", va="center", fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.35"})
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(r"Total flexibility, $\eta$")
    ax.set_ylabel(r"Spectral slope, $n$")
    ax.set_xlim(float(etas.min()), float(etas.max()))
    ax.set_ylim(float(slopes.min()), float(slopes.max()))
    ax.grid(color="white", linewidth=0.45, alpha=0.35)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(label)
    ax.annotate("transition ratio = 1", xy=(1.25, -0.05), xytext=(0.55, -0.72),
                arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    etas_1d = np.linspace(0.0, 2.0, 241)
    slopes_1d = np.linspace(-3.0, 1.0, 241)
    etas, slopes = np.meshgrid(etas_1d, slopes_1d)
    rv = force_ratio(etas, slopes, damping_gain=0.0)
    rd = (1.0 + etas) * rv

    save_heatmap(
        rv,
        etas,
        slopes,
        r"Force-demand ratio, $R_v$",
        r"Dimensionless force-demand ratio, $R_v$",
        FIG_DIR / "figure_1_verified_force_ratio.png",
        "RdYlBu_r",
        np.linspace(0.55, 1.75, 25),
    )
    save_heatmap(
        rd,
        etas,
        slopes,
        r"Total-displacement ratio, $R_d$",
        r"Dimensionless total-displacement ratio, $R_d$",
        FIG_DIR / "figure_2_verified_displacement_ratio.png",
        "PuOr_r",
        np.linspace(0.75, 5.25, 25),
    )

    report = {
        "figure_1": str(FIG_DIR / "figure_1_verified_force_ratio.png"),
        "figure_2": str(FIG_DIR / "figure_2_verified_displacement_ratio.png"),
        "eta_range": [float(etas_1d.min()), float(etas_1d.max())],
        "spectral_slope_range": [float(slopes_1d.min()), float(slopes_1d.max())],
        "damping_gain": 0.0,
        "force_formula": "Rv = (sqrt(1 + eta))**n",
        "displacement_formula": "Rd = (1 + eta) * Rv",
        "chatgpt_image_screening": {
            "accepted_for_replacement": False,
            "reason": "Generated force-ratio heatmap inverted the validated benchmark pattern.",
        },
    }
    (FIG_DIR / "figure_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
