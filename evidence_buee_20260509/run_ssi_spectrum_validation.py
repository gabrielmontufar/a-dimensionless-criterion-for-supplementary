from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE = Path(r"C:\Users\gjm31\OneDrive\Escritorio\116 avo art\evidence_buee_20260509")
RECORD_DIR = BASE / "records"
OUT_DIR = BASE / "spectrum_validation"


def load_record(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            nums = [float(p) for p in parts[:2]]
        except ValueError:
            continue
        values.append(nums)
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] < 10:
        raise ValueError(f"not enough numeric rows in {path}")
    # Most Vibrationdata files are time, acceleration(g). If the first column is
    # an index and the second is time, this heuristic swaps to the two most useful columns.
    t = arr[:, 0]
    a = arr[:, 1]
    if np.nanmedian(np.diff(t[: min(len(t), 100)])) <= 0:
        raise ValueError(f"time column not increasing in {path}")
    return t, a


def response_spectrum(time: np.ndarray, acc_g: np.ndarray, periods: np.ndarray, zeta: float) -> np.ndarray:
    # Average-acceleration Newmark integration for x'' + 2*zeta*w*x' + w^2*x = -ag.
    dt = float(np.median(np.diff(time)))
    ag = acc_g.astype(float)
    out = []
    gamma = 0.5
    beta = 0.25
    for T in periods:
        w = 2.0 * math.pi / T
        k = w * w
        c = 2.0 * zeta * w
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        keff = k + a0 + a1 * c
        u = 0.0
        v = 0.0
        rel_acc = -ag[0] - c * v - k * u
        umax = 0.0
        for i in range(1, len(ag)):
            p_eff = -ag[i] + a0 * u + a2 * v + a3 * rel_acc + c * (a1 * u + a4 * v + a5 * rel_acc)
            u_new = p_eff / keff
            rel_acc_new = a0 * (u_new - u) - a2 * v - a3 * rel_acc
            v_new = v + dt * ((1.0 - gamma) * rel_acc + gamma * rel_acc_new)
            u, v, rel_acc = u_new, v_new, rel_acc_new
            umax = max(umax, abs(u))
        out.append((w * w) * umax)
    return np.asarray(out)


def code_spectrum(periods: np.ndarray) -> np.ndarray:
    # Normalized elastic design spectrum shape with ascending, plateau,
    # constant-velocity and constant-displacement branches.
    tb, tc, td = 0.15, 0.60, 2.0
    sa = np.zeros_like(periods)
    for i, t in enumerate(periods):
        if t < tb:
            sa[i] = 0.4 + 0.6 * t / tb
        elif t < tc:
            sa[i] = 1.0
        elif t < td:
            sa[i] = tc / t
        else:
            sa[i] = tc * td / (t * t)
    return sa


def local_slope(periods: np.ndarray, sa: np.ndarray, t: float) -> float:
    lo = max(periods[0], t / 1.08)
    hi = min(periods[-1], t * 1.08)
    if hi <= lo:
        return 0.0
    s1 = float(np.interp(lo, periods, sa))
    s2 = float(np.interp(hi, periods, sa))
    return math.log(max(s2, 1e-12) / max(s1, 1e-12)) / math.log(hi / lo)


def make_plot(rows: list[dict[str, float | str]], out: Path, title: str, key: str) -> None:
    width, height = 760, 420
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        bold = ImageFont.truetype("arialbd.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        bold = font
    draw.text((30, 18), title, fill=(20, 20, 20), font=bold)
    x0, y0, w, h = 70, 60, 640, 300
    draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 0, 0))
    xs = [float(r["direct_Rv"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    xmax = max(1.8, max(xs) * 1.05)
    ymax = max(1.8, max(ys) * 1.05)
    draw.line([x0, y0 + h, x0 + w, y0], fill=(120, 120, 120), width=1)
    for x, y in zip(xs, ys):
        px = x0 + int(w * x / xmax)
        py = y0 + h - int(h * y / ymax)
        color = (30, 95, 170) if x < 1 else (200, 80, 40)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    draw.text((x0 + 250, y0 + h + 28), "direct spectral ratio", font=font, fill=(0, 0, 0))
    draw.text((15, y0 + 120), key, font=font, fill=(0, 0, 0))
    img.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods = np.linspace(0.05, 3.0, 180)
    zetas = [0.05, 0.075, 0.10, 0.125]
    spectra: dict[tuple[str, float], np.ndarray] = {}
    source_notes = {}

    spectra[("code_spectrum", 0.05)] = code_spectrum(periods)
    source_notes["code_spectrum"] = "normalized code-compatible idealized spectrum"
    for z in zetas[1:]:
        spectra[("code_spectrum", z)] = spectra[("code_spectrum", 0.05)] * (0.05 / z) ** 0.25

    for path in sorted(RECORD_DIR.glob("*.dat")):
        t, a = load_record(path)
        name = path.stem
        source_notes[name] = f"Vibrationdata El Centro file {path.name}; acceleration in g"
        for z in zetas:
            spectra[(name, z)] = response_spectrum(t, a, periods, z)

    ts_values = [0.2, 0.5, 1.0, 1.5]
    eta_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    damping_gain_values = [0.0, 0.05, 0.10, 0.15]
    kappa_values = [1.0, 0.95, 0.90, 0.85]
    ptheta_values = [0.0, 0.25, 0.50, 0.75, 1.0]

    rows = []
    for source in sorted(source_notes):
        base_sa = spectra[(source, 0.05)]
        for ts in ts_values:
            sa_fixed = float(np.interp(ts, periods, base_sa))
            n = local_slope(periods, base_sa, ts)
            for eta in eta_values:
                tssi = min(periods[-1], ts * math.sqrt(1.0 + eta))
                for dg in damping_gain_values:
                    zeta_eff = min(0.125, 0.05 + 0.5 * dg * eta / (1.0 + eta))
                    sa_flex = float(np.interp(tssi, periods, spectra[(source, min(zetas, key=lambda z: abs(z - zeta_eff)))]))
                    for kappa in kappa_values:
                        direct_rv = kappa * sa_flex / max(sa_fixed, 1e-12)
                        approx_rv = kappa * (math.sqrt(1.0 + eta) ** n) * max(0.45, 1.0 - dg * eta / (1.0 + eta))
                        direct_rd = (1.0 + eta) * direct_rv
                        for ptheta in ptheta_values:
                            rows.append(
                                {
                                    "source": source,
                                    "Ts": ts,
                                    "eta": eta,
                                    "ptheta": ptheta,
                                    "eta_x": eta * (1.0 - ptheta),
                                    "eta_theta": eta * ptheta,
                                    "kappa": kappa,
                                    "damping_gain": dg,
                                    "zeta_eff_nearest": min(zetas, key=lambda z: abs(z - zeta_eff)),
                                    "local_slope_n": n,
                                    "direct_Rv": direct_rv,
                                    "approx_Rv": approx_rv,
                                    "direct_Rd": direct_rd,
                                    "translation_component": eta * (1.0 - ptheta) * direct_rv,
                                    "rocking_component": eta * ptheta * direct_rv,
                                    "force_class_direct": "beneficial" if direct_rv < 1.0 else "detrimental_or_neutral",
                                    "force_class_approx": "beneficial" if approx_rv < 1.0 else "detrimental_or_neutral",
                                    "displacement_class_direct": "beneficial" if direct_rd < 1.0 else "detrimental_or_neutral",
                                }
                            )

    with (OUT_DIR / "direct_spectrum_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    class_ok = sum(1 for r in rows if r["force_class_direct"] == r["force_class_approx"])
    abs_errors = [abs(float(r["direct_Rv"]) - float(r["approx_Rv"])) for r in rows]
    mixed = sum(1 for r in rows if float(r["direct_Rv"]) < 1.0 and float(r["direct_Rd"]) >= 1.0)
    summary = {
        "sources": source_notes,
        "total_cases": total,
        "classification_accuracy_percent": 100.0 * class_ok / total,
        "mean_abs_force_ratio_error": float(np.mean(abs_errors)),
        "max_abs_force_ratio_error": float(np.max(abs_errors)),
        "force_beneficial_percent": 100.0 * sum(1 for r in rows if float(r["direct_Rv"]) < 1.0) / total,
        "total_displacement_beneficial_percent": 100.0 * sum(1 for r in rows if float(r["direct_Rd"]) < 1.0) / total,
        "mixed_force_benefit_displacement_penalty_percent": 100.0 * mixed / total,
        "notes": "Direct spectral validation uses one normalized code-compatible spectrum and three public El Centro Vibrationdata records/components.",
    }
    (OUT_DIR / "direct_spectrum_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_plot(rows, OUT_DIR / "direct_vs_approx_force_ratio.png", "Direct spectral ratio vs slope approximation", "approx_Rv")
    make_plot(rows, OUT_DIR / "direct_force_vs_total_displacement.png", "Direct force ratio vs total displacement ratio", "direct_Rd")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
