"""Synthetic Ti3C2Tx-style XRD campaign for pipeline checks.

Every file written here is simulated. The generator is an answer key for the
code, not a measurement and not a recommendation for how to etch a MAX phase.

The latent etching extent increases with time, temperature, HCl concentration,
LiF:MAX ratio, stirring speed, stirrer-bar size, and washing, and decreases as
the flask gets larger. That extent widens the c axis, sharpens the (002) line,
and lowers the residual MAX intensity. Air, heat, and long etches raise a
TiO2-like anatase line. Argon suppresses that line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xrd_analyzer import WAVELENGTH_CU_KA1, pseudo_voigt_profile, two_theta_from_d

DEMO_DIR = Path(__file__).resolve().parent / "data" / "demo"
DEMO_SEED = 42
DEMO_SAMPLES = 36

SYNTHESIS_COLUMNS = [
    "sample_id",
    "stirrer_bar_size_mm",
    "stirring_speed_rpm",
    "flask_volume_ml",
    "etching_time_h",
    "etching_temperature_C",
    "hcl_concentration_M",
    "lif_to_max_molar_ratio",
    "washing_cycles",
    "atmosphere",
]


def simulate_campaign(n: int, seed: int) -> pd.DataFrame:
    """Build a reproducible synthesis table and the true diffraction outcomes."""

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "sample_id": [f"MX-{index:03d}" for index in range(1, n + 1)],
            "stirrer_bar_size_mm": rng.choice([15, 25, 40], n),
            "stirring_speed_rpm": rng.choice([200, 400, 600, 800], n),
            "flask_volume_ml": rng.choice([50, 100, 250, 500], n),
            "etching_time_h": rng.choice([6, 12, 24, 36, 48], n),
            "etching_temperature_C": rng.choice([25, 35, 45, 55], n),
            "hcl_concentration_M": rng.choice([6.0, 9.0, 12.0], n),
            "lif_to_max_molar_ratio": rng.choice([0.5, 1.0, 1.5, 2.0], n),
            "washing_cycles": rng.choice([3, 5, 8], n),
            "atmosphere": rng.choice(["air", "argon"], n),
        }
    )
    time = frame["etching_time_h"].to_numpy(dtype=float)
    temperature = frame["etching_temperature_C"].to_numpy(dtype=float)
    hcl = frame["hcl_concentration_M"].to_numpy(dtype=float)
    lif = frame["lif_to_max_molar_ratio"].to_numpy(dtype=float)
    rpm = frame["stirring_speed_rpm"].to_numpy(dtype=float)
    flask = frame["flask_volume_ml"].to_numpy(dtype=float)
    bar = frame["stirrer_bar_size_mm"].to_numpy(dtype=float)
    washes = frame["washing_cycles"].to_numpy(dtype=float)
    argon = (frame["atmosphere"].to_numpy() == "argon").astype(float)

    drive = (
        0.055 * (time - 24.0)
        + 0.045 * (temperature - 35.0)
        + 0.09 * (hcl - 9.0)
        + 0.70 * (lif - 1.0)
        + 0.0011 * (rpm - 500.0)
        - 0.0022 * (flask - 200.0)
        + 0.012 * (bar - 25.0)
        + 0.03 * (washes - 5.0)
    )
    extent = 1.0 / (1.0 + np.exp(-drive))
    c_axis = np.clip(19.0 + 10.0 * extent + rng.normal(0.0, 0.08, n), 18.7, 32.0)
    fwhm = np.clip(
        1.25
        - 0.70 * extent
        + 0.00055 * (flask - 200.0)
        - 0.00035 * (rpm - 500.0)
        + rng.normal(0.0, 0.02, n),
        0.28,
        2.2,
    )
    max_fraction = np.clip(1.05 * (1.0 - extent) + rng.normal(0.0, 0.02, n), 0.0, 1.0)
    tio2_fraction = np.clip(
        0.015
        + 0.0042 * (temperature - 25.0)
        + 0.0012 * time
        + 0.10 * (1.0 - argon)
        + rng.normal(0.0, 0.008, n),
        0.0,
        0.55,
    )
    i_002 = 1000.0 + 1400.0 * extent
    i_104 = np.where(max_fraction > 0.04, 2200.0 * max_fraction, 0.0)
    i_anatase = np.where(tio2_fraction > 0.025, 2200.0 * tio2_fraction, 0.0)
    two_theta_002 = np.asarray(two_theta_from_d(c_axis / 2.0, WAVELENGTH_CU_KA1), dtype=float)

    frame["true_extent"] = extent
    frame["true_c_axis_A"] = c_axis
    frame["true_two_theta_002"] = two_theta_002
    frame["true_fwhm_002_deg"] = fwhm
    frame["true_i_002"] = i_002
    frame["true_i_104"] = i_104
    frame["true_i_anatase"] = i_anatase
    frame["true_max_impurity_index"] = i_104 / (i_104 + i_002)
    frame["true_tio2_impurity_index"] = i_anatase / (i_anatase + i_002)
    return frame


def render_pattern(row: pd.Series, two_theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one noisy pattern from a row of :func:`simulate_campaign`."""

    intensity = 30.0 + 200.0 * np.exp(-(two_theta - two_theta[0]) / 16.0)
    c_axis = float(row["true_c_axis_A"])
    fwhm = float(row["true_fwhm_002_deg"])
    i_002 = float(row["true_i_002"])
    i_104 = float(row["true_i_104"])
    i_anatase = float(row["true_i_anatase"])
    center_002 = float(row["true_two_theta_002"])

    peaks = [
        (i_002, center_002, fwhm, 0.30),
        (0.22 * i_002, float(two_theta_from_d(c_axis / 4.0, WAVELENGTH_CU_KA1)), min(fwhm * 1.1, 2.4), 0.35),
        (0.16 * i_002, 60.80, 0.42, 0.25),
    ]
    if i_104 > 0:
        peaks.append((i_104, 39.00, 0.18, 0.20))
        peaks.append((0.28 * i_104, 19.15, 0.20, 0.20))
        if abs(center_002 - 9.52) > 1.3 and 0.35 * i_104 > 80:
            peaks.append((0.35 * i_104, 9.52, 0.20, 0.25))
    if i_anatase > 0:
        peaks.append((i_anatase, 25.30, 0.22, 0.15))

    for height, center, width, eta in peaks:
        if center < two_theta[0] or center > two_theta[-1]:
            continue
        intensity = intensity + pseudo_voigt_profile(two_theta, height, center, width, eta)
    intensity = intensity + rng.normal(0.0, 6.0, size=two_theta.size)
    return np.clip(intensity, 0.0, None)


def generate_demo_dataset(
    output_dir: Path | str = DEMO_DIR,
    n: int = DEMO_SAMPLES,
    seed: int = DEMO_SEED,
    step: float = 0.02,
    two_theta_min: float = 3.0,
    two_theta_max: float = 65.0,
) -> Path:
    """Write synthetic patterns, the synthesis table, and the generator answer key."""

    output_dir = Path(output_dir)
    xrd_dir = output_dir / "xrd"
    xrd_dir.mkdir(parents=True, exist_ok=True)
    campaign = simulate_campaign(n, seed)
    two_theta = np.arange(two_theta_min, two_theta_max + step * 0.5, step)
    rng = np.random.default_rng(seed + 17)
    for _, row in campaign.iterrows():
        intensity = render_pattern(row, two_theta, rng)
        path = xrd_dir / f"{row['sample_id']}.csv"
        frame = pd.DataFrame({"two_theta_deg": two_theta, "intensity": intensity})
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# SYNTHETIC spectrum — not an experimental measurement\n")
            handle.write(f"# sample_id: {row['sample_id']}\n")
            handle.write(f"# wavelength_A: {WAVELENGTH_CU_KA1}\n")
            frame.to_csv(handle, index=False, float_format="%.4f")

    campaign[SYNTHESIS_COLUMNS].to_csv(output_dir / "synthesis_params.csv", index=False)
    campaign.to_csv(output_dir / "generator_truth.csv", index=False)
    marker = output_dir / "SYNTHETIC.txt"
    marker.write_text(
        "\n".join(
            [
                "SYNTHETIC DATA",
                "These XRD patterns and synthesis parameters were simulated by demo_data.py.",
                "They are not experimental MXene measurements.",
                f"samples: {n}",
                f"seed: {seed}",
                f"wavelength_A: {WAVELENGTH_CU_KA1}",
                f"two_theta_step_deg: {step}",
                "generator_truth.csv is the simulation answer key. The analysis pipeline does not read it.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_dir


def ensure_demo_dataset(output_dir: Path | str = DEMO_DIR, n: int = DEMO_SAMPLES, seed: int = DEMO_SEED) -> Path:
    """Generate the demo set once, then reuse it."""

    output_dir = Path(output_dir)
    marker = output_dir / "SYNTHETIC.txt"
    xrd_dir = output_dir / "xrd"
    existing = list(xrd_dir.glob("MX-*.csv")) if xrd_dir.exists() else []
    expected = f"samples: {n}"
    if marker.exists() and expected in marker.read_text(encoding="utf-8") and len(existing) == n:
        return output_dir
    return generate_demo_dataset(output_dir, n=n, seed=seed)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the synthetic MXene XRD demo dataset.")
    parser.add_argument("--output", type=Path, default=DEMO_DIR)
    parser.add_argument("--n", type=int, default=DEMO_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()
    path = generate_demo_dataset(args.output, n=args.n, seed=args.seed, step=args.step)
    print(f"Wrote synthetic demo data to {path}")


if __name__ == "__main__":
    main()
