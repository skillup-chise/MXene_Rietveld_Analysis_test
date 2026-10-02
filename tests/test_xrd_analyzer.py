"""Tests for Bragg calculations and peak recovery on known patterns."""

import numpy as np

from loader import XRDSpectrum
from xrd_analyzer import (
    XRDAnalysisConfig,
    analyze_spectrum,
    apparent_crystallite_size_nm,
    c_axis_from_00l,
    pseudo_voigt_profile,
    two_theta_from_d,
)


WAVELENGTH = 1.5406


def _spectrum(sample_id: str, peaks: list[dict], seed: int = 0, noise: float = 2.0) -> XRDSpectrum:
    two_theta = np.arange(3.0, 50.0 + 1e-9, 0.02)
    intensity = 25.0 + 140.0 * np.exp(-(two_theta - 3.0) / 18.0) + 0.2 * two_theta
    for peak in peaks:
        intensity = intensity + pseudo_voigt_profile(
            two_theta,
            peak["height"],
            peak["center"],
            peak["fwhm"],
            peak.get("eta", 0.25),
        )
    intensity = intensity + np.random.default_rng(seed).normal(0.0, noise, size=two_theta.size)
    return XRDSpectrum(sample_id, two_theta, np.clip(intensity, 0, None), f"{sample_id}.csv")


def test_bragg_and_scherrer_constants() -> None:
    c_axis = c_axis_from_00l(7.0, WAVELENGTH, l_index=2)
    assert abs(c_axis - 25.235657) < 1e-4
    recovered = float(two_theta_from_d(c_axis / 2.0, WAVELENGTH))
    assert abs(recovered - 7.0) < 1e-6
    size_nm = apparent_crystallite_size_nm(7.0, 0.8, WAVELENGTH, k_factor=0.9)
    assert abs(size_nm - 9.9489) < 1e-2


def test_recovers_002_position_width_and_c_axis() -> None:
    c_expected = c_axis_from_00l(7.0, WAVELENGTH)
    harmonic = float(two_theta_from_d(c_expected / 4.0, WAVELENGTH))
    spectrum = _spectrum(
        "clean",
        [
            {"height": 1200, "center": 7.0, "fwhm": 0.80, "eta": 0.2},
            {"height": 280, "center": harmonic, "fwhm": 0.90, "eta": 0.3},
        ],
    )
    result = analyze_spectrum(spectrum, XRDAnalysisConfig(wavelength_A=WAVELENGTH))
    features = result.features
    assert abs(features["mxene_002_center_deg"] - 7.0) < 0.05
    assert abs(features["mxene_002_fwhm_deg"] - 0.80) < 0.12
    assert abs(features["c_axis_A"] - c_expected) < 0.2
    assert abs(features["crystallite_size_002_nm"] - apparent_crystallite_size_nm(7.0, 0.80, WAVELENGTH)) < 1.5
    assert abs(features["harmonic_004_delta_deg"]) < 0.15
    assert features["max_impurity_index"] == 0.0
    assert features["mxene_002_fit_method"] == "pseudo_voigt"


def test_max_and_tio2_indices_track_peak_heights() -> None:
    spectrum = _spectrum(
        "impure",
        [
            {"height": 1000, "center": 6.8, "fwhm": 0.70},
            {"height": 500, "center": 39.0, "fwhm": 0.18},
            {"height": 250, "center": 25.30, "fwhm": 0.20},
        ],
        seed=1,
    )
    features = analyze_spectrum(spectrum).features
    assert 0.25 < features["max_impurity_index"] < 0.45
    assert 0.12 < features["tio2_impurity_index"] < 0.30
    assert abs(features["max_104_center_deg"] - 39.0) < 0.08


def test_split_basal_peaks_assign_residual_max() -> None:
    spectrum = _spectrum(
        "split",
        [
            {"height": 1400, "center": 6.6, "fwhm": 0.65},
            {"height": 450, "center": 9.52, "fwhm": 0.20},
        ],
        seed=2,
    )
    features = analyze_spectrum(spectrum).features
    assert features["assignment_flag"] == "mxene_002_with_residual_max"
    assert abs(features["mxene_002_center_deg"] - 6.6) < 0.08
    assert abs(features["max_002_center_deg"] - 9.52) < 0.1


def test_c_axis_scales_with_wavelength() -> None:
    spectrum = _spectrum("wave", [{"height": 1000, "center": 7.0, "fwhm": 0.7}], noise=1.0)
    cu = analyze_spectrum(spectrum, XRDAnalysisConfig(wavelength_A=1.5406)).features["c_axis_A"]
    co = analyze_spectrum(spectrum, XRDAnalysisConfig(wavelength_A=1.7890)).features["c_axis_A"]
    assert abs(co / cu - 1.7890 / 1.5406) < 0.01


def test_als_baseline_recovers_center() -> None:
    spectrum = _spectrum("als", [{"height": 900, "center": 7.4, "fwhm": 0.9}], seed=3, noise=1.5)
    features = analyze_spectrum(spectrum, XRDAnalysisConfig(baseline_method="als")).features
    assert abs(features["mxene_002_center_deg"] - 7.4) < 0.06
