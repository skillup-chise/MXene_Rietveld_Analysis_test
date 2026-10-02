"""Preprocess MXene XRD patterns and measure basal and impurity peaks.

The default peak windows match the common Ti3AlC2 → Ti3C2Tx case measured with
Cu Kα radiation. Edit ``XRDAnalysisConfig.reference_windows`` for other MAX or
MXene chemistries. Reported impurity numbers are intensity indices, not Rietveld
weight fractions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

from loader import XRDSpectrum

logger = logging.getLogger(__name__)

WAVELENGTH_CU_KA1 = 1.5406  # Å

FEATURE_COLUMNS = [
    "sample_id",
    "source_file",
    "wavelength_A",
    "n_points",
    "two_theta_min",
    "two_theta_max",
    "assignment_flag",
    "basal_peak_count",
    "mxene_002_center_deg",
    "mxene_002_center_stderr_deg",
    "mxene_002_intensity",
    "mxene_002_fwhm_deg",
    "mxene_002_fwhm_stderr_deg",
    "mxene_002_eta",
    "mxene_002_area",
    "mxene_002_r_squared",
    "mxene_002_fit_method",
    "d_002_A",
    "c_axis_A",
    "c_axis_stderr_A",
    "crystallite_size_002_nm",
    "crystallinity_index",
    "max_002_center_deg",
    "max_002_intensity",
    "max_104_center_deg",
    "max_104_intensity",
    "max_104_fwhm_deg",
    "max_to_002_intensity_ratio",
    "max_impurity_index",
    "tio2_anatase_intensity",
    "tio2_rutile_intensity",
    "tio2_to_002_intensity_ratio",
    "tio2_impurity_index",
    "harmonic_004_center_deg",
    "harmonic_004_delta_deg",
]


@dataclass(frozen=True)
class ReferenceWindow:
    """A fixed 2θ interval used to look for one impurity or secondary phase."""

    name: str
    two_theta_min: float
    two_theta_max: float
    phase: str
    description: str


# Ti3AlC2 (104) is the strong residual-MAX marker near 39°.
# Anatase (101) and rutile (110) are the usual TiO2 impurity lines.
DEFAULT_REFERENCE_WINDOWS: tuple[ReferenceWindow, ...] = (
    ReferenceWindow("max_104", 38.4, 39.9, "MAX", "Ti3AlC2-like (104)"),
    ReferenceWindow("tio2_anatase_101", 24.85, 25.85, "TiO2-anatase", "anatase (101)"),
    ReferenceWindow("tio2_rutile_110", 27.05, 27.90, "TiO2-rutile", "rutile (110)"),
)


@dataclass
class XRDAnalysisConfig:
    """Knobs for wavelength, baseline, and which reflections are measured."""

    wavelength_A: float = WAVELENGTH_CU_KA1
    scherrer_k: float = 0.9
    baseline_method: str = "opening"  # "opening" or "als"
    baseline_window_deg: float = 8.0
    als_lam: float = 1.0e6
    als_p: float = 0.01
    als_niter: int = 10
    smooth_window_deg: float = 0.22
    basal_min_deg: float = 4.5
    basal_max_deg: float = 10.8
    max_002_reference_deg: float = 9.52
    max_002_tolerance_deg: float = 0.50
    min_peak_separation_deg: float = 0.35
    min_fwhm_deg: float = 0.04
    max_fwhm_deg: float = 5.0
    prominence_sigma: float = 3.0
    prominence_fraction: float = 0.015
    second_peak_height_fraction: float = 0.12
    reference_windows: tuple[ReferenceWindow, ...] = DEFAULT_REFERENCE_WINDOWS

    def __post_init__(self) -> None:
        if self.wavelength_A <= 0:
            raise ValueError("wavelength_A must be positive")
        if self.baseline_method not in {"opening", "als"}:
            raise ValueError("baseline_method must be 'opening' or 'als'")


@dataclass
class PeakFit:
    """One measured reflection."""

    name: str
    phase: str
    detected: bool
    center_deg: float = np.nan
    center_stderr_deg: float = np.nan
    intensity: float = 0.0
    fwhm_deg: float = np.nan
    fwhm_stderr_deg: float = np.nan
    eta: float = np.nan
    area: float = np.nan
    r_squared: float = np.nan
    d_spacing_A: float = np.nan
    method: str = ""
    description: str = ""

    def as_row(self, sample_id: str) -> dict:
        return {
            "sample_id": sample_id,
            "peak_name": self.name,
            "phase": self.phase,
            "description": self.description,
            "detected": self.detected,
            "center_deg": self.center_deg,
            "center_stderr_deg": self.center_stderr_deg,
            "intensity": self.intensity,
            "fwhm_deg": self.fwhm_deg,
            "fwhm_stderr_deg": self.fwhm_stderr_deg,
            "eta": self.eta,
            "area": self.area,
            "r_squared": self.r_squared,
            "d_spacing_A": self.d_spacing_A,
            "method": self.method,
        }


@dataclass
class SpectrumResult:
    """Peak metrics plus the traces needed to draw the pattern."""

    sample_id: str
    features: dict
    peaks: list[PeakFit] = field(default_factory=list)
    two_theta: np.ndarray | None = None
    intensity_raw: np.ndarray | None = None
    baseline: np.ndarray | None = None
    intensity_corrected: np.ndarray | None = None

    @property
    def peak_rows(self) -> list[dict]:
        return [peak.as_row(self.sample_id) for peak in self.peaks]


def bragg_d(two_theta_deg: float | np.ndarray, wavelength: float) -> float | np.ndarray:
    """Interplanar spacing d = λ / (2 sin θ), with 2θ in degrees and λ in Å."""

    theta = np.deg2rad(np.asarray(two_theta_deg, dtype=float) / 2.0)
    sine = np.sin(theta)
    return wavelength / (2.0 * sine)


def two_theta_from_d(d_spacing_A: float | np.ndarray, wavelength: float) -> float | np.ndarray:
    """Invert Bragg's law. ``d`` and ``wavelength`` are both in Å."""

    sine = np.clip(wavelength / (2.0 * np.asarray(d_spacing_A, dtype=float)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(sine))


def c_axis_from_00l(two_theta_deg: float, wavelength: float, l_index: int = 2) -> float:
    """Hexagonal c axis from an (00l) reflection. For (002), c = 2 d_002."""

    if l_index <= 0 or not np.isfinite(two_theta_deg):
        return np.nan
    return float(l_index * bragg_d(two_theta_deg, wavelength))


def c_axis_uncertainty(two_theta_deg: float, two_theta_stderr_deg: float, wavelength: float, l_index: int = 2) -> float:
    """Propagate a 2θ standard error onto c with a numerical derivative."""

    if not np.isfinite(two_theta_deg) or not np.isfinite(two_theta_stderr_deg):
        return np.nan
    step = 1.0e-4
    high = c_axis_from_00l(two_theta_deg + step, wavelength, l_index)
    low = c_axis_from_00l(two_theta_deg - step, wavelength, l_index)
    return float(abs((high - low) / (2.0 * step)) * two_theta_stderr_deg)


def apparent_crystallite_size_nm(
    two_theta_deg: float,
    fwhm_deg: float,
    wavelength: float,
    k_factor: float = 0.9,
) -> float:
    """Scherrer size in nm. Instrumental broadening is not subtracted."""

    if not np.isfinite(two_theta_deg) or not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
        return np.nan
    theta = np.deg2rad(two_theta_deg / 2.0)
    beta = np.deg2rad(fwhm_deg)
    cosine = float(np.cos(theta))
    if cosine <= 0 or beta <= 0:
        return np.nan
    size_angstrom = k_factor * wavelength / (beta * cosine)
    return float(size_angstrom / 10.0)


def pseudo_voigt_profile(
    x: np.ndarray,
    amplitude: float,
    center: float,
    fwhm: float,
    eta: float,
    offset: float = 0.0,
    slope: float = 0.0,
) -> np.ndarray:
    """Pseudo-Voigt peak with a shared FWHM plus a local linear background.

    ``eta`` = 0 is Gaussian and ``eta`` = 1 is Lorentzian. Both components are
    normalized to unit height, so ``amplitude`` is the peak height above the
    local line and ``fwhm`` is the full width at half maximum in the same units
    as ``x`` (degrees 2θ).
    """

    x = np.asarray(x, dtype=float)
    fwhm = max(float(fwhm), 1.0e-6)
    eta = float(np.clip(eta, 0.0, 1.0))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentzian = (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)
    profile = (1.0 - eta) * gaussian + eta * lorentzian
    return offset + slope * (x - center) + amplitude * profile


def _odd_window(size: int, maximum: int) -> int:
    size = int(size)
    if size % 2 == 0:
        size += 1
    if maximum % 2 == 0:
        maximum -= 1
    return max(0, min(size, maximum))


def smooth_signal(y: np.ndarray, step_deg: float, window_deg: float) -> np.ndarray:
    """Savitzky–Golay smooth used only to locate peaks, not to measure FWHM."""

    y = np.asarray(y, dtype=float)
    if y.size < 7 or step_deg <= 0:
        return y.copy()
    points = int(round(window_deg / step_deg))
    window = _odd_window(max(points, 5), y.size - 1 if y.size % 2 == 0 else y.size)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=3, mode="interp")


def opening_baseline(y: np.ndarray, window_points: int) -> np.ndarray:
    """Morphological opening: a rolling minimum followed by a rolling maximum.

    The window should be wider than the Bragg peaks and narrower than the
    slow background, so the opening follows the background under the peaks.
    """

    y = np.asarray(y, dtype=float)
    window = _odd_window(window_points, max(y.size - 1, 1))
    if window < 3:
        return np.full_like(y, np.nanmin(y) if y.size else 0.0)
    opened = maximum_filter1d(minimum_filter1d(y, size=window, mode="nearest"), size=window, mode="nearest")
    smooth_window = _odd_window(max(5, window // 5), y.size - 1 if y.size % 2 == 0 else y.size)
    if smooth_window >= 5:
        opened = savgol_filter(opened, window_length=smooth_window, polyorder=2, mode="interp")
    return opened


def als_baseline(y: np.ndarray, lam: float = 1.0e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline (Eilers & Boelens)."""

    y = np.asarray(y, dtype=float)
    n_points = y.size
    if n_points < 5:
        return np.full_like(y, np.nanmin(y) if n_points else 0.0)
    difference = sparse.diags(
        [np.ones(n_points - 2), -2.0 * np.ones(n_points - 2), np.ones(n_points - 2)],
        [0, 1, 2],
        shape=(n_points - 2, n_points),
        format="csc",
    )
    weights = np.ones(n_points)
    baseline = y.copy()
    for _ in range(niter):
        weight_matrix = sparse.diags(weights, 0, shape=(n_points, n_points), format="csc")
        system = weight_matrix + lam * (difference.T @ difference)
        baseline = np.asarray(spsolve(system, weights * y), dtype=float).ravel()
        weights = p * (y > baseline) + (1.0 - p) * (y < baseline)
    return baseline


def estimate_step(two_theta: np.ndarray) -> float:
    diffs = np.diff(np.asarray(two_theta, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("2θ axis must contain at least two increasing points")
    return float(np.median(diffs))


def preprocess_spectrum(two_theta: np.ndarray, intensity: np.ndarray, config: XRDAnalysisConfig) -> dict:
    """Smooth for detection and subtract a slowly varying background."""

    step = estimate_step(two_theta)
    window_points = int(round(config.baseline_window_deg / step))
    # Keep the structuring element from swallowing a short, peak-only scan.
    window_points = min(window_points, max(5, int(0.35 * two_theta.size)))
    if config.baseline_method == "als":
        try:
            baseline = als_baseline(intensity, lam=config.als_lam, p=config.als_p, niter=config.als_niter)
        except Exception as exc:
            logger.warning("ALS baseline failed (%s); using a morphological opening.", exc)
            baseline = opening_baseline(intensity, window_points)
    else:
        baseline = opening_baseline(intensity, window_points)
    corrected = np.asarray(intensity, dtype=float) - baseline
    smoothed = smooth_signal(corrected, step, config.smooth_window_deg)
    residual = corrected - smoothed
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(residual)) if residual.size else 0.0
    return {
        "step_deg": step,
        "baseline": baseline,
        "corrected": corrected,
        "smoothed": smoothed,
        "noise_sigma": sigma,
        "global_max": float(np.nanmax(smoothed)) if smoothed.size else 0.0,
    }


def _undetected(name: str, phase: str, description: str = "") -> PeakFit:
    return PeakFit(name=name, phase=phase, detected=False, description=description)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _half_maximum_fwhm(x: np.ndarray, y: np.ndarray, peak_index: int) -> float:
    """FWHM by interpolating the half-height crossings on either side of the max."""

    if peak_index <= 0 or peak_index >= len(y) - 1:
        return np.nan
    span = max(3, len(y) // 8)
    left = max(0, peak_index - span)
    right = min(len(y), peak_index + span)
    local = y[left:right]
    local_base = float(np.percentile(local, 10))
    height = float(y[peak_index] - local_base)
    if height <= 0:
        return np.nan
    level = local_base + 0.5 * height

    def crossing(direction: int) -> float:
        index = peak_index
        while 0 <= index < len(y) and y[index] >= level:
            index += direction
        if index <= 0 or index >= len(y):
            return np.nan
        high = index - direction
        y_high = float(y[high])
        y_low = float(y[index])
        if y_high == y_low:
            return float(x[index])
        fraction = (y_high - level) / (y_high - y_low)
        return float(x[high] + fraction * (x[index] - x[high]))

    left_x = crossing(-1)
    right_x = crossing(1)
    if not np.isfinite(left_x) or not np.isfinite(right_x):
        return np.nan
    width = right_x - left_x
    return float(width) if width > 0 else np.nan


def _r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    residual = float(np.sum((y - y_hat) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0:
        return np.nan
    return 1.0 - residual / total


def _fit_pseudo_voigt(
    x: np.ndarray,
    y: np.ndarray,
    center_guess: float,
    config: XRDAnalysisConfig,
) -> dict | None:
    if x.size < 6:
        return None
    peak_index = int(np.argmin(np.abs(x - center_guess)))
    amplitude = float(y[peak_index] - np.percentile(y, 20))
    if not np.isfinite(amplitude) or amplitude <= 0:
        amplitude = float(np.max(y) - np.min(y))
    if amplitude <= 0:
        return None
    fwhm_guess = _half_maximum_fwhm(x, y, peak_index)
    if not np.isfinite(fwhm_guess):
        fwhm_guess = 0.6
    fwhm_guess = _clip(fwhm_guess, config.min_fwhm_deg, config.max_fwhm_deg)

    low_fwhm = config.min_fwhm_deg
    high_fwhm = min(config.max_fwhm_deg, max(float(x[-1] - x[0]), low_fwhm + 0.05))
    low_center = float(x[1] if x.size > 2 else x[0])
    high_center = float(x[-2] if x.size > 2 else x[-1])
    if high_center <= low_center:
        low_center, high_center = float(x[0]), float(x[-1])
    low_offset = float(np.min(y) - 0.2 * amplitude)
    high_offset = float(np.percentile(y, 40) + 0.2 * amplitude)
    if high_offset <= low_offset:
        high_offset = low_offset + 1.0
    slope_limit = 0.08 * max(amplitude, 1.0)
    lower = [0.0, low_center, low_fwhm, 0.0, low_offset, -slope_limit]
    upper = [max(amplitude * 3.0, 1.0), high_center, high_fwhm, 1.0, high_offset, slope_limit]
    if any(hi <= lo for lo, hi in zip(lower, upper)):
        return None
    guess = [
        _clip(amplitude, lower[0] + 1e-6, upper[0] - 1e-6),
        _clip(float(x[peak_index]), lower[1] + 1e-6, upper[1] - 1e-6),
        _clip(fwhm_guess, lower[2] + 1e-6, upper[2] - 1e-6),
        0.4,
        _clip(float(np.percentile(y, 15)), lower[4] + 1e-6, upper[4] - 1e-6),
        0.0,
    ]
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            parameters, covariance = curve_fit(
                pseudo_voigt_profile,
                x,
                y,
                p0=guess,
                bounds=(lower, upper),
                maxfev=20000,
            )
    except Exception:
        return None
    stderr = np.full(parameters.size, np.nan)
    if np.all(np.isfinite(covariance)):
        stderr = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    y_hat = pseudo_voigt_profile(x, *parameters)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = float(trapezoid(np.clip(y_hat - parameters[4], 0.0, None), x))
    return {
        "amplitude": float(parameters[0]),
        "center": float(parameters[1]),
        "fwhm": float(parameters[2]),
        "eta": float(parameters[3]),
        "offset": float(parameters[4]),
        "center_stderr": float(stderr[1]),
        "fwhm_stderr": float(stderr[2]),
        "r_squared": _r_squared(y, y_hat),
        "area": area,
        "method": "pseudo_voigt",
    }


def _direct_measurement(x: np.ndarray, y: np.ndarray) -> dict | None:
    if x.size < 5:
        return None
    peak_index = int(np.argmax(y))
    if peak_index <= 0 or peak_index >= len(y) - 1:
        return None
    local_base = float(np.percentile(y, 15))
    height = float(y[peak_index] - local_base)
    fwhm = _half_maximum_fwhm(x, y, peak_index)
    if height <= 0 or not np.isfinite(fwhm):
        return None
    return {
        "amplitude": height,
        "center": float(x[peak_index]),
        "fwhm": float(fwhm),
        "eta": np.nan,
        "center_stderr": np.nan,
        "fwhm_stderr": np.nan,
        "r_squared": np.nan,
        "area": np.nan,
        "method": "direct",
    }


def _measurement_to_peak(
    measurement: dict | None,
    name: str,
    phase: str,
    description: str,
    wavelength: float,
    threshold: float,
) -> PeakFit:
    if measurement is None or measurement["amplitude"] < threshold:
        return _undetected(name, phase, description)
    if measurement["method"] == "pseudo_voigt" and (
        not np.isfinite(measurement["r_squared"]) or measurement["r_squared"] < 0.3
    ):
        return _undetected(name, phase, description)
    center = measurement["center"]
    return PeakFit(
        name=name,
        phase=phase,
        detected=True,
        center_deg=center,
        center_stderr_deg=measurement["center_stderr"],
        intensity=measurement["amplitude"],
        fwhm_deg=measurement["fwhm"],
        fwhm_stderr_deg=measurement["fwhm_stderr"],
        eta=measurement["eta"],
        area=measurement["area"],
        r_squared=measurement["r_squared"],
        d_spacing_A=float(bragg_d(center, wavelength)),
        method=measurement["method"],
        description=description,
    )


def fit_interval(
    two_theta: np.ndarray,
    corrected: np.ndarray,
    smoothed: np.ndarray,
    x_min: float,
    x_max: float,
    threshold: float,
    config: XRDAnalysisConfig,
    name: str,
    phase: str,
    description: str = "",
    center_hint: float | None = None,
) -> PeakFit:
    """Fit the strongest peak inside ``[x_min, x_max]``."""

    mask = (two_theta >= x_min) & (two_theta <= x_max)
    if int(np.count_nonzero(mask)) < 8:
        return _undetected(name, phase, description)
    x = two_theta[mask]
    y = corrected[mask]
    y_smooth = smoothed[mask]
    if center_hint is None:
        peak_index = int(np.argmax(y_smooth))
    else:
        peak_index = int(np.argmin(np.abs(x - center_hint)))
    if peak_index <= 1 or peak_index >= len(y_smooth) - 2:
        return _undetected(name, phase, description)
    if float(y_smooth[peak_index]) < threshold and float(np.max(y_smooth)) < threshold:
        return _undetected(name, phase, description)

    fwhm_guess = _half_maximum_fwhm(x, y_smooth, peak_index)
    if not np.isfinite(fwhm_guess):
        fwhm_guess = 0.6
    half_width = float(np.clip(4.0 * fwhm_guess, 0.8, 3.0))
    local = (x >= x[peak_index] - half_width) & (x <= x[peak_index] + half_width)
    if int(np.count_nonzero(local)) < 6:
        local = np.ones(x.size, dtype=bool)
    measurement = _fit_pseudo_voigt(x[local], y[local], float(x[peak_index]), config)
    peak = _measurement_to_peak(measurement, name, phase, description, config.wavelength_A, threshold * 0.5)
    if peak.detected:
        return peak
    direct = _direct_measurement(x[local], y[local])
    return _measurement_to_peak(direct, name, phase, description, config.wavelength_A, threshold * 0.5)


def detect_basal_peaks(
    two_theta: np.ndarray,
    corrected: np.ndarray,
    smoothed: np.ndarray,
    step_deg: float,
    threshold: float,
    config: XRDAnalysisConfig,
) -> list[PeakFit]:
    """Find one or two reflections in the low-angle (002) region."""

    mask = (two_theta >= config.basal_min_deg) & (two_theta <= config.basal_max_deg)
    if int(np.count_nonzero(mask)) < 8:
        return []
    x = two_theta[mask]
    y_smooth = smoothed[mask]
    distance = max(1, int(round(config.min_peak_separation_deg / step_deg)))
    indices, _properties = find_peaks(y_smooth, prominence=threshold, distance=distance)
    if indices.size == 0 and float(np.max(y_smooth)) >= threshold:
        indices = np.array([int(np.argmax(y_smooth))])
    if indices.size == 0:
        return []

    strongest = float(np.max(y_smooth[indices]))
    keep = [int(index) for index in indices if y_smooth[index] >= config.second_peak_height_fraction * strongest]
    if not keep:
        keep = [int(np.argmax(y_smooth))]
    centers = [float(x[index]) for index in keep]
    peaks: list[PeakFit] = []
    for order, center in enumerate(centers):
        left_limit = config.basal_min_deg if order == 0 else (centers[order - 1] + center) / 2.0
        right_limit = config.basal_max_deg if order == len(centers) - 1 else (center + centers[order + 1]) / 2.0
        peaks.append(
            fit_interval(
                two_theta,
                corrected,
                smoothed,
                max(config.basal_min_deg, left_limit),
                min(config.basal_max_deg, right_limit),
                threshold,
                config,
                name=f"basal_{order + 1}",
                phase="basal",
                description="low-angle basal reflection",
                center_hint=center,
            )
        )
    return [peak for peak in peaks if peak.detected]


def assign_basal_peaks(peaks: list[PeakFit], config: XRDAnalysisConfig) -> tuple[PeakFit | None, PeakFit | None, str]:
    """Label the product (002) and, when it is a separate line, residual MAX (002).

    The lowest-angle basal peak is the interlayer reflection used for the c
    axis. A second peak sitting on the Ti3AlC2 (002) position is recorded as
    residual MAX. A single peak near that position is still used for c, and
    the assignment flag says the spacing matches unetched MAX.
    """

    detected = sorted(peaks, key=lambda peak: peak.center_deg)
    if not detected:
        return None, None, "not_detected"
    product = detected[0]
    max_peak = None
    for peak in detected[1:]:
        if abs(peak.center_deg - config.max_002_reference_deg) <= config.max_002_tolerance_deg:
            max_peak = peak
            break
    if max_peak is not None:
        return product, max_peak, "mxene_002_with_residual_max"
    if abs(product.center_deg - config.max_002_reference_deg) <= 0.35:
        return product, None, "basal_matches_max_position"
    if product.center_deg >= 8.9:
        return product, None, "ambiguous_basal_peak"
    return product, None, "mxene_002"


def _intensity(peak: PeakFit | None) -> float:
    if peak is None or not peak.detected:
        return 0.0
    return float(peak.intensity)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _blank_features(sample_id: str, source_file: str, flag: str, wavelength: float) -> dict:
    row = {column: np.nan for column in FEATURE_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "source_file": source_file,
            "wavelength_A": wavelength,
            "assignment_flag": flag,
            "basal_peak_count": 0,
            "mxene_002_intensity": 0.0,
            "max_002_intensity": 0.0,
            "max_104_intensity": 0.0,
            "tio2_anatase_intensity": 0.0,
            "tio2_rutile_intensity": 0.0,
            "mxene_002_fit_method": "",
        }
    )
    return row


def _peak_by_name(peaks: list[PeakFit], name: str) -> PeakFit | None:
    for peak in peaks:
        if peak.name == name and peak.detected:
            return peak
    return None


def analyze_spectrum(spectrum: XRDSpectrum, config: XRDAnalysisConfig | None = None) -> SpectrumResult:
    """Background-correct one pattern and extract MXene, MAX, and TiO2 metrics."""

    config = config or XRDAnalysisConfig()
    source_file = spectrum.source_path.rsplit("/", 1)[-1]
    features = _blank_features(spectrum.sample_id, source_file, "analysis_failed", config.wavelength_A)
    if spectrum.two_theta.size < 20:
        features["assignment_flag"] = "too_few_points"
        return SpectrumResult(sample_id=spectrum.sample_id, features=features, peaks=[])

    prepared = preprocess_spectrum(spectrum.two_theta, spectrum.intensity, config)
    threshold = max(
        config.prominence_sigma * prepared["noise_sigma"],
        config.prominence_fraction * max(prepared["global_max"], 0.0),
    )
    reference_peaks: list[PeakFit] = []
    for window in config.reference_windows:
        reference_peaks.append(
            fit_interval(
                spectrum.two_theta,
                prepared["corrected"],
                prepared["smoothed"],
                window.two_theta_min,
                window.two_theta_max,
                threshold,
                config,
                name=window.name,
                phase=window.phase,
                description=window.description,
            )
        )
    basal = detect_basal_peaks(
        spectrum.two_theta,
        prepared["corrected"],
        prepared["smoothed"],
        prepared["step_deg"],
        threshold,
        config,
    )
    product, max_002, flag = assign_basal_peaks(basal, config)
    if product is not None:
        product.name = "mxene_002"
        product.phase = "MXene"
        product.description = "interlayer (002) used for the c axis"
    if max_002 is not None:
        max_002.name = "max_002"
        max_002.phase = "MAX"
        max_002.description = "residual MAX (002)"

    harmonic = _undetected("mxene_004", "MXene", "predicted (004) harmonic")
    c_axis = np.nan
    c_stderr = np.nan
    d_002 = np.nan
    if product is not None:
        d_002 = float(product.d_spacing_A)
        c_axis = c_axis_from_00l(product.center_deg, config.wavelength_A, l_index=2)
        c_stderr = c_axis_uncertainty(
            product.center_deg,
            product.center_stderr_deg,
            config.wavelength_A,
            l_index=2,
        )
        if np.isfinite(c_axis) and c_axis > 0:
            predicted = float(two_theta_from_d(c_axis / 4.0, config.wavelength_A))
            harmonic = fit_interval(
                spectrum.two_theta,
                prepared["corrected"],
                prepared["smoothed"],
                predicted - 0.6,
                predicted + 0.6,
                threshold,
                config,
                name="mxene_004",
                phase="MXene",
                description="(004) harmonic check",
                center_hint=predicted,
            )
            if harmonic.detected:
                harmonic = PeakFit(**{**harmonic.__dict__, "name": "mxene_004"})

    max_104 = _peak_by_name(reference_peaks, "max_104")
    anatase = _peak_by_name(reference_peaks, "tio2_anatase_101")
    rutile = _peak_by_name(reference_peaks, "tio2_rutile_110")
    i_002 = _intensity(product)
    i_104 = _intensity(max_104)
    i_anatase = _intensity(anatase)
    i_rutile = _intensity(rutile)
    i_oxide = i_anatase + i_rutile
    fwhm = product.fwhm_deg if product is not None else np.nan
    crystallite = (
        apparent_crystallite_size_nm(product.center_deg, product.fwhm_deg, config.wavelength_A, config.scherrer_k)
        if product is not None
        else np.nan
    )
    harmonic_delta = (
        float(harmonic.center_deg - two_theta_from_d(c_axis / 4.0, config.wavelength_A))
        if harmonic.detected and np.isfinite(c_axis)
        else np.nan
    )

    features.update(
        {
            "n_points": int(spectrum.two_theta.size),
            "two_theta_min": float(spectrum.two_theta.min()),
            "two_theta_max": float(spectrum.two_theta.max()),
            "assignment_flag": flag,
            "basal_peak_count": int(len(basal)),
            "mxene_002_center_deg": product.center_deg if product is not None else np.nan,
            "mxene_002_center_stderr_deg": product.center_stderr_deg if product is not None else np.nan,
            "mxene_002_intensity": i_002,
            "mxene_002_fwhm_deg": fwhm,
            "mxene_002_fwhm_stderr_deg": product.fwhm_stderr_deg if product is not None else np.nan,
            "mxene_002_eta": product.eta if product is not None else np.nan,
            "mxene_002_area": product.area if product is not None else np.nan,
            "mxene_002_r_squared": product.r_squared if product is not None else np.nan,
            "mxene_002_fit_method": product.method if product is not None else "",
            "d_002_A": d_002,
            "c_axis_A": c_axis,
            "c_axis_stderr_A": c_stderr,
            "crystallite_size_002_nm": crystallite,
            "crystallinity_index": _safe_ratio(1.0, fwhm) if np.isfinite(fwhm) and fwhm > 0 else np.nan,
            "max_002_center_deg": max_002.center_deg if max_002 is not None else np.nan,
            "max_002_intensity": _intensity(max_002),
            "max_104_center_deg": max_104.center_deg if max_104 is not None else np.nan,
            "max_104_intensity": i_104,
            "max_104_fwhm_deg": max_104.fwhm_deg if max_104 is not None else np.nan,
            "max_to_002_intensity_ratio": _safe_ratio(i_104, i_002),
            "max_impurity_index": (
                np.nan if i_002 <= 0 and i_104 <= 0 else (1.0 if i_002 <= 0 else i_104 / (i_104 + i_002))
            ),
            "tio2_anatase_intensity": i_anatase,
            "tio2_rutile_intensity": i_rutile,
            "tio2_to_002_intensity_ratio": _safe_ratio(i_oxide, i_002),
            "tio2_impurity_index": (
                np.nan if i_002 <= 0 and i_oxide <= 0 else (1.0 if i_002 <= 0 else i_oxide / (i_oxide + i_002))
            ),
            "harmonic_004_center_deg": harmonic.center_deg if harmonic.detected else np.nan,
            "harmonic_004_delta_deg": harmonic_delta,
        }
    )
    peaks = [peak for peak in [product, max_002, harmonic, *reference_peaks] if peak is not None]
    return SpectrumResult(
        sample_id=spectrum.sample_id,
        features=features,
        peaks=peaks,
        two_theta=np.asarray(spectrum.two_theta, dtype=float),
        intensity_raw=np.asarray(spectrum.intensity, dtype=float),
        baseline=prepared["baseline"],
        intensity_corrected=prepared["corrected"],
    )


def analyze_spectra(
    spectra: list[XRDSpectrum],
    config: XRDAnalysisConfig | None = None,
) -> list[SpectrumResult]:
    """Analyze each pattern, recording a failure row instead of aborting the batch."""

    config = config or XRDAnalysisConfig()
    results: list[SpectrumResult] = []
    for spectrum in spectra:
        try:
            results.append(analyze_spectrum(spectrum, config))
        except Exception as exc:
            logger.exception("XRD analysis failed for %s", spectrum.sample_id)
            features = _blank_features(
                spectrum.sample_id,
                spectrum.source_path.rsplit("/", 1)[-1],
                "analysis_failed",
                config.wavelength_A,
            )
            features["assignment_flag"] = f"analysis_failed: {exc}"
            results.append(SpectrumResult(sample_id=spectrum.sample_id, features=features))
    return results


def results_to_frames(results: list[SpectrumResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the sample-level feature table and the long peak table."""

    features = pd.DataFrame([result.features for result in results])
    if features.empty:
        features = pd.DataFrame(columns=FEATURE_COLUMNS)
    else:
        features = features.reindex(columns=FEATURE_COLUMNS)
        features = features.sort_values("sample_id").reset_index(drop=True)
    peak_rows = [row for result in results for row in result.peak_rows]
    peaks = pd.DataFrame(peak_rows)
    if not peaks.empty:
        peaks = peaks.sort_values(["sample_id", "peak_name"]).reset_index(drop=True)
    return features, peaks


def summarize_features(features: pd.DataFrame) -> list[str]:
    """Short textual summary of the extracted XRD metrics."""

    if features.empty:
        return ["No XRD features were extracted."]
    lines = [f"Samples analyzed: {len(features)}"]
    flags = features["assignment_flag"].value_counts(dropna=False)
    lines.append("Basal-peak assignments: " + ", ".join(f"{name}={count}" for name, count in flags.items()))
    for column, label, digits in (
        ("c_axis_A", "c axis (Å)", 3),
        ("d_002_A", "d(002) (Å)", 3),
        ("mxene_002_fwhm_deg", "(002) FWHM (°)", 3),
        ("crystallite_size_002_nm", "apparent (002) crystallite size (nm)", 2),
        ("max_impurity_index", "MAX intensity index", 3),
        ("tio2_impurity_index", "TiO2 intensity index", 3),
    ):
        values = pd.to_numeric(features[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            lines.append(f"{label}: not measured")
            continue
        lines.append(
            f"{label}: median {finite.median():.{digits}f}, "
            f"range {finite.min():.{digits}f}–{finite.max():.{digits}f} (n={len(finite)})"
        )
    return lines


def plot_spectra(results: list[SpectrumResult], output_dir) -> list[str]:
    """Save a basal-region overlay and a few fully labeled example patterns."""

    from pathlib import Path

    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    overlay = _plot_basal_overlay(results, plot_dir / "basal_002_overlay.png")
    if overlay is not None:
        written.append(str(overlay))
    for result in _example_results(results, count=4):
        path = plot_dir / f"spectrum_{_slug(result.sample_id)}.png"
        _plot_one_spectrum(result, path)
        written.append(str(path))
    return written


def _example_results(results: list[SpectrumResult], count: int) -> list[SpectrumResult]:
    usable = [result for result in results if np.isfinite(result.features.get("c_axis_A", np.nan))]
    pool = usable or [result for result in results if result.two_theta is not None]
    if not pool:
        return []
    pool = sorted(pool, key=lambda result: result.features.get("c_axis_A", 0.0))
    if len(pool) <= count:
        return pool
    indices = np.linspace(0, len(pool) - 1, count)
    return [pool[int(round(index))] for index in indices]


def _plot_basal_overlay(results: list[SpectrumResult], path) -> object | None:
    drawable = [result for result in results if result.two_theta is not None]
    if not drawable:
        return None
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    c_values = np.array([result.features.get("c_axis_A", np.nan) for result in drawable], dtype=float)
    finite = c_values[np.isfinite(c_values)]
    normalize = None
    if finite.size:
        normalize = plt.Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
    colormap = plt.get_cmap("viridis")
    for result, c_axis in zip(drawable, c_values):
        mask = (result.two_theta >= 4.0) & (result.two_theta <= 12.0)
        if int(np.count_nonzero(mask)) < 5:
            continue
        y = np.asarray(result.intensity_corrected)[mask]
        scale = float(np.nanmax(y))
        if not np.isfinite(scale) or scale <= 0:
            continue
        color = colormap(normalize(c_axis)) if normalize is not None and np.isfinite(c_axis) else "#355070"
        axis.plot(result.two_theta[mask], y / scale, color=color, alpha=0.55 if len(drawable) > 12 else 0.85, lw=1.1)
    axis.set_xlabel("2θ (degrees)")
    axis.set_ylabel("Baseline-corrected intensity / (002) max")
    axis.set_title("MXene basal region")
    axis.set_xlim(4.0, 12.0)
    if normalize is not None:
        mappable = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
        mappable.set_array([])
        colorbar = figure.colorbar(mappable, ax=axis)
        colorbar.set_label("c axis (Å)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _plot_one_spectrum(result: SpectrumResult, path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    axes[0].plot(result.two_theta, result.intensity_raw, color="#4c4c4c", lw=1.0, label="measured")
    axes[0].plot(result.two_theta, result.baseline, color="#c44900", lw=1.1, label="baseline")
    axes[0].set_ylabel("Intensity")
    axes[0].legend(frameon=False)
    axes[0].set_title(f"{result.sample_id}  ({result.features.get('assignment_flag', '')})")
    axes[1].plot(result.two_theta, result.intensity_corrected, color="#1f4e79", lw=1.0)
    for peak in result.peaks:
        if not peak.detected or not np.isfinite(peak.center_deg):
            continue
        axes[1].axvline(peak.center_deg, color="#b03a2e", lw=0.7, alpha=0.8)
        axes[1].text(peak.center_deg, axes[1].get_ylim()[1], peak.name, rotation=90, va="top", ha="right", fontsize=7)
    axes[1].set_xlabel("2θ (degrees)")
    axes[1].set_ylabel("Baseline-corrected intensity")
    c_axis = result.features.get("c_axis_A", np.nan)
    fwhm = result.features.get("mxene_002_fwhm_deg", np.nan)
    axes[1].text(
        0.99,
        0.95,
        f"c = {_format_number(c_axis, 2)} Å\nFWHM(002) = {_format_number(fwhm, 3)}°",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _format_number(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Analyze one XRD file and print peak metrics.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH_CU_KA1)
    parser.add_argument("--baseline", choices=["opening", "als"], default="opening")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spectra = __import__("loader").load_xrd_file(args.path)
    config = XRDAnalysisConfig(wavelength_A=args.wavelength, baseline_method=args.baseline)
    for spectrum in spectra:
        result = analyze_spectrum(spectrum, config)
        print(pd.DataFrame([result.features]).T.to_string(header=False))


if __name__ == "__main__":
    main()
