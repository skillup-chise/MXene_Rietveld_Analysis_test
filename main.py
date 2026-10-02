#!/usr/bin/env python3
"""Run MXene XRD preprocessing, peak analysis, and synthesis-parameter ranking.

Place experimental patterns in ``data/xrd/`` and conditions in
``data/synthesis_params.csv``, then run ``python main.py``. ``python main.py
--demo`` runs the same steps on a labeled synthetic campaign. ``python
inspect_data.py`` prints formats and column names before a full fit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from correlation_analyzer import CorrelationConfig, run_correlation_analysis
from demo_data import DEMO_DIR, ensure_demo_dataset
from loader import format_inspection_report, inspect_inputs, load_dataset
from xrd_analyzer import (
    WAVELENGTH_CU_KA1,
    XRDAnalysisConfig,
    analyze_spectra,
    plot_spectra,
    results_to_frames,
    summarize_features,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MXene XRD patterns and rank synthesis parameters.")
    parser.add_argument("--xrd-dir", type=Path, default=Path("data/xrd"), help="Directory of XRD CSV, Excel, or text files")
    parser.add_argument("--synthesis", type=Path, default=Path("data/synthesis_params.csv"), help="Synthesis-parameter table")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory for tables and plots")
    parser.add_argument("--wavelength", type=float, default=WAVELENGTH_CU_KA1, help="X-ray wavelength in angstroms (Cu Kα1 = 1.5406)")
    parser.add_argument("--baseline", choices=["opening", "als"], default="opening", help="Background method")
    parser.add_argument("--baseline-window", type=float, default=8.0, help="Morphological baseline window in degrees 2θ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the forest and permutation importance")
    parser.add_argument("--demo", action="store_true", help="Analyze the bundled synthetic demo dataset")
    parser.add_argument("--inspect", action="store_true", help="Print file formats and column names, then stop")
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.demo:
        ensure_demo_dataset(DEMO_DIR)
        return DEMO_DIR / "xrd", DEMO_DIR / "synthesis_params.csv", "synthetic-demo"
    return args.xrd_dir, args.synthesis, "experimental"


def run_pipeline(
    xrd_dir: Path,
    synthesis_path: Path,
    output_dir: Path,
    dataset_label: str,
    analysis: XRDAnalysisConfig,
    correlation: CorrelationConfig,
) -> dict:
    """Execute loading, peak analysis, optional ranking, and file export."""

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(xrd_dir, synthesis_path)
    (output_dir / "inspection_report.json").write_text(
        json.dumps(bundle.inspection, indent=2),
        encoding="utf-8",
    )
    results = analyze_spectra(bundle.spectra, analysis)
    features, peaks = results_to_frames(results)
    features.to_csv(output_dir / "xrd_features.csv", index=False)
    peaks.to_csv(output_dir / "peaks_long.csv", index=False)
    plot_paths = plot_spectra(results, output_dir)

    if bundle.synthesis is not None:
        correlation_result = run_correlation_analysis(
            features,
            bundle.synthesis,
            output_dir,
            dataset_label=dataset_label,
            config=correlation,
        )
    else:
        correlation_result = None
        report = ["# MXene XRD report", ""]
        if dataset_label == "synthetic-demo":
            report.append("Dataset: synthetic demonstration.")
        report.append("")
        report.append("No synthesis table was found, so parameter ranking was skipped.")
        report.append("")
        for line in summarize_features(features):
            report.append(f"- {line}")
        (output_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    return {
        "features": features,
        "peaks": peaks,
        "inspection": bundle.inspection,
        "plot_paths": plot_paths,
        "correlation": correlation_result,
        "output_dir": output_dir,
    }


def _print_rank_preview(result: dict) -> None:
    correlation = result["correlation"]
    if correlation is None or correlation.importance.empty:
        print("Parameter ranking was not produced.")
        return
    print("Top synthesis parameters by target:")
    for target, group in correlation.importance.groupby("target"):
        top = group.nsmallest(3, "rank")
        pieces = [
            f"{row.feature} (r={row.pearson_r:.2f})" if pd_isfinite(row.pearson_r) else str(row.feature)
            for row in top.itertuples()
        ]
        print(f"  {target}: {', '.join(pieces)}")


def pd_isfinite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def main(argv: list[str] | None = None) -> dict | None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    xrd_dir, synthesis_path, dataset_label = resolve_inputs(args)
    if dataset_label != "synthetic-demo" and (xrd_dir.parent / "SYNTHETIC.txt").exists():
        dataset_label = "synthetic-demo"
        print("Detected SYNTHETIC.txt beside the XRD folder. This run is a simulation.\n")
    if dataset_label == "synthetic-demo":
        print("Using SYNTHETIC data. These are not experimental measurements.\n")

    if args.inspect:
        report = inspect_inputs(xrd_dir, synthesis_path)
        print(format_inspection_report(report))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"inspection": report}

    analysis = XRDAnalysisConfig(
        wavelength_A=args.wavelength,
        baseline_method=args.baseline,
        baseline_window_deg=args.baseline_window,
    )
    correlation = CorrelationConfig(random_state=args.seed)
    result = run_pipeline(xrd_dir, synthesis_path, args.output, dataset_label, analysis, correlation)
    print()
    for line in summarize_features(result["features"]):
        print(line)
    print()
    _print_rank_preview(result)
    print(f"\nWrote tables and plots to {args.output}")
    return result


if __name__ == "__main__":
    main()
