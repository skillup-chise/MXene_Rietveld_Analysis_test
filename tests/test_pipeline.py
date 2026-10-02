"""End-to-end check on a small synthetic campaign."""

from pathlib import Path

import numpy as np
import pandas as pd

from demo_data import generate_demo_dataset
from main import main


def test_pipeline_recovers_simulated_c_axis_and_writes_outputs(tmp_path: Path) -> None:
    demo = generate_demo_dataset(tmp_path / "demo", n=4, seed=3, step=0.03)
    output = tmp_path / "outputs"
    result = main(
        [
            "--xrd-dir",
            str(demo / "xrd"),
            "--synthesis",
            str(demo / "synthesis_params.csv"),
            "--output",
            str(output),
            "--seed",
            "3",
        ]
    )
    features = result["features"]
    truth = pd.read_csv(demo / "generator_truth.csv")
    compared = features.merge(truth[["sample_id", "true_c_axis_A", "true_max_impurity_index"]], on="sample_id")
    c_error = np.abs(compared["c_axis_A"] - compared["true_c_axis_A"])
    assert c_error.median() < 0.2
    assert c_error.max() < 0.45
    spearman = compared["max_impurity_index"].corr(compared["true_max_impurity_index"], method="spearman")
    assert spearman > 0.8

    for name in (
        "xrd_features.csv",
        "peaks_long.csv",
        "merged_dataset.csv",
        "ranked_parameters.csv",
        "analysis_report.md",
        "summary.json",
        "plots/basal_002_overlay.png",
        "plots/correlation_heatmap.png",
    ):
        assert (output / name).exists(), name
    report = (output / "analysis_report.md").read_text(encoding="utf-8")
    assert "synthetic demonstration" in report
