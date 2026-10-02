"""Tests for joining synthesis tables and ranking known drivers."""

from pathlib import Path

import pandas as pd

from correlation_analyzer import CorrelationConfig, merge_features, rank_parameters, run_correlation_analysis
from demo_data import simulate_campaign


def test_merge_matches_ids_that_differ_by_case_and_suffix() -> None:
    features = pd.DataFrame({"sample_id": ["MX-001", "MX-002"], "c_axis_A": [24.0, 27.0]})
    synthesis = pd.DataFrame({"sample_id": ["mx-001_xrd", "MX-002"], "etching_time_h": [12, 48]})
    merged = merge_features(features, synthesis)
    assert merged.loc[merged["sample_id"] == "MX-001", "etching_time_h"].item() == 12
    assert merged["etching_time_h"].notna().all()


def test_known_drivers_rank_at_the_top() -> None:
    campaign = simulate_campaign(n=200, seed=0)
    frame = campaign.rename(
        columns={
            "true_c_axis_A": "c_axis_A",
            "true_fwhm_002_deg": "mxene_002_fwhm_deg",
            "true_max_impurity_index": "max_impurity_index",
            "true_tio2_impurity_index": "tio2_impurity_index",
        }
    )
    ranked = rank_parameters(
        frame,
        CorrelationConfig(
            targets=("c_axis_A", "mxene_002_fwhm_deg", "max_impurity_index", "tio2_impurity_index"),
            n_estimators=80,
            permutation_repeats=5,
            random_state=0,
        ),
    )
    importance = ranked["importance"]
    c_axis = importance[importance["target"] == "c_axis_A"].sort_values("rank")
    assert c_axis.iloc[0]["feature"] == "etching_time_h"
    assert c_axis.iloc[0]["pearson_r"] > 0.5
    flask = c_axis[c_axis["feature"] == "flask_volume_ml"].iloc[0]
    assert flask["pearson_r"] < -0.15

    fwhm = importance[importance["target"] == "mxene_002_fwhm_deg"]
    time_on_fwhm = fwhm[fwhm["feature"] == "etching_time_h"].iloc[0]
    assert time_on_fwhm["pearson_r"] < 0

    oxide = importance[importance["target"] == "tio2_impurity_index"]
    argon = oxide[oxide["feature"] == "atmosphere_argon"].iloc[0]
    assert argon["pearson_r"] < -0.25


def test_run_writes_ranking_files(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(12)],
            "c_axis_A": [20 + i * 0.4 for i in range(12)],
            "mxene_002_fwhm_deg": [1.2 - i * 0.04 for i in range(12)],
        }
    )
    synthesis = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(12)],
            "etching_time_h": [6 + i * 3 for i in range(12)],
            "flask_volume_ml": [500 - i * 20 for i in range(12)],
        }
    )
    result = run_correlation_analysis(
        features,
        synthesis,
        tmp_path,
        dataset_label="experimental",
        config=CorrelationConfig(
            targets=("c_axis_A", "mxene_002_fwhm_deg"),
            n_estimators=40,
            permutation_repeats=4,
        ),
    )
    assert (tmp_path / "ranked_parameters.csv").exists()
    assert (tmp_path / "analysis_report.md").exists()
    assert (tmp_path / "plots" / "correlation_heatmap.png").exists()
    report = (tmp_path / "analysis_report.md").read_text(encoding="utf-8")
    assert "experimental" in report
    assert not result.importance.empty
