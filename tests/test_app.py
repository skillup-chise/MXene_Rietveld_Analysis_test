"""Tests for the Streamlit app's data preparation, without opening a browser."""

from pathlib import Path

import pandas as pd

from app import align_synthesis, default_synthesis_frame, execute_analysis, stage_demo_subset


def test_default_grid_has_the_synthesis_fields() -> None:
    frame = default_synthesis_frame(["MX-001", "MX-002"])
    assert list(frame["sample_id"]) == ["MX-001", "MX-002"]
    assert frame.loc[0, "stirrer_bar_size_mm"] == 25.0
    assert frame.loc[0, "stirring_speed_rpm"] == 400.0
    assert frame.loc[0, "flask_volume_ml"] == 100.0
    assert frame.loc[0, "atmosphere"] == "air"


def test_uploaded_conditions_overlay_matching_ids() -> None:
    uploaded = pd.DataFrame(
        {
            "sample_id": ["mx-001", "MX-003"],
            "stirring_speed_rpm": [650, 800],
            "flask_volume_ml": [250, 500],
            "atmosphere": ["argon", "air"],
            "etchant_note": ["LiF-HCl", "LiF-HCl"],
        }
    )
    frame = align_synthesis(["MX-001", "MX-002"], uploaded)
    assert frame.loc[frame["sample_id"] == "MX-001", "stirring_speed_rpm"].item() == 650
    assert frame.loc[frame["sample_id"] == "MX-001", "flask_volume_ml"].item() == 250
    assert frame.loc[frame["sample_id"] == "MX-001", "atmosphere"].item() == "argon"
    assert frame.loc[frame["sample_id"] == "MX-002", "stirring_speed_rpm"].item() == 400
    assert "etchant_note" in frame.columns


def test_demo_subset_and_analysis(tmp_path: Path) -> None:
    xrd_dir = stage_demo_subset(tmp_path / "demo", n=2)
    assert len(list(xrd_dir.glob("*.csv"))) == 2
    synthesis = default_synthesis_frame(["MX-001", "MX-002"])
    synthesis.loc[0, "etching_time_h"] = 12
    synthesis.loc[1, "etching_time_h"] = 48
    result = execute_analysis(xrd_dir, synthesis, tmp_path / "outputs")
    assert set(result["features"]["sample_id"]) == {"MX-001", "MX-002"}
    assert result["features"]["c_axis_A"].notna().all()
    assert (tmp_path / "outputs" / "xrd_features.csv").exists()
    assert result["correlation"] is not None
