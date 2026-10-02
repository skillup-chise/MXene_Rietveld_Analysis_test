"""Tests for XRD and synthesis-file ingestion."""

from pathlib import Path

import numpy as np
import pandas as pd

from loader import canon_sample_id, inspect_inputs, load_dataset, load_xrd_file, load_synthesis_table


def _angle_grid() -> np.ndarray:
    return np.linspace(5.0, 20.0, 40)


def test_single_csv_with_comments(tmp_path: Path) -> None:
    angle = _angle_grid()
    intensity = 10.0 + angle
    frame = pd.DataFrame({"2θ": angle, "Intensity (a.u.)": intensity})
    path = tmp_path / "MX-001_xrd.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# exported from the diffractometer\n")
        frame.to_csv(handle, index=False)

    spectra = load_xrd_file(path)
    assert len(spectra) == 1
    assert spectra[0].sample_id == "MX-001"
    assert spectra[0].two_theta[0] == angle[0]
    assert spectra[0].metadata["layout"] == "single"


def test_headerless_xy_and_wide_csv(tmp_path: Path) -> None:
    angle = _angle_grid()
    xy = tmp_path / "MX-007.xy"
    xy.write_text("\n".join(f"{a:.4f}  {100 + a:.2f}" for a in angle), encoding="utf-8")
    loaded = load_xrd_file(xy)
    assert loaded[0].sample_id == "MX-007"
    assert loaded[0].two_theta.size == angle.size

    wide = pd.DataFrame({"two_theta": angle, "MX-A": angle, "MX-B": angle * 2})
    wide_path = tmp_path / "batch.csv"
    wide.to_csv(wide_path, index=False)
    spectra = load_xrd_file(wide_path)
    assert {spectrum.sample_id for spectrum in spectra} == {"MX-A", "MX-B"}


def test_long_csv_and_excel_sheet(tmp_path: Path) -> None:
    angle = _angle_grid()
    rows = []
    for sample_id, scale in (("S1", 1.0), ("S2", 2.0)):
        for value in angle:
            rows.append({"sample_id": sample_id, "two_theta": value, "counts": scale * value})
    long_path = tmp_path / "stacked.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False)
    spectra = load_xrd_file(long_path)
    assert {spectrum.sample_id for spectrum in spectra} == {"S1", "S2"}
    s2 = next(spectrum for spectrum in spectra if spectrum.sample_id == "S2")
    assert np.isclose(s2.intensity[0], 2.0 * angle[0])

    excel_path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame({"Angle (deg)": angle, "cps": np.ones_like(angle)}).to_excel(
            writer, sheet_name="MX-10", index=False
        )
        pd.DataFrame({"Angle (deg)": angle, "cps": 2 * np.ones_like(angle)}).to_excel(
            writer, sheet_name="MX-11", index=False
        )
    excel_spectra = load_xrd_file(excel_path)
    assert {spectrum.sample_id for spectrum in excel_spectra} == {"MX-10", "MX-11"}


def test_duplicate_angles_are_averaged(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "2theta": [5.0, 5.0, 6.0],
            "intensity": [10.0, 30.0, 5.0],
        }
    )
    path = tmp_path / "dup.csv"
    frame.to_csv(path, index=False)
    spectrum = load_xrd_file(path)[0]
    assert spectrum.two_theta.tolist() == [5.0, 6.0]
    assert spectrum.intensity[0] == 20.0


def test_synthesis_types_and_id_alignment(tmp_path: Path) -> None:
    xrd = tmp_path / "xrd"
    xrd.mkdir()
    angle = _angle_grid()
    pd.DataFrame({"2theta": angle, "intensity": angle}).to_csv(xrd / "MX-001.csv", index=False)
    synthesis = pd.DataFrame(
        {
            "Sample ID": ["mx-001", "MX-999"],
            "stirring_speed_rpm": [400, 800],
            "atmosphere": ["air", "argon"],
            "notes": ["first etch on Monday", "repeat after lunch"],
        }
    )
    synthesis_path = tmp_path / "synthesis.csv"
    synthesis.to_csv(synthesis_path, index=False)
    table, id_column, description = load_synthesis_table(synthesis_path)
    assert id_column == "Sample ID"
    assert "stirring_speed_rpm" in description["numeric_parameters"]
    assert "atmosphere" in description["categorical_parameters"]
    assert "notes" in description["ignored_columns"]
    assert canon_sample_id(table.loc[0, "sample_id"]) == canon_sample_id("MX-001")

    report = inspect_inputs(xrd, synthesis_path)
    assert report["n_spectra"] == 1
    assert report["alignment"]["n_matched"] == 1
    assert report["alignment"]["unmatched_synthesis"] == ["MX-999"]

    bundle = load_dataset(xrd, synthesis_path)
    assert bundle.spectra[0].sample_id == "MX-001"
    assert bundle.synthesis is not None
