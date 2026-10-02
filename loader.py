"""Load XRD spectra and synthesis-parameter tables.

The loader accepts the file layouts that powder-diffraction and lab notebooks
usually produce:

* one spectrum per CSV, Excel, or whitespace-separated text file
* a wide table with a shared 2θ column and one intensity column per sample
* a long table with ``sample_id``, 2θ, and intensity columns
* multi-sheet Excel workbooks (one sample per sheet)

Column names are inferred. Synthesis tables keep every experimental condition;
numeric columns stay numeric and low-cardinality text columns stay categorical.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

XRD_EXTENSIONS = {".csv", ".txt", ".dat", ".xy", ".xlsx", ".xls"}

# Names that mean "this column is intensity" rather than a sample id.
GENERIC_INTENSITY = {
    "intensity",
    "intensityau",
    "intensitycounts",
    "counts",
    "countsau",
    "cps",
    "i",
    "int",
    "y",
    "signal",
    "iobs",
    "iobsau",
    "count",
}

SAMPLE_ID_KEYS = {
    "sampleid",
    "sample",
    "samplename",
    "samplecode",
    "specimen",
    "specimenid",
    "filename",
    "name",
    "id",
}

# Text columns that are notes or identifiers, not synthesis factors.
NON_FEATURE_NAME = re.compile(
    r"(date|datetime|timestamp|operator|analyst|notes?|comment|remark|description|source)$",
    re.IGNORECASE,
)


@dataclass
class XRDSpectrum:
    """One powder pattern aligned to a sample id."""

    sample_id: str
    two_theta: np.ndarray
    intensity: np.ndarray
    source_path: str
    metadata: dict = field(default_factory=dict)

    def to_inspection_row(self) -> dict:
        two_theta = self.two_theta
        return {
            "sample_id": self.sample_id,
            "source_file": Path(self.source_path).name,
            "n_points": int(two_theta.size),
            "two_theta_min": float(two_theta.min()) if two_theta.size else None,
            "two_theta_max": float(two_theta.max()) if two_theta.size else None,
            "intensity_min": float(np.min(self.intensity)) if two_theta.size else None,
            "intensity_max": float(np.max(self.intensity)) if two_theta.size else None,
        }


@dataclass
class DatasetBundle:
    """Spectra plus the optional synthesis table and an inspection report."""

    spectra: list[XRDSpectrum]
    synthesis: pd.DataFrame | None
    id_column: str | None
    inspection: dict


def _simplify(name: object) -> str:
    text = str(name).casefold().replace("θ", "theta").replace("°", "deg")
    return re.sub(r"[^a-z0-9]+", "", text)


def is_two_theta_name(name: object) -> bool:
    simplified = _simplify(name)
    if simplified in {
        "2theta",
        "twotheta",
        "angle",
        "angledeg",
        "theta",
        "deg",
        "2thetadeg",
        "twothetadeg",
        "2thetaangle",
        "angle2theta",
    }:
        return True
    return "2theta" in simplified or "twotheta" in simplified


def is_intensity_name(name: object) -> bool:
    simplified = _simplify(name)
    if simplified in GENERIC_INTENSITY:
        return True
    return simplified.startswith("intensity") or simplified.startswith("counts") or simplified.startswith("cps")


def is_sample_id_name(name: object) -> bool:
    return _simplify(name) in SAMPLE_ID_KEYS


def canon_sample_id(value: object) -> str:
    """Normalize a sample id so filenames and spreadsheet ids can be joined."""

    text = str(value).strip()
    text = re.sub(r"(?i)[_\-\s]?(xrd|scan|data|spectrum)$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def sample_id_from_filename(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)[_\-\s]?(xrd|scan|data|spectrum)$", "", stem).strip()
    return cleaned or stem


def discover_xrd_files(xrd_dir: Path) -> list[Path]:
    """Return XRD files under ``xrd_dir``, skipping Excel lock files."""

    if not xrd_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(xrd_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(("~$", ".")):
            continue
        if path.suffix.lower() in XRD_EXTENSIONS:
            files.append(path)
    return files


def _header_is_data(frame: pd.DataFrame) -> bool:
    if frame.shape[1] == 0:
        return False
    numericish = 0
    for column in frame.columns:
        try:
            float(str(column).strip())
        except ValueError:
            continue
        numericish += 1
    return numericish >= 1 and numericish >= len(frame.columns) / 2


def detect_separator(path: Path) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
            if len(lines) >= 20:
                break
    if not lines:
        return ","
    sample = "\n".join(lines[:5])
    counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
    separator, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        return r"\s+"
    return separator


def read_text_table(path: Path) -> pd.DataFrame:
    """Read a delimited text file, including headerless ``.xy`` patterns."""

    separator = detect_separator(path)
    kwargs = {
        "sep": separator,
        "comment": "#",
        "engine": "python",
        "encoding": "utf-8-sig",
    }
    frame = pd.read_csv(path, **kwargs)
    if frame.empty:
        raise ValueError(f"No data rows in {path}")
    if _header_is_data(frame):
        frame = pd.read_csv(path, header=None, **kwargs)
        frame.columns = [f"column_{index + 1}" for index in range(frame.shape[1])]
    if separator == ";":
        frame = _convert_european_decimals(frame)
    return frame


def _convert_european_decimals(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        if converted[column].dtype != object:
            continue
        numeric = pd.to_numeric(
            converted[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if numeric.notna().mean() >= 0.8:
            converted[column] = numeric
    return converted


def load_xrd_file(path: Path) -> list[XRDSpectrum]:
    """Load every spectrum stored in one file."""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        spectra: list[XRDSpectrum] = []
        multiple = len(sheets) > 1
        for sheet_name, frame in sheets.items():
            default_id = str(sheet_name) if multiple else sample_id_from_filename(path)
            spectra.extend(_table_to_spectra(frame, path, default_id))
        if not spectra:
            raise ValueError(f"No XRD spectra found in {path}")
        return spectra

    frame = read_text_table(path)
    spectra = _table_to_spectra(frame, path, sample_id_from_filename(path))
    if not spectra:
        raise ValueError(f"No XRD spectra found in {path}")
    return spectra


def _table_to_spectra(frame: pd.DataFrame, path: Path, default_sample_id: str) -> list[XRDSpectrum]:
    if frame is None or frame.empty:
        raise ValueError(f"Empty table in {path}")
    working = frame.copy()
    working.columns = [str(column).strip() for column in working.columns]

    two_theta_col = _pick_column(working.columns, is_two_theta_name)
    sample_col = _pick_column(working.columns, is_sample_id_name)
    intensity_cols = [column for column in working.columns if is_intensity_name(column)]

    if two_theta_col is None:
        two_theta_col, guessed_intensity = _guess_xy_columns(working)
        if two_theta_col is None:
            raise ValueError(f"Could not find a 2θ column in {path.name}")
        if not intensity_cols and guessed_intensity is not None:
            intensity_cols = [guessed_intensity]

    if sample_col is not None and intensity_cols:
        return _spectra_from_long(working, path, two_theta_col, intensity_cols[0], sample_col)

    numeric_columns = [
        column
        for column in working.columns
        if column not in {two_theta_col, sample_col} and _mostly_numeric(working[column])
    ]
    if len(numeric_columns) >= 2 and not (len(intensity_cols) == 1 and len(numeric_columns) == 1):
        # A shared angle axis with one column of counts per sample.
        wide_columns = intensity_cols if len(intensity_cols) >= 2 else numeric_columns
        return _spectra_from_wide(working, path, two_theta_col, wide_columns, default_sample_id)

    intensity_col = intensity_cols[0] if intensity_cols else (numeric_columns[0] if numeric_columns else None)
    if intensity_col is None:
        raise ValueError(f"Could not find an intensity column in {path.name}")
    spectrum = _build_spectrum(
        sample_id=default_sample_id,
        two_theta=working[two_theta_col],
        intensity=working[intensity_col],
        path=path,
        metadata={"two_theta_column": two_theta_col, "intensity_column": intensity_col, "layout": "single"},
    )
    return [spectrum]


def _pick_column(columns, predicate) -> str | None:
    for column in columns:
        if predicate(column):
            return str(column)
    return None


def _mostly_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.notna().mean() >= 0.8)


def _guess_xy_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """Treat a monotonic degree-scale column as 2θ when names are uninformative."""

    candidates: list[tuple[str, float]] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < 5:
            continue
        diffs = np.diff(finite)
        if diffs.size == 0 or np.mean(diffs > 0) < 0.9:
            continue
        span = float(finite.max() - finite.min())
        upper = float(finite.max())
        if 2.0 <= span <= 175.0 and 0.0 <= upper <= 180.0:
            candidates.append((str(column), span))
    if not candidates:
        return None, None
    angle_col = max(candidates, key=lambda item: item[1])[0]
    others = [str(column) for column in frame.columns if str(column) != angle_col and _mostly_numeric(frame[column])]
    intensity_col = others[0] if others else None
    return angle_col, intensity_col


def _spectra_from_long(
    frame: pd.DataFrame,
    path: Path,
    two_theta_col: str,
    intensity_col: str,
    sample_col: str,
) -> list[XRDSpectrum]:
    spectra: list[XRDSpectrum] = []
    for sample_id, group in frame.groupby(frame[sample_col].astype(str).str.strip(), sort=False):
        if not str(sample_id):
            continue
        spectra.append(
            _build_spectrum(
                sample_id=str(sample_id),
                two_theta=group[two_theta_col],
                intensity=group[intensity_col],
                path=path,
                metadata={
                    "two_theta_column": two_theta_col,
                    "intensity_column": intensity_col,
                    "sample_column": sample_col,
                    "layout": "long",
                },
            )
        )
    return spectra


def _spectra_from_wide(
    frame: pd.DataFrame,
    path: Path,
    two_theta_col: str,
    intensity_columns: list[str],
    default_sample_id: str,
) -> list[XRDSpectrum]:
    spectra: list[XRDSpectrum] = []
    used: set[str] = set()
    generic_count = 0
    for column in intensity_columns:
        simplified = _simplify(column)
        if simplified in GENERIC_INTENSITY:
            generic_count += 1
            sample_id = default_sample_id if generic_count == 1 else f"{default_sample_id}__{generic_count}"
        else:
            sample_id = str(column).strip()
        if sample_id in used:
            sample_id = f"{sample_id}__{len(used) + 1}"
        used.add(sample_id)
        spectra.append(
            _build_spectrum(
                sample_id=sample_id,
                two_theta=frame[two_theta_col],
                intensity=frame[column],
                path=path,
                metadata={"two_theta_column": two_theta_col, "intensity_column": column, "layout": "wide"},
            )
        )
    return spectra


def _build_spectrum(
    sample_id: str,
    two_theta: pd.Series,
    intensity: pd.Series,
    path: Path,
    metadata: dict,
) -> XRDSpectrum:
    angle, counts = clean_xy(two_theta, intensity)
    if angle.size < 5:
        raise ValueError(f"Sample {sample_id} in {path.name} has fewer than 5 finite points")
    return XRDSpectrum(
        sample_id=str(sample_id).strip(),
        two_theta=angle,
        intensity=counts,
        source_path=str(path),
        metadata=metadata,
    )


def clean_xy(two_theta: pd.Series | np.ndarray, intensity: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite points, sort by 2θ, and average duplicate angles."""

    angle = pd.to_numeric(pd.Series(two_theta), errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(pd.Series(intensity), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(angle) & np.isfinite(counts)
    angle = angle[mask]
    counts = counts[mask]
    if angle.size == 0:
        return angle, counts
    order = np.argsort(angle, kind="mergesort")
    angle = angle[order]
    counts = counts[order]
    rounded = np.round(angle, 6)
    unique, inverse = np.unique(rounded, return_inverse=True)
    if unique.size != angle.size:
        summed = np.zeros(unique.size, dtype=float)
        occurrences = np.zeros(unique.size, dtype=float)
        np.add.at(summed, inverse, counts)
        np.add.at(occurrences, inverse, 1.0)
        angle = unique
        counts = summed / occurrences
    return angle, counts


def load_synthesis_table(path: Path) -> tuple[pd.DataFrame, str, dict]:
    """Load synthesis conditions and classify columns.

    Returns the table, the original id-column name, and a description of which
    columns are numeric, categorical, or ignored.
    """

    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        separator = detect_separator(path)
        frame = pd.read_csv(path, sep=separator, comment="#", engine="python", encoding="utf-8-sig")
        if separator == ";":
            frame = _convert_european_decimals(frame)
    if frame.empty:
        raise ValueError(f"Synthesis file {path} has no rows")
    frame.columns = [str(column).strip() for column in frame.columns]
    id_column = _pick_column(frame.columns, is_sample_id_name) or str(frame.columns[0])
    if id_column != "sample_id":
        if "sample_id" in frame.columns:
            raise ValueError(f"Synthesis file uses id column {id_column!r} and also has sample_id")
        frame = frame.rename(columns={id_column: "sample_id"})
    frame["sample_id"] = frame["sample_id"].map(lambda value: str(value).strip())
    frame = frame.drop_duplicates(subset=["sample_id"], keep="first")

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    ignored_columns: list[str] = []
    for column in frame.columns:
        if column == "sample_id":
            continue
        if NON_FEATURE_NAME.search(column):
            ignored_columns.append(column)
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() >= 0.8 and frame[column].notna().any():
            frame[column] = numeric
            numeric_columns.append(column)
            continue
        text = frame[column].astype(str).str.strip()
        text = text.mask(frame[column].isna() | text.isin(["", "nan", "None"]), other=pd.NA)
        unique = int(text.nunique(dropna=True))
        row_count = max(int(text.notna().sum()), 1)
        if unique == 0 or unique > max(8, int(0.5 * row_count)):
            ignored_columns.append(column)
            continue
        frame[column] = text
        categorical_columns.append(column)

    description = {
        "path": str(path),
        "id_column": id_column,
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "numeric_parameters": numeric_columns,
        "categorical_parameters": categorical_columns,
        "ignored_columns": ignored_columns,
        "sample_ids": frame["sample_id"].tolist(),
    }
    return frame, id_column, description


def _disambiguate_sample_ids(spectra: list[XRDSpectrum]) -> list[str]:
    """Rename repeated ids so later joins stay one-row-per-pattern."""

    seen: dict[str, int] = {}
    warnings: list[str] = []
    for spectrum in spectra:
        count = seen.get(spectrum.sample_id, 0) + 1
        seen[spectrum.sample_id] = count
        if count > 1:
            renamed = f"{spectrum.sample_id}__{count}"
            warnings.append(f"{spectrum.sample_id} -> {renamed}")
            spectrum.sample_id = renamed
    return warnings


def match_sample_ids(spectrum_ids: list[str], synthesis_ids: list[str]) -> dict:
    spectrum_map: dict[str, str] = {}
    for sample_id in spectrum_ids:
        spectrum_map.setdefault(canon_sample_id(sample_id), sample_id)
    synthesis_map: dict[str, str] = {}
    for sample_id in synthesis_ids:
        synthesis_map.setdefault(canon_sample_id(sample_id), sample_id)
    matched = [
        {"spectrum": sample_id, "synthesis": synthesis_map[key]}
        for key, sample_id in spectrum_map.items()
        if key in synthesis_map
    ]
    unmatched_spectra = [sample_id for key, sample_id in spectrum_map.items() if key not in synthesis_map]
    unmatched_synthesis = [sample_id for key, sample_id in synthesis_map.items() if key not in spectrum_map]
    return {
        "n_matched": len(matched),
        "matched": matched,
        "unmatched_spectra": unmatched_spectra,
        "unmatched_synthesis": unmatched_synthesis,
    }


def inspect_inputs(xrd_dir: Path | str, synthesis_path: Path | str | None = None) -> dict:
    """Describe file formats, column names, and id alignment without fitting peaks."""

    xrd_dir = Path(xrd_dir)
    files: list[dict] = []
    errors: list[dict] = []
    spectra: list[XRDSpectrum] = []
    for path in discover_xrd_files(xrd_dir):
        try:
            loaded = load_xrd_file(path)
        except Exception as exc:  # one bad export should not hide the rest of the folder
            errors.append({"path": str(path), "error": str(exc)})
            continue
        spectra.extend(loaded)
        files.append(
            {
                "path": str(path),
                "format": path.suffix.lower().lstrip("."),
                "layout": loaded[0].metadata.get("layout") if loaded else None,
                "columns_used": sorted(
                    {
                        str(value)
                        for spectrum in loaded
                        for key, value in spectrum.metadata.items()
                        if key.endswith("column")
                    }
                ),
                "sample_ids": [spectrum.sample_id for spectrum in loaded],
                "n_points": [int(spectrum.two_theta.size) for spectrum in loaded],
                "two_theta_min": [float(spectrum.two_theta.min()) for spectrum in loaded],
                "two_theta_max": [float(spectrum.two_theta.max()) for spectrum in loaded],
            }
        )

    synthesis_info = None
    alignment = None
    if synthesis_path is not None and Path(synthesis_path).exists():
        table, _id_column, synthesis_info = load_synthesis_table(Path(synthesis_path))
        alignment = match_sample_ids(
            [spectrum.sample_id for spectrum in spectra],
            table["sample_id"].tolist(),
        )
    elif synthesis_path is not None:
        synthesis_info = {"path": str(synthesis_path), "missing": True}

    return {
        "xrd_dir": str(xrd_dir),
        "xrd_dir_exists": xrd_dir.exists(),
        "n_files": len(files),
        "n_spectra": len(spectra),
        "files": files,
        "errors": errors,
        "synthesis": synthesis_info,
        "alignment": alignment,
        "expected_layouts": expected_layout_help(),
    }


def expected_layout_help() -> dict:
    return {
        "xrd_single_file": "CSV/TXT/XY/XLSX with a 2θ column and an intensity column. Sample id is the file name.",
        "xrd_wide": "One 2θ column plus one intensity column per sample. Column headers are the sample ids.",
        "xrd_long": "Columns sample_id, 2θ, and intensity, with many samples stacked in one file.",
        "recognized_angle_names": ["2theta", "two_theta", "2θ", "angle", "Angle (deg)"],
        "recognized_intensity_names": ["intensity", "counts", "cps", "I"],
        "synthesis": "A CSV/XLSX keyed by sample_id. Other columns are synthesis conditions.",
        "example_synthesis_columns": [
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
        ],
    }


def load_dataset(xrd_dir: Path | str, synthesis_path: Path | str | None = None) -> DatasetBundle:
    """Load spectra and, when present, the synthesis table."""

    xrd_dir = Path(xrd_dir)
    spectra: list[XRDSpectrum] = []
    errors: list[dict] = []
    for path in discover_xrd_files(xrd_dir):
        try:
            spectra.extend(load_xrd_file(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            logger.warning("Skipping %s: %s", path, exc)
    if not spectra:
        hint = (
            f"No XRD spectra found in {xrd_dir}. "
            "Place CSV, Excel, or text patterns in that folder, or run `python main.py --demo`."
        )
        raise FileNotFoundError(hint)

    renamed = _disambiguate_sample_ids(spectra)
    synthesis = None
    id_column = None
    synthesis_info = None
    alignment = None
    if synthesis_path is not None and Path(synthesis_path).exists():
        synthesis, id_column, synthesis_info = load_synthesis_table(Path(synthesis_path))
        alignment = match_sample_ids(
            [spectrum.sample_id for spectrum in spectra],
            synthesis["sample_id"].tolist(),
        )
    elif synthesis_path is not None:
        synthesis_info = {"path": str(synthesis_path), "missing": True}
        logger.warning("Synthesis file not found: %s. XRD metrics will still be computed.", synthesis_path)

    inspection = {
        "xrd_dir": str(xrd_dir),
        "n_files": len(discover_xrd_files(xrd_dir)),
        "n_spectra": len(spectra),
        "duplicate_ids_renamed": renamed,
        "errors": errors,
        "spectra": [spectrum.to_inspection_row() for spectrum in spectra],
        "synthesis": synthesis_info,
        "alignment": alignment,
    }
    return DatasetBundle(spectra=spectra, synthesis=synthesis, id_column=id_column, inspection=inspection)


def format_inspection_report(report: dict) -> str:
    """Render an inspection report as plain text for the terminal."""

    lines: list[str] = []
    lines.append(f"XRD directory: {report.get('xrd_dir')}")
    if not report.get("xrd_dir_exists", True):
        lines.append("That directory does not exist yet.")
    lines.append(f"Files read: {report.get('n_files', 0)}")
    lines.append(f"Spectra found: {report.get('n_spectra', 0)}")
    errors = report.get("errors") or []
    if errors:
        lines.append(f"Files skipped: {len(errors)}")
        for error in errors:
            lines.append(f"  - {error['path']}: {error['error']}")

    for file_info in report.get("files") or []:
        lines.append("")
        lines.append(f"File: {file_info['path']}")
        lines.append(f"  format: {file_info['format']}  layout: {file_info['layout']}")
        lines.append(f"  columns used: {', '.join(file_info['columns_used']) or '(inferred positions)'}")
        for sample_id, n_points, t_min, t_max in zip(
            file_info["sample_ids"],
            file_info["n_points"],
            file_info["two_theta_min"],
            file_info["two_theta_max"],
        ):
            lines.append(
                f"  sample {sample_id}: {n_points} points, 2θ {t_min:.3f}–{t_max:.3f}°"
            )

    synthesis = report.get("synthesis")
    lines.append("")
    if not synthesis:
        lines.append("Synthesis file: not provided")
    elif synthesis.get("missing"):
        lines.append(f"Synthesis file: missing ({synthesis.get('path')})")
    else:
        lines.append(f"Synthesis file: {synthesis.get('path')}")
        lines.append(f"  id column: {synthesis.get('id_column')}")
        lines.append(f"  rows: {synthesis.get('n_rows')}")
        lines.append(f"  numeric parameters: {', '.join(synthesis.get('numeric_parameters') or []) or '(none)'}")
        lines.append(
            "  categorical parameters: "
            + (", ".join(synthesis.get("categorical_parameters") or []) or "(none)")
        )
        ignored = synthesis.get("ignored_columns") or []
        if ignored:
            lines.append(f"  ignored columns: {', '.join(ignored)}")

    alignment = report.get("alignment")
    if alignment:
        lines.append("")
        lines.append(f"Matched sample ids: {alignment['n_matched']}")
        if alignment["unmatched_spectra"]:
            lines.append("Spectra without synthesis rows: " + ", ".join(alignment["unmatched_spectra"]))
        if alignment["unmatched_synthesis"]:
            lines.append("Synthesis rows without spectra: " + ", ".join(alignment["unmatched_synthesis"]))

    if report.get("n_spectra", 0) == 0:
        help_text = report.get("expected_layouts") or expected_layout_help()
        lines.append("")
        lines.append("No spectra were loaded. Expected layouts:")
        for key in ("xrd_single_file", "xrd_wide", "xrd_long", "synthesis"):
            lines.append(f"  - {help_text[key]}")
        lines.append("  example synthesis columns: " + ", ".join(help_text["example_synthesis_columns"]))
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect XRD and synthesis files.")
    parser.add_argument("--xrd-dir", type=Path, default=Path("data/xrd"))
    parser.add_argument("--synthesis", type=Path, default=Path("data/synthesis_params.csv"))
    args = parser.parse_args()
    report = inspect_inputs(args.xrd_dir, args.synthesis)
    print(format_inspection_report(report))


if __name__ == "__main__":
    main()
