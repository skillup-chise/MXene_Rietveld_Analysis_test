# MXene XRD analysis

This pipeline loads powder XRD patterns of MXenes, measures the basal (002) reflection and common impurity lines, and ranks which synthesis settings track those metrics.

It is a peak-analysis workflow (background, smoothing, pseudo-Voigt fits, intensity ratios). It is not a Rietveld refinement and it does not report phase weight fractions.

The repository does not ship experimental patterns. `data/xrd/` is where those files go. A labeled synthetic Ti3C2Tx-style campaign lives in `data/demo/` so the code can be inspected and run before real data are added.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer.

## Inspect files first

```bash
python inspect_data.py --xrd-dir data/xrd --synthesis data/synthesis_params.csv
python inspect_data.py --demo
```

The inspector prints the format, the columns it will use as 2θ and intensity, the sample ids, the 2θ range, and which synthesis columns are numeric or categorical. Sample ids are matched with case and separators ignored, and a trailing `_xrd` / `-scan` suffix is ignored.

### XRD layouts that load

* One spectrum per file. CSV, `.txt`, `.dat`, `.xy`, or Excel. A 2θ column and an intensity column. The sample id is the file name (`MX-001.csv`, `MX-001_xrd.csv`).
* A wide table: one 2θ column and one intensity column per sample. Headers are the sample ids.
* A long table: `sample_id`, 2θ, and intensity stacked in one file.
* A multi-sheet workbook: one pattern per sheet. The sheet name is the sample id.

Headerless two-column text is treated as 2θ then intensity. Lines starting with `#` are skipped. Recognized angle names include `2theta`, `two_theta`, `2θ`, and `Angle (deg)`. Recognized intensity names include `intensity`, `counts`, and `cps`.

### Synthesis table

CSV or Excel, one row per sample. The id column may be named `sample_id`, `sample`, or `Sample ID`. Other columns are the conditions. Numeric columns stay numeric. Text columns with only a few repeated values (for example `air` / `argon`) are treated as categories. Free-text notes are ignored.

Example columns:

```text
sample_id,stirrer_bar_size_mm,stirring_speed_rpm,flask_volume_ml,etching_time_h,etching_temperature_C,hcl_concentration_M,lif_to_max_molar_ratio,washing_cycles,atmosphere
```

## Run

```bash
python main.py --xrd-dir data/xrd --synthesis data/synthesis_params.csv --output outputs
python main.py --demo
```

Useful flags:

* `--wavelength 1.5406` — angstroms. The default is Cu Kα1. Use `1.5418` for a Cu Kα average, or `1.789` for Co Kα.
* `--baseline opening` or `--baseline als` — morphological opening (default) or asymmetric least squares.
* `--baseline-window 8` — opening width in degrees. It should be wider than the peaks.
* `--inspect` — same report as `inspect_data.py`, written to `outputs/inspection_report.json`, then stop.

## What is measured

Default windows are for Ti3AlC2 → Ti3C2Tx. Change `XRDAnalysisConfig` and `DEFAULT_REFERENCE_WINDOWS` in `xrd_analyzer.py` for another MAX or MXene chemistry.

* Background subtraction, then a pseudo-Voigt fit (Gaussian and Lorentzian sharing one FWHM) on the baseline-corrected counts. If the fit fails, the peak position and FWHM are read directly from the half-maximum crossings.
* The lowest-angle peak between 4.5° and 10.8° 2θ is the interlayer (002) line. `d = λ / (2 sin θ)` and `c = 2 d(002)`.
* A second basal peak within 0.5° of 9.52° is recorded as residual MAX (002). A single peak sitting on that position is still used for `c`, and the assignment flag says the spacing matches unetched MAX.
* Residual MAX is quantified mainly from the (104) line near 39.0°. The MAX index is `I(104) / (I(104) + I(002))`.
* Anatase (101) near 25.3° and rutile (110) near 27.4° form the TiO2 index in the same way.
* Apparent crystallite size uses the Scherrer equation `D = 0.9 λ / (β cos θ)` with β the fitted FWHM in radians. Instrumental broadening is not removed.
* Crystallinity index is `1 / FWHM(002)`. Sharper basal peaks score higher.
* The predicted (004) position is checked against the pattern. `harmonic_004_delta_deg` near zero supports the (002) indexing.

Intensity indices are not weight percent. Absolute counts are not compared across samples.

## Synthesis ranking

XRD metrics are joined to the synthesis table on the sample id. The pipeline then writes:

* Pearson and Spearman correlations
* Random-forest permutation importance, when at least 8 complete rows are available
* Standardized ridge coefficients (change in the metric per 1 standard deviation of the input)
* A heatmap, importance bars, and trend plots

The primary rank is permutation importance. The Pearson sign shows the direction. With fewer than 8 complete rows the rank falls back to `|Pearson r|`. Cross-validated R² is reported so a rank built on a model that does not predict held-out rows is visible.

These are associations in the supplied table. Conditions that were changed together can share credit. The report lists pairs with `|r| ≥ 0.85`.

## Outputs

`outputs/` contains:

| File | Contents |
| --- | --- |
| `xrd_features.csv` | One row per sample: position, FWHM, c axis, crystallite size, impurity indices |
| `peaks_long.csv` | One row per sample per reflection |
| `merged_dataset.csv` | XRD metrics joined to synthesis conditions |
| `ranked_parameters.csv` | Ranked conditions for each XRD metric |
| `correlation_pearson.csv`, `correlation_spearman.csv` | Correlation matrices |
| `ridge_coefficients.csv`, `model_scores.csv` | Multivariate fit details |
| `analysis_report.md`, `summary.json` | Written summary |
| `plots/` | Heatmap, importance bars, trend plots, basal-region overlay, example patterns |

## Synthetic demo

`python main.py --demo` analyzes `data/demo/`. `SYNTHETIC.txt` and the `# SYNTHETIC` header on each pattern mark the campaign as simulated. `generator_truth.csv` is the answer key; the fitter does not read it.

In that generator, etching time is the strongest driver of a larger c axis and a lower MAX index. Temperature and the LiF:MAX ratio push the same way. A larger flask works against conversion. Stirring speed narrows the (002) line. Air, temperature, and time raise the anatase-like line; argon lowers it. A successful run should put etching time at or near the top of the c-axis ranking. That is a software check, not an etching recipe.

Regenerate the files with:

```bash
python demo_data.py --output data/demo --n 36 --seed 42
```

## Tests

```bash
python -m pytest
```

## Layout

* `loader.py` — read and align spectra and synthesis tables
* `xrd_analyzer.py` — baseline, peak fits, c axis, impurity indices, spectrum plots
* `correlation_analyzer.py` — correlations, forest importance, ridge model, ranking plots
* `main.py` — end-to-end run
* `inspect_data.py` — format and column report
* `demo_data.py` — synthetic campaign
