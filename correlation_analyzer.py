"""Join XRD metrics to synthesis conditions and rank the conditions.

Pearson and Spearman correlations describe linear and monotonic association.
Random-forest permutation importance and a standardized ridge regression give a
multivariate view when several conditions were changed in the same campaign.
These rankings are associations in the supplied table. They are not, by
themselves, a causal proof that changing one setting will change the product.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from loader import canon_sample_id
from xrd_analyzer import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TARGET_LABELS = {
    "c_axis_A": "c-axis length (Å)",
    "mxene_002_fwhm_deg": "(002) FWHM (degrees)",
    "crystallite_size_002_nm": "apparent crystallite size (nm)",
    "max_impurity_index": "MAX impurity index",
    "tio2_impurity_index": "TiO2 impurity index",
    "crystallinity_index": "crystallinity index (1/FWHM)",
}

DEFAULT_TARGETS: tuple[str, ...] = tuple(TARGET_LABELS)

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class CorrelationConfig:
    """Modeling settings for the synthesis-parameter ranking."""

    targets: tuple[str, ...] = DEFAULT_TARGETS
    random_state: int = 42
    n_estimators: int = 400
    min_samples_leaf: int = 2
    permutation_repeats: int = 25
    min_samples_for_model: int = 8
    max_missing_feature_fraction: float = 0.2


@dataclass
class CorrelationResult:
    """Tables and plot paths produced by one correlation run."""

    merged: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    importance: pd.DataFrame
    coefficients: pd.DataFrame
    model_scores: pd.DataFrame
    collinear_pairs: pd.DataFrame
    plot_paths: list[str]
    dataset_label: str
    notes: list[str]


def merge_features(features: pd.DataFrame, synthesis: pd.DataFrame) -> pd.DataFrame:
    """Left-join XRD features to synthesis rows on a normalized sample id."""

    if features.empty:
        return features.copy()
    left = features.copy()
    left["_key"] = left["sample_id"].map(canon_sample_id)
    if synthesis is None or synthesis.empty:
        return left.drop(columns=["_key"])

    right = synthesis.copy()
    right["_key"] = right["sample_id"].map(canon_sample_id)
    duplicated = int(right["_key"].duplicated().sum())
    if duplicated:
        logger.warning("Dropping %s duplicate synthesis ids after normalization", duplicated)
        right = right.drop_duplicates(subset=["_key"], keep="first")
    right = right.drop(columns=["sample_id"])
    overlap = [column for column in right.columns if column in left.columns and column != "_key"]
    if overlap:
        right = right.drop(columns=overlap)
    merged = left.merge(right, on="_key", how="left")
    return merged.drop(columns=["_key"])


def _feature_columns(frame: pd.DataFrame, targets: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split synthesis columns into numeric and categorical predictors."""

    blocked = set(FEATURE_COLUMNS) | set(targets) | {"sample_id"}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in blocked or str(column).startswith("true_"):
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().sum() == 0 or float(series.nunique(dropna=True)) <= 1:
                continue
            numeric.append(column)
            continue
        unique = int(series.nunique(dropna=True))
        observed = int(series.notna().sum())
        if unique <= 1 or unique > max(8, int(0.5 * max(observed, 1))):
            continue
        categorical.append(column)
    return numeric, categorical


def _design_matrix(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    max_missing: float,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if numeric_columns:
        pieces.append(frame[numeric_columns].apply(pd.to_numeric, errors="coerce"))
    if categorical_columns:
        categories = frame[categorical_columns].copy()
        for column in categorical_columns:
            categories[column] = categories[column].astype("object").where(categories[column].notna(), other="unknown")
            categories[column] = categories[column].astype(str)
        pieces.append(pd.get_dummies(categories, drop_first=True, dtype=float))
    if not pieces:
        return pd.DataFrame(index=frame.index)
    design = pd.concat(pieces, axis=1)
    design = design.loc[:, design.notna().mean() >= 1.0 - max_missing]
    nunique = design.nunique(dropna=True)
    design = design.loc[:, nunique > 1]
    return design


def _correlation_tables(design: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = pd.DataFrame(index=design.columns, columns=outcomes.columns, dtype=float)
    spearman = pd.DataFrame(index=design.columns, columns=outcomes.columns, dtype=float)
    for target in outcomes.columns:
        y = outcomes[target]
        for feature in design.columns:
            pair = pd.concat([design[feature], y], axis=1).dropna()
            if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                pearson.loc[feature, target] = np.nan
                spearman.loc[feature, target] = np.nan
                continue
            pearson.loc[feature, target] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson")
            spearman.loc[feature, target] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return pearson.astype(float), spearman.astype(float)


def _collinear_pairs(design: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    if design.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])
    matrix = design.corr(method="pearson")
    rows: list[dict] = []
    columns = list(matrix.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = matrix.loc[left, right]
            if np.isfinite(value) and abs(value) >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "pearson_r": float(value)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "pearson_r"])


def _fit_target(
    design: pd.DataFrame,
    y: pd.Series,
    config: CorrelationConfig,
) -> dict | None:
    pair = design.copy()
    pair["_y"] = y
    pair = pair.dropna()
    if len(pair) < config.min_samples_for_model or pair.shape[1] < 2:
        return None
    if pair["_y"].nunique() < 2:
        return None
    x_matrix = pair.drop(columns=["_y"]).to_numpy(dtype=float)
    y_vector = pair["_y"].to_numpy(dtype=float)
    feature_names = list(pair.drop(columns=["_y"]).columns)
    forest = RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=1,
    )
    forest.fit(x_matrix, y_vector)
    in_sample = float(forest.score(x_matrix, y_vector))
    splits = 5 if len(pair) >= 20 else 3
    cv_mean = np.nan
    cv_std = np.nan
    if len(pair) // splits >= 2:
        folder = KFold(n_splits=splits, shuffle=True, random_state=config.random_state)
        scores = cross_val_score(forest, x_matrix, y_vector, cv=folder, scoring="r2")
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
    permuted = permutation_importance(
        forest,
        x_matrix,
        y_vector,
        n_repeats=config.permutation_repeats,
        random_state=config.random_state,
        scoring="r2",
        n_jobs=1,
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_matrix)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 16))
    ridge.fit(scaled, y_vector)
    return {
        "n_samples": int(len(pair)),
        "n_features": len(feature_names),
        "in_sample_r2": in_sample,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "features": feature_names,
        "importance_mean": permuted.importances_mean,
        "importance_std": permuted.importances_std,
        "ridge_coefficient": ridge.coef_,
        "ridge_alpha": float(ridge.alpha_),
    }


def rank_parameters(
    merged: pd.DataFrame,
    config: CorrelationConfig | None = None,
) -> dict:
    """Compute correlations, importances, and a per-target ranking."""

    config = config or CorrelationConfig()
    available_targets = [target for target in config.targets if target in merged.columns]
    numeric, categorical = _feature_columns(merged, tuple(available_targets))
    design = _design_matrix(merged, numeric, categorical, config.max_missing_feature_fraction)
    notes = [
        f"Numeric synthesis columns: {', '.join(numeric) if numeric else '(none)'}",
        f"Categorical synthesis columns: {', '.join(categorical) if categorical else '(none)'}",
    ]
    if design.empty or not available_targets:
        notes.append("Not enough synthesis columns and XRD targets to rank parameters.")
        empty = pd.DataFrame()
        return {
            "design_columns": [],
            "pearson": empty,
            "spearman": empty,
            "importance": empty,
            "coefficients": empty,
            "model_scores": empty,
            "collinear_pairs": empty,
            "notes": notes,
        }

    outcomes = merged[available_targets].apply(pd.to_numeric, errors="coerce")
    pearson, spearman = _correlation_tables(design, outcomes)
    collinear = _collinear_pairs(design)
    importance_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    score_rows: list[dict] = []

    for target in available_targets:
        fitted = _fit_target(design, outcomes[target], config)
        pearson_target = pearson[target] if target in pearson.columns else pd.Series(dtype=float)
        if fitted is None:
            score_rows.append(
                {
                    "target": target,
                    "n_samples": int(outcomes[target].notna().sum()),
                    "n_features": int(design.shape[1]),
                    "in_sample_r2": np.nan,
                    "cv_r2_mean": np.nan,
                    "cv_r2_std": np.nan,
                    "ranking_method": "abs_pearson",
                    "ridge_alpha": np.nan,
                }
            )
            order = pearson_target.abs().sort_values(ascending=False)
            for rank, feature in enumerate(order.index, start=1):
                importance_rows.append(
                    {
                        "target": target,
                        "feature": feature,
                        "rank": rank,
                        "ranking_method": "abs_pearson",
                        "permutation_importance_mean": np.nan,
                        "permutation_importance_std": np.nan,
                        "pearson_r": pearson_target.get(feature, np.nan),
                        "spearman_r": spearman[target].get(feature, np.nan) if target in spearman else np.nan,
                        "ridge_coefficient": np.nan,
                    }
                )
            notes.append(
                f"{target}: fewer than {config.min_samples_for_model} complete rows, so the ranking uses |Pearson r|."
            )
            continue

        score_rows.append(
            {
                "target": target,
                "n_samples": fitted["n_samples"],
                "n_features": fitted["n_features"],
                "in_sample_r2": fitted["in_sample_r2"],
                "cv_r2_mean": fitted["cv_r2_mean"],
                "cv_r2_std": fitted["cv_r2_std"],
                "ranking_method": "permutation_importance",
                "ridge_alpha": fitted["ridge_alpha"],
            }
        )
        order = np.argsort(-fitted["importance_mean"])
        for rank, index in enumerate(order, start=1):
            feature = fitted["features"][index]
            importance_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "rank": rank,
                    "ranking_method": "permutation_importance",
                    "permutation_importance_mean": float(fitted["importance_mean"][index]),
                    "permutation_importance_std": float(fitted["importance_std"][index]),
                    "pearson_r": float(pearson_target.get(feature, np.nan)),
                    "spearman_r": float(spearman[target].get(feature, np.nan)) if target in spearman else np.nan,
                    "ridge_coefficient": float(fitted["ridge_coefficient"][index]),
                }
            )
            coefficient_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "ridge_coefficient_per_std": float(fitted["ridge_coefficient"][index]),
                    "pearson_r": float(pearson_target.get(feature, np.nan)),
                }
            )

    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance = importance.sort_values(["target", "rank"]).reset_index(drop=True)
    return {
        "design_columns": list(design.columns),
        "design": design,
        "outcomes": outcomes,
        "pearson": pearson,
        "spearman": spearman,
        "importance": importance,
        "coefficients": pd.DataFrame(coefficient_rows),
        "model_scores": pd.DataFrame(score_rows),
        "collinear_pairs": collinear,
        "notes": notes,
    }


def run_correlation_analysis(
    features: pd.DataFrame,
    synthesis: pd.DataFrame | None,
    output_dir: Path | str,
    dataset_label: str = "experimental",
    config: CorrelationConfig | None = None,
) -> CorrelationResult:
    """Merge, rank, plot, and write the correlation products."""

    config = config or CorrelationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_features(features, synthesis if synthesis is not None else pd.DataFrame())
    ranked = rank_parameters(merged, config)
    plot_paths = _write_plots(merged, ranked, output_dir / "plots")
    notes = list(ranked["notes"])
    if dataset_label == "synthetic-demo":
        notes.append(
            "This table is a synthetic demonstration. The ranking is a check that the pipeline "
            "can recover the factors built into the generator. It is not an experimental synthesis map."
        )
    result = CorrelationResult(
        merged=merged,
        pearson=ranked["pearson"],
        spearman=ranked["spearman"],
        importance=ranked["importance"],
        coefficients=ranked["coefficients"],
        model_scores=ranked["model_scores"],
        collinear_pairs=ranked["collinear_pairs"],
        plot_paths=plot_paths,
        dataset_label=dataset_label,
        notes=notes,
    )
    _write_tables(result, output_dir)
    report = render_report(features, result)
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    summary = {
        "dataset_label": dataset_label,
        "n_samples": int(len(merged)),
        "notes": notes,
        "model_scores": _records(result.model_scores),
        "top_parameters": _top_parameter_records(result.importance, limit=5),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    cleaned = frame.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def _top_parameter_records(importance: pd.DataFrame, limit: int) -> dict:
    if importance is None or importance.empty:
        return {}
    grouped: dict[str, list[dict]] = {}
    for target, group in importance.groupby("target"):
        rows = group.nsmallest(limit, "rank")
        grouped[str(target)] = _records(rows)
    return grouped


def _write_tables(result: CorrelationResult, output_dir: Path) -> None:
    result.merged.to_csv(output_dir / "merged_dataset.csv", index=False)
    result.pearson.to_csv(output_dir / "correlation_pearson.csv")
    result.spearman.to_csv(output_dir / "correlation_spearman.csv")
    result.importance.to_csv(output_dir / "ranked_parameters.csv", index=False)
    result.coefficients.to_csv(output_dir / "ridge_coefficients.csv", index=False)
    result.model_scores.to_csv(output_dir / "model_scores.csv", index=False)
    result.collinear_pairs.to_csv(output_dir / "collinear_pairs.csv", index=False)


def _write_plots(merged: pd.DataFrame, ranked: dict, plot_dir: Path) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    pearson = ranked["pearson"]
    if isinstance(pearson, pd.DataFrame) and not pearson.empty:
        path = plot_dir / "correlation_heatmap.png"
        _plot_heatmap(pearson, path)
        written.append(str(path))
    importance = ranked["importance"]
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        for target, group in importance.groupby("target"):
            path = plot_dir / f"feature_importance_{_slug(str(target))}.png"
            _plot_importance(group, str(target), path)
            written.append(str(path))
            for feature in group.nsmallest(2, "rank")["feature"]:
                trend = plot_dir / f"trend_{_slug(str(feature))}_vs_{_slug(str(target))}.png"
                if _plot_trend(merged, str(feature), str(target), trend):
                    written.append(str(trend))
    return written


def _plot_heatmap(pearson: pd.DataFrame, path: Path) -> None:
    values = pearson.astype(float)
    fig_w = max(6.5, 1.3 * values.shape[1] + 3)
    fig_h = max(4.5, 0.45 * values.shape[0] + 2)
    figure, axis = plt.subplots(figsize=(fig_w, fig_h))
    image = axis.imshow(values.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(values.shape[1]), [TARGET_LABELS.get(col, col) for col in values.columns], rotation=30, ha="right")
    axis.set_yticks(range(values.shape[0]), list(values.index))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values.iloc[row, col]
            if np.isfinite(value):
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#1a1a1a")
    axis.set_title("Pearson correlation: synthesis parameters vs XRD metrics")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_importance(group: pd.DataFrame, target: str, path: Path) -> None:
    plot_rows = group.nsmallest(10, "rank").iloc[::-1]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#1f4e79" if (np.isfinite(r) and r >= 0) else "#b03a2e" for r in plot_rows["pearson_r"]]
    values = plot_rows["permutation_importance_mean"].to_numpy(dtype=float)
    if np.all(~np.isfinite(values)):
        values = plot_rows["pearson_r"].abs().to_numpy(dtype=float)
        axis.set_xlabel("|Pearson r|")
    else:
        axis.set_xlabel("Permutation importance (drop in R²)")
        axis.barh(plot_rows["feature"], values, color=colors, xerr=plot_rows["permutation_importance_std"], capsize=3)
        axis.set_title(TARGET_LABELS.get(target, target) + "\nbar color follows the sign of Pearson r")
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        return
    axis.barh(plot_rows["feature"], values, color=colors)
    axis.set_title(TARGET_LABELS.get(target, target))
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_trend(merged: pd.DataFrame, feature: str, target: str, path: Path) -> bool:
    if feature not in merged.columns or target not in merged.columns:
        # Encoded dummies are not columns of the raw merged table.
        dummy = _dummy_series(merged, feature)
        if dummy is None:
            return False
        frame = pd.DataFrame({feature: dummy, target: pd.to_numeric(merged[target], errors="coerce")}).dropna()
    else:
        frame = pd.DataFrame(
            {
                feature: merged[feature],
                target: pd.to_numeric(merged[target], errors="coerce"),
            }
        ).dropna()
    if len(frame) < 3:
        return False
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    y = frame[target].to_numpy(dtype=float)
    numeric_feature = pd.api.types.is_numeric_dtype(frame[feature]) and frame[feature].nunique(dropna=True) > 2
    if numeric_feature:
        x = frame[feature].to_numpy(dtype=float)
        axis.scatter(x, y, color="#1f4e79", alpha=0.85, edgecolor="white", linewidth=0.4)
        if np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 50)
            axis.plot(x_line, slope * x_line + intercept, color="#c44900", lw=1.4)
        axis.set_xlabel(feature)
    else:
        labels = frame[feature].astype(str)
        groups = [y[labels == label] for label in sorted(labels.unique())]
        names = sorted(labels.unique())
        axis.boxplot(groups, showfliers=False)
        axis.set_xticks(range(1, len(names) + 1), names)
        for index, label in enumerate(sorted(labels.unique()), start=1):
            points = y[labels == label]
            axis.scatter(np.full(points.shape, index), points, color="#1f4e79", alpha=0.7, zorder=3)
        axis.set_xlabel(feature)
    axis.set_ylabel(TARGET_LABELS.get(target, target))
    axis.set_title(f"{feature} vs {TARGET_LABELS.get(target, target)}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


def _dummy_series(merged: pd.DataFrame, feature: str) -> pd.Series | None:
    """Rebuild a drop-first dummy such as ``atmosphere_argon`` for plotting."""

    for column in merged.columns:
        if not feature.startswith(f"{column}_"):
            continue
        if pd.api.types.is_numeric_dtype(merged[column]):
            continue
        level = feature[len(column) + 1 :]
        return (merged[column].astype(str) == level).astype(float)
    return None


def render_report(features: pd.DataFrame, result: CorrelationResult) -> str:
    """Write the markdown report saved to ``outputs/analysis_report.md``."""

    from xrd_analyzer import summarize_features

    lines: list[str] = ["# MXene XRD and synthesis-parameter report", ""]
    if result.dataset_label == "synthetic-demo":
        lines.append(
            "Dataset: **synthetic demonstration**. The patterns and synthesis table were "
            "simulated so the pipeline can be run and checked. Do not use these rankings "
            "as experimental etching guidance."
        )
    else:
        lines.append(
            "Dataset: experimental files supplied to this run. Rankings describe associations "
            "in this table. They do not by themselves prove that changing a setting will change the product."
        )
    lines.append("")
    lines.append("## XRD metrics")
    lines.append("")
    for line in summarize_features(features):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## How the metrics are defined")
    lines.append("")
    lines.append("- Interplanar spacing uses Bragg's law, d = λ / (2 sin θ), with θ = 2θ / 2.")
    lines.append("- The c axis is taken from the lowest-angle basal peak as c = 2 d(002), assuming that peak is (002).")
    lines.append("- Apparent crystallite size uses the Scherrer equation, D = K λ / (β cos θ), with K = 0.9 and β the fitted FWHM in radians. Instrumental broadening is not removed, so this size is smaller than the physical domain size whenever the instrument itself broadens the line.")
    lines.append("- Crystallinity index is 1 / FWHM(002). Sharper basal peaks score higher. It is a shape metric, not a phase fraction.")
    lines.append("- MAX impurity index = I(104) / (I(104) + I(002)). TiO2 impurity index uses the anatase (101) and rutile (110) heights in the same way. These are relative intensity indices. They are not weight fractions from a Rietveld refinement.")
    lines.append("")
    lines.append("## Parameter ranking")
    lines.append("")
    if result.importance is None or result.importance.empty:
        lines.append("No ranking was produced. Check that sample ids in the XRD files match the synthesis table.")
    else:
        scores = result.model_scores.set_index("target") if not result.model_scores.empty else pd.DataFrame()
        for target, group in result.importance.groupby("target"):
            label = TARGET_LABELS.get(str(target), str(target))
            lines.append(f"### {label}")
            lines.append("")
            if str(target) in scores.index:
                score = scores.loc[str(target)]
                if isinstance(score, pd.DataFrame):
                    score = score.iloc[0]
                lines.append(
                    f"Complete rows: {int(score['n_samples'])}. "
                    f"Ranking method: {score['ranking_method']}. "
                    f"Random-forest in-sample R²: {_fmt(score['in_sample_r2'])}. "
                    f"Cross-validated R²: {_fmt(score['cv_r2_mean'])} ± {_fmt(score['cv_r2_std'])}."
                )
                lines.append("")
            lines.append("| rank | parameter | permutation importance | Pearson r | ridge coef (per 1 SD) |")
            lines.append("| --- | --- | --- | --- | --- |")
            for _, row in group.head(8).iterrows():
                lines.append(
                    f"| {int(row['rank'])} | {row['feature']} | {_fmt(row['permutation_importance_mean'])} | "
                    f"{_fmt(row['pearson_r'])} | {_fmt(row['ridge_coefficient'])} |"
                )
            lines.append("")
            top = group.iloc[0]
            lines.append(_direction_sentence(str(target), str(top["feature"]), top.get("pearson_r", np.nan)))
            lines.append("")
    if result.collinear_pairs is not None and not result.collinear_pairs.empty:
        lines.append("## Collinear synthesis parameters")
        lines.append("")
        lines.append("These pairs have |Pearson r| ≥ 0.85, so the model can split importance between them.")
        lines.append("")
        for _, row in result.collinear_pairs.iterrows():
            lines.append(f"- {row['feature_a']} and {row['feature_b']}: r = {_fmt(row['pearson_r'])}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in result.notes:
        lines.append(f"- {note}")
    lines.append("- Absolute XRD intensities are not compared across samples. Packing, mass, and slit settings move the whole scale. Within-scan ratios, peak positions, and widths are the quantities used here.")
    lines.append("- Default impurity windows assume a Ti3AlC2-type MAX phase and TiO2. Other MXenes need different windows in `XRDAnalysisConfig`.")
    lines.append("")
    return "\n".join(lines)


def _direction_sentence(target: str, feature: str, pearson_r: float) -> str:
    label = TARGET_LABELS.get(target, target)
    if not np.isfinite(pearson_r) or abs(pearson_r) < 0.1:
        return (
            f"The strongest ranked input for {label} is {feature}. "
            "The linear correlation is weak, so the rank may reflect a nonlinear pattern or importance shared with other inputs."
        )
    direction = "higher" if pearson_r > 0 else "lower"
    return (
        f"In this dataset, higher {feature} tracks {direction} {label} (Pearson r = {_fmt(pearson_r)})."
    )


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _slug(name: str) -> str:
    return _SLUG.sub("_", name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rank synthesis parameters against a feature table.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    features = pd.read_csv(args.features)
    synthesis = pd.read_csv(args.synthesis)
    result = run_correlation_analysis(features, synthesis, args.output)
    print(f"Wrote {args.output / 'analysis_report.md'}")
    print(result.importance.groupby("target").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
