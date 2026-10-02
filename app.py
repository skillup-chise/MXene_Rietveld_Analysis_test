#!/usr/bin/env python3
"""Streamlit UI for the MXene XRD pipeline.

Run locally with ``streamlit run app.py``. Streamlit Community Cloud uses this
file as the entry point. The command-line batch run remains ``python main.py``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from correlation_analyzer import TARGET_LABELS
from demo_data import DEMO_DIR, ensure_demo_dataset
from loader import canon_sample_id, load_dataset, load_synthesis_table
from main import run_pipeline
from xrd_analyzer import WAVELENGTH_CU_KA1, XRDAnalysisConfig

APP_WORK = Path(tempfile.gettempdir()) / "mxene_xrd_app"

PARAMETER_DEFAULTS: list[tuple[str, str, object]] = [
    ("stirrer_bar_size_mm", "攪拌子サイズ (mm)", 25.0),
    ("stirring_speed_rpm", "回転数 (rpm)", 400.0),
    ("flask_volume_ml", "フラスコ容量 (mL)", 100.0),
    ("etching_time_h", "エッチング時間 (h)", 24.0),
    ("etching_temperature_C", "エッチング温度 (°C)", 35.0),
    ("hcl_concentration_M", "HCl 濃度 (M)", 9.0),
    ("lif_to_max_molar_ratio", "LiF/MAX モル比", 1.0),
    ("washing_cycles", "洗浄回数", 5),
    ("atmosphere", "雰囲気", "air"),
]

METRIC_COLUMNS = [
    ("c_axis_A", "c軸 (Å)", "{:.2f}"),
    ("mxene_002_fwhm_deg", "(002) FWHM (°)", "{:.3f}"),
    ("crystallite_size_002_nm", "見掛け結晶子径 (nm)", "{:.2f}"),
    ("max_impurity_index", "MAX 指数", "{:.3f}"),
    ("tio2_impurity_index", "TiO2 指数", "{:.3f}"),
]

FLAG_LABELS = {
    "mxene_002": "MXene (002)",
    "mxene_002_with_residual_max": "MXene (002) と残留 MAX (002)",
    "basal_matches_max_position": "(002) が未反応 MAX の位置に近い",
    "ambiguous_basal_peak": "低角ピークの帰属があいまい",
    "not_detected": "(002) 未検出",
    "too_few_points": "点数不足",
    "analysis_failed": "解析失敗",
}

WAVELENGTHS = {
    "Cu Kα1 (1.5406 Å)": 1.5406,
    "Cu Kα 平均 (1.5418 Å)": 1.5418,
    "Co Kα (1.7889 Å)": 1.7889,
}


def default_synthesis_frame(sample_ids: list[str]) -> pd.DataFrame:
    """One row per sample, filled with ordinary LiF/HCl starting values."""

    frame = pd.DataFrame({"sample_id": [str(sample_id) for sample_id in sample_ids]})
    for name, _label, default in PARAMETER_DEFAULTS:
        frame[name] = default
    return frame


def align_synthesis(sample_ids: list[str], uploaded: pd.DataFrame | None) -> pd.DataFrame:
    """Start from the default grid and overlay any uploaded conditions by sample id."""

    base = default_synthesis_frame(sample_ids)
    if uploaded is None or uploaded.empty or "sample_id" not in uploaded.columns:
        return base
    table = uploaded.copy()
    table["_key"] = table["sample_id"].map(canon_sample_id)
    table = table.drop_duplicates("_key", keep="first")
    base["_key"] = base["sample_id"].map(canon_sample_id)
    indexed = table.set_index("_key")
    for column in indexed.columns:
        if column == "sample_id":
            continue
        if column not in base.columns:
            base[column] = pd.NA
        mapped = base["_key"].map(indexed[column])
        base[column] = mapped.where(mapped.notna(), base[column])
    return base.drop(columns=["_key"])


def write_xrd_files(payloads: list[tuple[str, bytes]], dest: Path) -> Path:
    """Write uploaded bytes into ``dest/xrd`` using only the file name."""

    xrd_dir = dest / "xrd"
    if xrd_dir.exists():
        shutil.rmtree(xrd_dir)
    xrd_dir.mkdir(parents=True)
    for name, data in payloads:
        safe = Path(str(name)).name
        if not safe or safe.startswith(".") or safe.startswith("~$"):
            continue
        (xrd_dir / safe).write_bytes(data)
    return xrd_dir


def stage_demo_subset(dest: Path, n: int) -> Path:
    """Copy the first ``n`` synthetic patterns into a working folder."""

    ensure_demo_dataset(DEMO_DIR)
    files = sorted((DEMO_DIR / "xrd").glob("*.csv"))[: int(n)]
    if not files:
        raise FileNotFoundError(f"No demo patterns in {DEMO_DIR / 'xrd'}")
    xrd_dir = dest / "xrd"
    if xrd_dir.exists():
        shutil.rmtree(xrd_dir)
    xrd_dir.mkdir(parents=True)
    for path in files:
        shutil.copy(path, xrd_dir / path.name)
    marker = DEMO_DIR / "SYNTHETIC.txt"
    if marker.exists():
        shutil.copy(marker, dest / "SYNTHETIC.txt")
    return xrd_dir


def demo_synthesis_table() -> pd.DataFrame:
    table, _id_column, _description = load_synthesis_table(DEMO_DIR / "synthesis_params.csv")
    return table


def sample_ids_in(xrd_dir: Path) -> list[str]:
    bundle = load_dataset(xrd_dir, None)
    return [spectrum.sample_id for spectrum in bundle.spectra]


def execute_analysis(
    xrd_dir: Path,
    synthesis: pd.DataFrame | None,
    output_dir: Path,
    *,
    wavelength: float = WAVELENGTH_CU_KA1,
    baseline: str = "opening",
    baseline_window: float = 8.0,
    seed: int = 42,
) -> dict:
    """Run the batch pipeline on a folder of patterns and an optional condition table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    synthesis_path = output_dir / "synthesis_input.csv"
    if synthesis is not None and not synthesis.empty:
        synthesis.to_csv(synthesis_path, index=False)
    else:
        synthesis_path = output_dir / "synthesis_missing.csv"
    label = "synthetic-demo" if (xrd_dir.parent / "SYNTHETIC.txt").exists() else "experimental"
    analysis = XRDAnalysisConfig(
        wavelength_A=float(wavelength),
        baseline_method=baseline,
        baseline_window_deg=float(baseline_window),
    )
    from correlation_analyzer import CorrelationConfig

    return run_pipeline(
        xrd_dir,
        synthesis_path,
        output_dir,
        label,
        analysis,
        CorrelationConfig(random_state=int(seed)),
    )


def _median_text(features: pd.DataFrame, column: str, template: str) -> str:
    if column not in features.columns:
        return "—"
    values = pd.to_numeric(features[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return "—"
    return template.format(float(finite.median()))


def _display_features(features: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "sample_id": "サンプル",
        "assignment_flag": "帰属",
        "c_axis_A": "c軸 (Å)",
        "d_002_A": "d(002) (Å)",
        "mxene_002_center_deg": "(002) 2θ (°)",
        "mxene_002_fwhm_deg": "FWHM (°)",
        "crystallite_size_002_nm": "結晶子径 (nm)",
        "crystallinity_index": "結晶性指数",
        "max_impurity_index": "MAX 指数",
        "max_to_002_intensity_ratio": "I(104)/I(002)",
        "tio2_impurity_index": "TiO2 指数",
        "harmonic_004_delta_deg": "(004) ずれ (°)",
    }
    present = [column for column in columns if column in features.columns]
    view = features[present].rename(columns=columns)
    if "帰属" in view.columns:
        view["帰属"] = view["帰属"].map(lambda value: FLAG_LABELS.get(str(value), str(value)))
    return view


def _cjk_font():
    """Return a font that can draw Japanese labels, when the machine has one."""

    from matplotlib import font_manager

    for path in (
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).exists():
            return font_manager.FontProperties(fname=path)
    return None


def spectrum_figure(result):
    """Two-panel figure: measured pattern with baseline, then corrected intensity."""

    import matplotlib.pyplot as plt

    if result.two_theta is None:
        return None
    font = _cjk_font()
    text = {"fontproperties": font} if font is not None else {}
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(result.two_theta, result.intensity_raw, color="#4c4c4c", lw=1.0, label="測定")
    axes[0].plot(result.two_theta, result.baseline, color="#c44900", lw=1.1, label="バックグラウンド")
    axes[0].set_ylabel("強度", **text)
    if font is not None:
        axes[0].legend(frameon=False, prop=font)
    else:
        axes[0].legend(frameon=False)
    flag = FLAG_LABELS.get(str(result.features.get("assignment_flag", "")), "")
    axes[0].set_title(f"{result.sample_id}  {flag}", **text)
    axes[1].plot(result.two_theta, result.intensity_corrected, color="#1f4e79", lw=1.0)
    for peak in result.peaks:
        if peak.detected and np.isfinite(peak.center_deg):
            axes[1].axvline(peak.center_deg, color="#b03a2e", lw=0.7, alpha=0.8)
    axes[1].set_xlabel("2θ (度)", **text)
    axes[1].set_ylabel("補正後の強度", **text)
    c_axis = result.features.get("c_axis_A", np.nan)
    fwhm = result.features.get("mxene_002_fwhm_deg", np.nan)
    c_text = f"{c_axis:.2f}" if np.isfinite(c_axis) else "—"
    f_text = f"{fwhm:.3f}" if np.isfinite(fwhm) else "—"
    axes[1].text(
        0.99,
        0.95,
        f"c = {c_text} Å\nFWHM(002) = {f_text}°",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        **text,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    return figure


def _column_config(frame: pd.DataFrame) -> dict:
    config = {
        "sample_id": st.column_config.TextColumn("サンプルID", disabled=True, width="small"),
    }
    labels = {name: label for name, label, _default in PARAMETER_DEFAULTS}
    for name, label in labels.items():
        if name not in frame.columns or name == "atmosphere":
            continue
        decimals = {
            "hcl_concentration_M": "%.1f",
            "lif_to_max_molar_ratio": "%.2f",
            "etching_time_h": "%.1f",
        }.get(name, "%.0f")
        config[name] = st.column_config.NumberColumn(label, min_value=0.0, format=decimals)
    if "atmosphere" in frame.columns:
        observed = {str(value) for value in frame["atmosphere"].dropna()}
        if observed.issubset({"air", "argon", ""}):
            config["atmosphere"] = st.column_config.SelectboxColumn("雰囲気", options=["air", "argon"])
        else:
            config["atmosphere"] = st.column_config.TextColumn("雰囲気")
    return config


def _render_overview(analysis: dict) -> None:
    features = analysis["features"]
    if (analysis["output_dir"].parent / "SYNTHETIC.txt").exists() or (
        analysis["correlation"] is not None and analysis["correlation"].dataset_label == "synthetic-demo"
    ):
        st.warning("この結果は同梱の合成データです。実験のエッチング条件の根拠には使えません。")
    columns = st.columns(len(METRIC_COLUMNS))
    for column, (key, label, template) in zip(columns, METRIC_COLUMNS):
        column.metric(label, _median_text(features, key, template))
    st.caption("上の数値はサンプル全体の中央値です。結晶子径は装置由来の広がりを差し引いていません。")
    st.dataframe(_display_features(features), width="stretch", hide_index=True)
    overlay = Path(analysis["output_dir"]) / "plots" / "basal_002_overlay.png"
    if overlay.exists():
        st.image(str(overlay), caption="低角領域。色は c 軸長（Å）です。", width="stretch")


def _render_spectrum(analysis: dict) -> None:
    results = [result for result in analysis.get("results") or [] if result.two_theta is not None]
    if not results:
        st.info("表示できるパターンがありません。")
        return
    labels = [result.sample_id for result in results]
    selected = st.selectbox("サンプル", labels)
    result = next(item for item in results if item.sample_id == selected)
    figure = spectrum_figure(result)
    if figure is not None:
        st.pyplot(figure)
        import matplotlib.pyplot as plt

        plt.close(figure)
    peaks = analysis["peaks"]
    if not peaks.empty:
        sample_peaks = peaks[peaks["sample_id"] == selected]
        st.dataframe(sample_peaks, width="stretch", hide_index=True)


def _render_impurity(analysis: dict) -> None:
    features = analysis["features"]
    st.markdown(
        "MAX 指数は `I(104) / (I(104) + I(002))`、TiO2 指数はアナターゼ (101) とルチル (110) の"
        "高さから同じ形で計算しています。同じスキャンの中の強度比であり、重量分率ではありません。"
    )
    chart_columns = [column for column in ("max_impurity_index", "tio2_impurity_index") if column in features.columns]
    if chart_columns:
        chart = features.set_index("sample_id")[chart_columns].apply(pd.to_numeric, errors="coerce")
        chart = chart.rename(columns={"max_impurity_index": "MAX 指数", "tio2_impurity_index": "TiO2 指数"})
        st.bar_chart(chart, width="stretch")
    st.dataframe(_display_features(features), width="stretch", hide_index=True)


def _render_ranking(analysis: dict) -> None:
    correlation = analysis["correlation"]
    if correlation is None or correlation.importance is None or correlation.importance.empty:
        st.info("条件の順位を出すには、合成条件の表と、条件が異なる複数の XRD が必要です。目安は 8 サンプル以上です。")
        return
    st.caption("順位はランダムフォレストの並べ替え重要度です。8 件未満のときはピアソン相関の絶対値に切り替わります。符号はピアソンの r です。")
    scores = correlation.model_scores
    if scores is not None and not scores.empty:
        score_view = scores.copy()
        score_view["target"] = score_view["target"].map(lambda name: TARGET_LABELS.get(str(name), str(name)))
        st.dataframe(score_view, width="stretch", hide_index=True)
    for target, group in correlation.importance.groupby("target"):
        st.subheader(TARGET_LABELS.get(str(target), str(target)))
        show = group.head(8)[
            ["rank", "feature", "permutation_importance_mean", "pearson_r", "ridge_coefficient"]
        ].rename(
            columns={
                "rank": "順位",
                "feature": "条件",
                "permutation_importance_mean": "並べ替え重要度",
                "pearson_r": "ピアソン r",
                "ridge_coefficient": "リッジ係数 / 1SD",
            }
        )
        st.dataframe(show, width="stretch", hide_index=True)
    plot_dir = Path(analysis["output_dir"]) / "plots"
    heatmap = plot_dir / "correlation_heatmap.png"
    if heatmap.exists():
        st.image(str(heatmap), caption="合成条件と XRD 指標のピアソン相関", width="stretch")
    others = sorted(path for path in plot_dir.glob("*.png") if path.name != "correlation_heatmap.png" and not path.name.startswith("spectrum_"))
    if others:
        choice = st.selectbox("図", others, format_func=lambda path: path.stem)
        st.image(str(choice), width="stretch")
    if correlation.notes:
        with st.expander("モデルの注記"):
            for note in correlation.notes:
                st.write(note)


def _render_downloads(analysis: dict) -> None:
    output_dir = Path(analysis["output_dir"])
    features = analysis["features"]
    st.download_button(
        "XRD 指標 CSV",
        features.to_csv(index=False).encode("utf-8"),
        file_name="xrd_features.csv",
        mime="text/csv",
    )
    ranked = output_dir / "ranked_parameters.csv"
    if ranked.exists():
        st.download_button(
            "条件の順位 CSV",
            ranked.read_bytes(),
            file_name="ranked_parameters.csv",
            mime="text/csv",
        )
    report = output_dir / "analysis_report.md"
    if report.exists():
        st.download_button(
            "レポート Markdown",
            report.read_bytes(),
            file_name="analysis_report.md",
            mime="text/markdown",
        )
        with st.expander("詳細レポート"):
            st.markdown(report.read_text(encoding="utf-8"))


def _uploaded_synthesis_frame(upload) -> pd.DataFrame | None:
    if upload is None:
        return None
    suffix = Path(upload.name).suffix.lower() or ".csv"
    temp_path = APP_WORK / f"uploaded_synthesis{suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(upload.getvalue())
    table, _id_column, _description = load_synthesis_table(temp_path)
    return table


def _prepare_inputs(use_demo: bool, demo_n: int, xrd_uploads, synthesis_upload) -> tuple[Path, pd.DataFrame] | None:
    APP_WORK.mkdir(parents=True, exist_ok=True)
    if use_demo:
        stage_key = f"demo-{int(demo_n)}"
        dest = APP_WORK / stage_key
        if st.session_state.get("stage_key") != stage_key:
            xrd_dir = stage_demo_subset(dest, int(demo_n))
            st.session_state.stage_key = stage_key
            st.session_state.xrd_dir = str(xrd_dir)
            st.session_state.pop("analysis", None)
        xrd_dir = Path(st.session_state.xrd_dir)
        uploaded = demo_synthesis_table()
    else:
        if not xrd_uploads:
            return None
        fingerprint = tuple((file.name, len(file.getvalue())) for file in xrd_uploads)
        synthesis_fingerprint = None
        if synthesis_upload is not None:
            synthesis_fingerprint = (synthesis_upload.name, len(synthesis_upload.getvalue()))
        stage_key = "upload-" + str(hash((fingerprint, synthesis_fingerprint)))
        dest = APP_WORK / "upload"
        if st.session_state.get("stage_key") != stage_key:
            xrd_dir = write_xrd_files([(file.name, file.getvalue()) for file in xrd_uploads], dest)
            st.session_state.stage_key = stage_key
            st.session_state.xrd_dir = str(xrd_dir)
            st.session_state.pop("analysis", None)
        xrd_dir = Path(st.session_state.xrd_dir)
        uploaded = _uploaded_synthesis_frame(synthesis_upload)
    ids = sample_ids_in(xrd_dir)
    return xrd_dir, align_synthesis(ids, uploaded)


def main() -> None:
    st.set_page_config(page_title="MXene XRD", layout="wide")
    st.title("MXene XRD 解析")
    st.caption(
        "(002) から層間の c 軸長とピーク幅を求め、MAX 相と TiO2 に相当するピークの強度比を出します。"
        "サンプルが複数あるときは、攪拌やエッチング条件のどれが結果と強く連動しているかを並べます。"
    )

    with st.sidebar:
        st.header("解析設定")
        wavelength_label = st.selectbox("波長", list(WAVELENGTHS))
        baseline_label = st.selectbox(
            "バックグラウンド",
            ["モルフォロジー開口", "非対称最小二乗 (ALS)"],
        )
        baseline = "opening" if baseline_label.startswith("モルフォロジー") else "als"
        baseline_window = st.slider("ベースライン窓 (°)", min_value=4.0, max_value=15.0, value=8.0, step=0.5)
        st.header("データ")
        use_demo = st.checkbox(
            "同梱の合成デモを使う",
            value=False,
            help="実験データではありません。動作確認用のシミュレーションです。",
        )
        demo_n = 8
        xrd_uploads = []
        synthesis_upload = None
        if use_demo:
            demo_n = st.selectbox("デモのサンプル数", [8, 16, 36], index=0)
        else:
            xrd_uploads = st.file_uploader(
                "XRD ファイル",
                type=["csv", "txt", "dat", "xy", "xlsx", "xls"],
                accept_multiple_files=True,
                help="CSV、Excel、空白区切りテキスト。2θ と強度の列を自動で判別します。",
            )
            synthesis_upload = st.file_uploader(
                "合成条件表（任意）",
                type=["csv", "xlsx", "xls"],
                help="sample_id で XRD と結びます。無ければ下の表に手入力できます。",
            )
        run = st.button("解析を実行", type="primary", width="stretch")

    try:
        prepared = _prepare_inputs(use_demo, int(demo_n), xrd_uploads, synthesis_upload)
    except Exception as exc:
        st.error(f"ファイルを読めませんでした: {exc}")
        return

    if prepared is None:
        st.info("サイドバーから XRD ファイルを選ぶか、「同梱の合成デモを使う」にチェックを入れてください。")
        st.markdown(
            "対応レイアウトは、1 ファイル 1 サンプル、2θ を共有する横持ち表、"
            "`sample_id` 付きの縦持ち表、シートごとの Excel です。"
        )
        template = default_synthesis_frame(["MX-001"])
        st.download_button(
            "合成条件テンプレート CSV",
            template.to_csv(index=False).encode("utf-8"),
            file_name="synthesis_params_template.csv",
            mime="text/csv",
        )
        return

    xrd_dir, synthesis_frame = prepared
    st.subheader("合成条件")
    st.caption(
        "攪拌子サイズ、回転数、フラスコ容量、エッチング時間・温度などをサンプルごとに編集できます。"
        "この表が、条件の影響ランキングに使われます。"
    )
    edited = st.data_editor(
        synthesis_frame,
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        column_config=_column_config(synthesis_frame),
        key=f"synthesis-{st.session_state.get('stage_key', 'none')}",
    )
    st.caption(f"読み込んだスペクトル: {len(synthesis_frame)} 件")

    if run:
        with st.spinner("バックグラウンドを引いてピークをフィットしています…"):
            try:
                output_dir = xrd_dir.parent / "outputs"
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                analysis = execute_analysis(
                    xrd_dir,
                    edited,
                    output_dir,
                    wavelength=WAVELENGTHS[wavelength_label],
                    baseline=baseline,
                    baseline_window=float(baseline_window),
                )
            except Exception as exc:
                st.error(f"解析に失敗しました: {exc}")
                return
        st.session_state.analysis = analysis

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("条件を確認したら、サイドバーの「解析を実行」を押してください。")
        return

    overview, spectra_tab, impurity_tab, ranking_tab, download_tab = st.tabs(
        ["概要", "スペクトル", "不純物と結晶性", "条件の影響", "ダウンロード"]
    )
    with overview:
        _render_overview(analysis)
    with spectra_tab:
        _render_spectrum(analysis)
    with impurity_tab:
        _render_impurity(analysis)
    with ranking_tab:
        _render_ranking(analysis)
    with download_tab:
        _render_downloads(analysis)


def _launch_if_streamlit() -> None:
    """Start the UI under ``streamlit run`` without running it on a plain import."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return
    if get_script_run_ctx() is not None:
        main()


_launch_if_streamlit()
