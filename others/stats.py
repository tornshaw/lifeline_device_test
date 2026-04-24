from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

try:
    from tabulate import tabulate
    TAB = True
except ImportError:
    TAB = False

plt.rcParams.update({"figure.dpi": 120, "axes.unicode_minus": False})
ZH_FONT = fm.FontProperties(family="SimSun", size=12)
EN_FONT = fm.FontProperties(family="Times New Roman", size=12)
TIME_COL = "time"


# ------------------------------------------------ 数据 IO ------------------------------------------------
def load_monitor_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="ISO8601", errors="coerce").dt.tz_localize(None)
    if df[TIME_COL].isna().any():
        raise ValueError("时间字段解析失败，请检查数据格式。")

    if {"sensor_id", "value"}.issubset(df.columns):
        return df.rename(columns={"value": "reading"})
    sensor_cols = [c for c in df.columns if c != TIME_COL]
    return df.melt(id_vars=[TIME_COL], value_vars=sensor_cols,
                   var_name="sensor_id", value_name="reading")


def filter_data(df: pd.DataFrame, sensors: List[str], time_range: Tuple[str, str]) -> pd.DataFrame:
    start, end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
    return df.loc[df["sensor_id"].isin(sensors) & (df[TIME_COL] >= start) & (df[TIME_COL] <= end)].copy()


# ------------------------------------------------ 统计 & GMM ------------------------------------------------
def basic_stats(series: pd.Series) -> Dict[str, float]:
    return {"mean": series.mean(), "var": series.var(ddof=0),
            "max": series.max(), "min": series.min()}


def fit_gmm(series: pd.Series, k: int = 2) -> GaussianMixture:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
    gmm.fit(series.dropna().values.reshape(-1, 1))
    return gmm


# ------------------------------------------------ 可视化 ------------------------------------------------
def plot_hist_gmm(series: pd.Series, gmm: GaussianMixture, sensor: str, fig_dir: Path | None = None, show: bool = True):
    """绘制直方图 + GMM PDF。
    Parameters
    ----------
    series : pd.Series            目标数据列
    gmm    : GaussianMixture      已拟合模型 (k=2)
    sensor : str                  传感器 ID/列名 (用于标题与文件名)
    fig_dir: Path | None          若指定，则把图片保存到此文件夹
    show   : bool                 是否调用 plt.show()
    """
    data = series.dropna().values
    xs = np.linspace(data.min(), data.max(), 400).reshape(-1, 1)
    pdf_total = np.exp(gmm.score_samples(xs))
    pdf_comps = [w * norm.pdf(xs, m[0], np.sqrt(c[0, 0]))
                 for w, m, c in zip(gmm.weights_, gmm.means_, gmm.covariances_)]

    plt.figure(figsize=(6, 3.5))
    plt.hist(data, bins=50, density=True, alpha=0.65, edgecolor="black", label="直方图")
    plt.plot(xs, pdf_total, "r-", lw=2, label="GMM(k=2)")
    plt.title(f"传感器：{sensor}", fontproperties=ZH_FONT, fontsize=14)
    plt.xlabel("值", fontproperties=ZH_FONT)
    plt.ylabel("概率密度", fontproperties=ZH_FONT)
    plt.legend(prop=ZH_FONT, fontsize=9, loc="upper right")
    plt.tight_layout()

    if fig_dir is not None:
        fig_dir.mkdir(exist_ok=True, parents=True)
        fname = fig_dir / f"{sensor}_hist_gmm.png"
        plt.savefig(fname, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------ 管道接口 ------------------------------------------------
def run_pipeline(
    csv_path: str | Path,
    sensors: List[str],
    time_range: Tuple[str, str],
    out_dir: str | Path = "reports",
    fig_dir: str | Path | None = None,
    plot: bool = True,
):
    df_all = load_monitor_data(csv_path)
    df_sel = filter_data(df_all, sensors, time_range)

    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    rows = []

    for sid in sensors:
        ser = df_sel.loc[df_sel["sensor_id"] == sid, "reading"]
        stats_dict = basic_stats(ser)
        gmm = fit_gmm(ser)

        if plot:
            plot_hist_gmm(ser, gmm, sid, fig_dir=Path(fig_dir) if fig_dir else None, show=True)

        rows.append([
            sid,
            f"{stats_dict['mean']:.3f}",
            f"{stats_dict['var']:.3f}",
            f"{stats_dict['max']:.3f}",
            f"{stats_dict['min']:.3f}",
            np.round(gmm.weights_, 4).tolist(),
            np.round(gmm.means_.flatten(), 4).tolist(),
            [float(np.round(c[0, 0], 4)) for c in gmm.covariances_],
        ])

    headers_cn = ["传感器", "均值", "方差", "最大值", "最小值", "GMM权重", "GMM均值向量", "GMM方差"]
    if TAB:
        print(tabulate(rows, headers=headers_cn, tablefmt="github", stralign="center", numalign="center"))
    else:
        print(pd.DataFrame(rows, columns=headers_cn).to_string(index=False))

    pd.DataFrame(rows, columns=headers_cn).to_csv(
        out_dir / f"stats_{time_range[0]}~{time_range[1]}.csv", index=False
    )
    print("统计结果结果已保存至", out_dir.resolve())
    if fig_dir:
        print("图像已保存至", Path(fig_dir).resolve())
    else:
        print("未指定图像保存目录。")
