"""Core functions for SARIMA trend forecasting."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import logging

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from tabulate import tabulate   
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------- LOG & FONT ---------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
plt.rcParams["axes.unicode_minus"] = False
try:
    ZH_FONT = fm.FontProperties(family="SimSun")
    _ = fm.findfont(ZH_FONT)
except Exception:
    logging.warning("SimSun 字体未找到，可能导致中文乱码。")
    ZH_FONT = fm.FontProperties()

# ------------------------------------------------------------------
# 1. load
# ------------------------------------------------------------------
def load_series(
    csv_path: str | Path,
    time_col: str,
    value_col: str,
    agg: str = "median",
    interp: str = "time",
) -> pd.Series:
    """Load CSV → 1 h series (median aggregation) → drop NaN."""
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df[time_col])
    s = pd.Series(df[value_col].values, index=ts).sort_index()

    med_delta = s.index.to_series().diff().median()
    if pd.notna(med_delta) and med_delta.total_seconds() < 3600:
        s = getattr(s.resample("1h"), agg)()
    s = s.asfreq("1h").interpolate(interp).dropna()
    s.index = pd.DatetimeIndex(s.index, freq="h")
    return s

# ------------------------------------------------------------------
# 2. train
# ------------------------------------------------------------------
def train_sarima(
    series: pd.Series,
    seasonal_period: int = 24,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
):
    if seasonal_order is None:
        seasonal_order = (1, 1, 1, seasonal_period)
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

# ------------------------------------------------------------------
# 3. plot
# ------------------------------------------------------------------
def plot_forecast(
    series: pd.Series,
    res,
    steps_ahead: int = 24,
    title: str = "趋势预测",
    out_path: str | Path = "figures/24h预测值.png",
    show: bool = True,                    # ← 新增开关
):
    fitted = res.get_prediction().predicted_mean
    fc_df = res.get_forecast(steps=steps_ahead).summary_frame(alpha=0.05)
    if not isinstance(fc_df.index, pd.DatetimeIndex):
        start = series.index[-1] + pd.Timedelta(hours=1)
        fc_df.index = pd.date_range(start, periods=steps_ahead, freq="h")

    plt.figure(figsize=(10, 4))
    series.plot(label="原始数据", color="black")
    fitted.plot(label="拟合值", color="tab:red", ls="--")
    fc_df["mean"].plot(label=f"未来 {steps_ahead}h 预测", color="tab:blue")
    plt.fill_between(fc_df.index, fc_df["mean_ci_lower"], fc_df["mean_ci_upper"],
                     color="blue", alpha=0.15, label="95%置信区间")
    plt.xlabel("时间", fontproperties=ZH_FONT)
    plt.ylabel("数值", fontproperties=ZH_FONT)
    plt.title(title, fontproperties=ZH_FONT)
    plt.legend(prop=ZH_FONT)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=150)

    if show:              # 同时弹窗展示
        plt.show()
    else:
        plt.close()

    logging.info("图已保存至 %s", out_path.resolve())
    return fc_df


# ------------------------------------------------------------------
# 4. pipeline
# ------------------------------------------------------------------
def run_pipeline(
    csv_path: str | Path,
    time_col: str = "time",
    value_col: str = "value",
    steps: int = 24,
    seasonal_period: int = 24,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] | None = None,
    out_fig_dir: str | Path = "figures",
    out_csv_dir: str | Path = "output",
):
    series = load_series(csv_path, time_col, value_col)
    res = train_sarima(series, seasonal_period, order, seasonal_order)
    fig_path = Path(out_fig_dir) / "趋势分析与预测曲线.png"
    fc_df = plot_forecast(series, res, steps, "趋势分析与预测", fig_path, show=True)

    # ===== 中文列名映射（一次性完成：保存 & 打印共用） =====
    fc_cn = fc_df.rename(columns={
        "mean": "预测均值",
        "mean_ci_lower": "下限 (95%)",
        "mean_ci_upper": "上限 (95%)",
    })[["预测均值", "下限 (95%)", "上限 (95%)"]]

    # -------- 保存 CSV --------
    out_csv_dir = Path(out_csv_dir); out_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_csv_dir / "24h预测值.csv"
    fc_cn.to_csv(csv_path, index=False)
    logging.info("预测结果已保存至 %s", csv_path.resolve())

    # -------- 终端打印 --------
    print("\n", tabulate(fc_cn.reset_index(drop=True),
                         headers="keys", showindex=False,
                         tablefmt="github", floatfmt=".3f"))
    return fc_cn
