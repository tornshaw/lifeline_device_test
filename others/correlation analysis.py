"""核心实现：
- correlation_analysis : 线性回归拟合
- plot_correlation     : 散点 + 拟合直线
- plot_time_series_comparison : 双 y 轴时间序列对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression

# === 字体设置：中文宋体，英文 Times New Roman ===
CH_FONT = font_manager.FontProperties(family="SimSun")
EN_FONT = font_manager.FontProperties(family="Times New Roman")

# -----------------------------------------------------------------------------
# 数据分析
# -----------------------------------------------------------------------------

def correlation_analysis(data: pd.DataFrame, x_cols: list[str], y_col: str):
    """对 ``x_cols`` 求均值后与 ``y_col`` 做线性回归。

    Returns
    -------
    x_data : np.ndarray, shape (n, 1)
    y_data : np.ndarray, shape (n, 1)
    model  : sklearn.linear_model.LinearRegression
    """
    x_data = data.loc[:, x_cols].mean(axis=1).values.reshape(-1, 1)
    y_data = data[y_col].values.reshape(-1, 1)

    model = LinearRegression().fit(x_data, y_data)
    return x_data, y_data, model

# -----------------------------------------------------------------------------
# 可视化
# -----------------------------------------------------------------------------

def plot_correlation(x_data, y_data, model, x_name="温度", x_unit="℃", y_name="位移", y_unit="mm"):
    """散点 + 拟合直线"""
    x_label = f"{x_name}({x_unit})"
    y_label = f"{y_name}({y_unit})"
    title = f"{y_name}与{x_name}的相关性分析"

    k = float(model.coef_[0][0])
    b = float(model.intercept_[0])

    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(x_data, y_data, color="black", s=10, label="原始数据")

    x_range = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
    y_fit = model.predict(x_range)
    plt.plot(x_range, y_fit, color="red", linewidth=2.5, label=f"拟合直线：y = {k:.3f}x + {b:.3f}")

    plt.xlabel(x_label, fontproperties=CH_FONT, fontsize=12)
    plt.ylabel(y_label, fontproperties=CH_FONT, fontsize=12)
    plt.title(title, fontproperties=CH_FONT, fontsize=14)
    plt.legend(prop=CH_FONT, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_time_series_comparison(
    data: pd.DataFrame,
    time_col: str | None,
    disp_col: str,
    temp_cols: list[str],
    disp_name: str = "伸缩缝位移",
    temp_name: str = "温度",
    max_xticks: int = 10,
):
    """双 y 轴时间序列图"""
    if time_col:
        times = pd.to_datetime(data[time_col], format="ISO8601", errors="coerce", utc=True)
    else:
        times = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index, format="ISO8601", errors="coerce", utc=True)

    displacement = data[disp_col]
    temperature = data[temp_cols].mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=100)
    color1 = "tab:blue"
    ax1.set_xlabel("时间", fontproperties=CH_FONT, fontsize=14)
    ax1.set_ylabel(f"{disp_name}(mm)", color=color1, fontproperties=CH_FONT, fontsize=14)
    ax1.plot(times, displacement, color=color1, label=disp_name)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel(f"{temp_name}(℃)", color=color2, fontproperties=CH_FONT, fontsize=14)
    ax2.plot(times, temperature, color=color2, linestyle="--", label=temp_name)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.get_xaxis().set_visible(False)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=max_xticks)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    plt.title(f"{disp_name}与{temp_name}的时间对比图", fontproperties=CH_FONT, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    plt.show()