import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    print("⚠️ 请先安装 tabulate 库：pip install tabulate")
    tabulate = None

CH_FONT = font_manager.FontProperties(family="SimSun")
plt.rcParams["axes.unicode_minus"] = False

def fill_missing(series: pd.Series, method="time") -> pd.Series:
    series = series.interpolate(method=method).bfill().ffill()
    return series

def detect_outliers_robust_iterative(series, threshold=3, window=12, min_amplitude=None, max_iter=5):
    clean_series = series.copy()
    outliers_all = pd.Series(False, index=series.index)
    for _ in range(max_iter):
        median = clean_series.rolling(window, center=True, min_periods=1).median()
        mad = clean_series.rolling(window, center=True, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        ).replace(0, 1e-6)
        robust_z = 0.6745 * (clean_series - median) / mad
        outliers = np.abs(robust_z) > threshold
        if min_amplitude is not None:
            outliers &= np.abs(clean_series - median) > min_amplitude
        if outliers.sum() == 0:
            break
        outliers_all |= outliers
        clean_series[outliers] = np.nan
        clean_series = fill_missing(clean_series)
    return outliers_all

def preprocess_series(series, outlier_thresh=3, window_size=12, min_amplitude=None):
    series = series.sort_index()
    outliers = detect_outliers_robust_iterative(series, threshold=outlier_thresh,
                                                window=window_size, min_amplitude=min_amplitude)
    clean = series.copy()
    clean[outliers] = np.nan
    clean = fill_missing(clean)
    return clean, outliers, "鲁棒滑动中位数+MAD迭代检测", "时间插值 + bfill + ffill"

def plot_comparison(time, original, cleaned, outliers, title="异常识别与处理曲线", out_path="figures/cleaning_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)

    # 画图
    ax.plot(time, original, label="原始数据", color="black", linestyle='--', alpha=0.6, linewidth=1.0)
    ax.plot(time, cleaned, label="清洗后数据", color="tab:orange", linewidth=1.5)
    ax.scatter(time[outliers], original[outliers], color="red", label="离群点", zorder=5, s=30, edgecolor='black')

    # 设置字体与标题
    ax.set_xlabel("时间", fontproperties=CH_FONT, fontsize=14)
    ax.set_ylabel("数值", fontproperties=CH_FONT, fontsize=14)
    ax.set_title(title, fontproperties=CH_FONT, fontsize=14)
    ax.legend(prop=CH_FONT)

    # 设置主刻度为“天”，次刻度为“小时”
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    ax.xaxis.set_minor_locator(HourLocator(interval=1))  # 次刻度每1小时一个
    ax.xaxis.set_minor_formatter(DateFormatter("%H:%M"))

    # 时间标签美化
    fig.autofmt_xdate(rotation=30)

    # 网格和布局
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    fig.tight_layout()

    # 保存与展示
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"✅ 图像已保存至: {Path(out_path).resolve()}")


def print_outliers(time, series, outliers):
    df_outliers = pd.DataFrame({
        "时间": time[outliers],
        "异常值": series[outliers]
    })

    total_outliers = df_outliers.shape[0]

    if total_outliers == 0:
        print("ℹ️ 未检测到异常点。")
        return

    print(f"\n🛑 检测到的异常点数量：{total_outliers} 个")
    print("📋 异常点详细信息如下：")
    if tabulate is not None:
        print(tabulate(df_outliers, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(df_outliers)

