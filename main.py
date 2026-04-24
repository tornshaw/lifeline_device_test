# main.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, QTabWidget, 
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
                             QMessageBox, QProgressBar, QSplitter, QButtonGroup, QRadioButton,
                             QFrame, QScrollArea, QCheckBox, QGridLayout, QSizePolicy,
                             QHeaderView, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 功率谱分析模块
class PowerSpectrumAnalysis:
    @staticmethod
    def bandpass_filter(signal, low_freq, high_freq, fs, order=4):
        """带通滤波"""
        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * fs
        low = low_freq / nyquist
        high = high_freq / nyquist
        if low >= 1.0:
            low = 0.99
        if high >= 1.0:
            high = 0.99
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    @staticmethod
    def analyze_power_spectrum(data, time_col, signal_cols, nperseg=256, 
                              use_filter=False, low_freq=0.1, high_freq=20.0):
        """
        分析功率谱
        """
        from scipy.signal import welch
        
        # 提取时间和信号
        time = pd.to_numeric(data[time_col], errors='coerce')
        time = time.dropna().values
        
        # 检查时间单位，假设 CSV 是秒，如果是毫秒可除以1000
        if np.mean(np.diff(time)) > 100:  # 如果 dt 很大，可能单位是毫秒
            time = time / 1000.0

        # 自动计算采样频率
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt
        print(f"Estimated sampling frequency: {fs:.2f} Hz")

        # 提取信号
        signals = data[signal_cols].astype(float).values
        
        # 应用带通滤波
        if use_filter:
            for i in range(signals.shape[1]):
                signals[:, i] = PowerSpectrumAnalysis.bandpass_filter(
                    signals[:, i], low_freq, high_freq, fs
                )
        
        # 计算功率谱密度 (Welch)
        nperseg = min(nperseg, signals.shape[0])  # 避免长度不足报错
        freqs, psd_all = [], []

        for i in range(signals.shape[1]):
            sig = signals[:, i] - np.mean(signals[:, i])  # 去直流
            f, Pxx = welch(
                sig,
                fs=fs,
                nperseg=nperseg,
                window='hann',
                scaling='density',
                detrend=False
            )
            freqs.append(f)
            psd_all.append(Pxx)

        return freqs, psd_all, fs

# 异常数据识别与处理模块 - 使用安全的索引对齐方法
class OutlierDetection:
    @staticmethod
    def fill_missing(series: pd.Series, method="time") -> pd.Series:
        # 确保索引是有序的
        series = series.sort_index()
        # 数据补全：插值 + 向后填充 + 向前填充
        series = series.interpolate(method=method)
        series = series.bfill().ffill()
        return series

    @staticmethod
    def detect_outliers_robust_iterative(
        series: pd.Series,
        threshold=3,
        window=12,
        min_amplitude=None,
        max_iter=5
    ) -> pd.Series:
        # 确保输入series是有序的
        series = series.sort_index()
        clean_series = series.copy()
        outliers_all = pd.Series(False, index=series.index)

        for i in range(max_iter):
            # 确保滚动窗口操作的索引对齐
            rolling_median = clean_series.rolling(window, center=True, min_periods=1).median()
            mad = clean_series.rolling(window, center=True, min_periods=1).apply(
                lambda x: np.median(np.abs(x - np.median(x))), raw=True
            )
            mad = mad.replace(0, 1e-6)

            robust_z = 0.6745 * (clean_series - rolling_median) / mad
            outliers = np.abs(robust_z) > threshold
            if min_amplitude is not None:
                outliers &= (np.abs(clean_series - rolling_median) > min_amplitude)

            # 确保outliers与clean_series的索引完全对齐
            outliers = outliers.reindex(clean_series.index, fill_value=False)

            if outliers.sum() == 0:
                break

            # 使用reindex确保outliers_all的索引对齐
            outliers_all = outliers_all.reindex(clean_series.index, fill_value=False)
            outliers_all |= outliers
            
            # 使用reindex确保clean_series的索引对齐，然后安全地设置异常值
            clean_series = clean_series.reindex(series.index)
            clean_series = clean_series.astype(float)  # 确保是浮点类型以支持NaN
            # 安全地设置异常值 - 使用reindex确保索引对齐
            mask = outliers.reindex(clean_series.index, fill_value=False)
            clean_series.loc[mask] = np.nan
            clean_series = OutlierDetection.fill_missing(clean_series)

        # 最终确保返回的outliers_all与原始series索引对齐
        return outliers_all.reindex(series.index, fill_value=False)

    @staticmethod
    def preprocess_series(
        series: pd.Series,
        outlier_thresh=3,
        window_size=12,
        min_amplitude=None,
    ):
        # 确保输入series是有序的
        series = series.sort_index()
        outliers = OutlierDetection.detect_outliers_robust_iterative(
            series,
            threshold=outlier_thresh,
            window=window_size,
            min_amplitude=min_amplitude,
            max_iter=5
        )
        clean = series.copy()
        # 确保outliers与clean的索引对齐
        outliers_aligned = outliers.reindex(clean.index, fill_value=False)
        clean.loc[outliers_aligned] = np.nan
        clean = OutlierDetection.fill_missing(clean)
        return clean, outliers

# 相关性分析模块
class CorrelationAnalysis:
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def analyze(self, data, x_cols, y_col):
        x_data = data.loc[:, x_cols].mean(axis=1).values.reshape(-1, 1)
        y_data = data[y_col].values.reshape(-1, 1)
        self.model.fit(x_data, y_data)
        return x_data, y_data, self.model

    def analyze_multiple(self, data, x_cols, y_col):
        """多变量相关性分析"""
        results = []
        for x_col in x_cols:
            x_data = data[x_col].values.reshape(-1, 1)
            y_data = data[y_col].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_data, y_data)
            correlation = np.corrcoef(x_data.flatten(), y_data.flatten())[0, 1]
            results.append({
                'x_col': x_col,
                'y_col': y_col,
                'correlation': correlation,
                'slope': model.coef_[0][0],
                'intercept': model.intercept_[0],
                'r_squared': correlation**2
            })
        return results

# 统计分析模块
class StatisticalAnalysis:
    def __init__(self):
        from sklearn.mixture import GaussianMixture
        self.gmm = GaussianMixture(n_components=2, random_state=0)

    def basic_stats(self, series):
        return {
            "mean": series.mean(),
            "var": series.var(ddof=0),
            "std": series.std(),
            "max": series.max(),
            "min": series.min(),
            "count": series.count(),
            "median": series.median(),
            "q25": series.quantile(0.25),
            "q75": series.quantile(0.75),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis()
        }

    def fit_gmm(self, series):
        self.gmm.fit(series.dropna().values.reshape(-1, 1))
        return self.gmm

    def advanced_stats(self, series):
        """高级统计分析"""
        from scipy import stats
        clean_series = series.dropna()
        
        # 计算各种统计量
        stats_dict = self.basic_stats(series)
        
        # 添加额外的统计量
        stats_dict.update({
            "range": series.max() - series.min(),
            "iqr": series.quantile(0.75) - series.quantile(0.25),
            "cv": stats_dict["std"] / stats_dict["mean"] if stats_dict["mean"] != 0 else 0,
            "mad": (clean_series - clean_series.mean()).abs().mean(),  # 平均绝对偏差
            "z_score": (clean_series - clean_series.mean()) / clean_series.std() if clean_series.std() != 0 else 0
        })
        
        return stats_dict

# 趋势分析模块
class TrendAnalysis:
    def __init__(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.model_class = SARIMAX

    def load_series(self, data, time_col, value_col):
        # 尝试多种时间格式解析
        time_series = pd.to_datetime(data[time_col], infer_datetime_format=True, errors='coerce')
        if time_series.isna().all():
            # 如果默认解析失败，尝试常见格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S', 
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%m/%d/%Y %H:%M:%S',
                '%d/%m/%Y %H:%M:%S'
            ]
            for fmt in formats:
                try:
                    time_series = pd.to_datetime(data[time_col], format=fmt, errors='coerce')
                    if not time_series.isna().all():
                        break
                except:
                    continue
        
        if time_series.isna().all():
            raise ValueError(f"无法解析时间列 '{time_col}' 的格式")
        
        s = pd.Series(data[value_col].values, index=time_series).sort_index()
        return s

    def train_sarima(self, series, order=(1,1,1), seasonal_order=(1,1,1,24)):
        model = self.model_class(
            series, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        return model.fit(disp=False)

    def analyze_trend_components(self, series):
        """分析趋势的各个组成部分"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # 确保数据频率
        if len(series) < 24:
            # 如果数据点太少，使用移动平均
            trend = series.rolling(window=min(5, len(series)), center=True).mean()
            seasonal = pd.Series([0] * len(series), index=series.index)
            residual = series - trend
        else:
            # 使用季节性分解
            decomposition = seasonal_decompose(series, model='additive', period=24)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
        
        return trend, seasonal, residual

# 数据加载器类
class DataLoader:
    @staticmethod
    def load_csv_with_time_handling(file_path):
        """
        加载CSV文件，自动处理时间列
        """
        try:
            # 尝试读取第一行判断是否为列名
            first_row = pd.read_csv(file_path, nrows=1, header=None).iloc[0]
            
            # 检查第一行是否包含非数值内容
            first_row_numeric = pd.to_numeric(first_row, errors='coerce')
            has_non_numeric = first_row_numeric.isna().any()
            
            if has_non_numeric:
                # 第一行包含非数值内容，将其作为列名
                df = pd.read_csv(file_path)
            else:
                # 第一行为数值，设置列名为"通道{列序号+1}"
                df = pd.read_csv(file_path, header=None)
                col_names = [f'通道{i+1}' for i in range(len(df.columns))]
                df.columns = col_names
            
            if df.empty:
                raise ValueError("CSV文件为空")
            
            # 检查是否有足够列
            if len(df.columns) < 2:
                raise ValueError("CSV文件至少需要时间列和一个数据列")
            
            # 尝试识别时间列
            time_col_candidates = []
            for col in df.columns:
                # 检查列名是否包含时间相关关键词
                if any(keyword in col.lower() for keyword in ['time', 'date', 'datetime', 'timestamp', '通道1']):
                    time_col_candidates.append(col)
            
            # 如果没有找到时间关键词列，尝试解析第一列
            if not time_col_candidates:
                time_col = df.columns[0]
            else:
                time_col = time_col_candidates[0]
            
            # 尝试解析时间列
            df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True, errors='coerce')
            
            # 如果解析失败，尝试其他常见格式
            if df[time_col].isna().all():
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S', 
                    '%Y-%m-%d %H:%M',
                    '%Y/%m/%d %H:%M',
                    '%Y-%m-%d',
                    '%Y/%m/%d',
                    '%m/%d/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S'
                ]
                for fmt in formats:
                    try:
                        df[time_col] = pd.to_datetime(df[time_col], format=fmt, errors='coerce')
                        if not df[time_col].isna().all():
                            break
                    except:
                        continue
            
            if df[time_col].isna().all():
                raise ValueError(f"无法解析列 '{time_col}' 为时间格式")
            
            # 确保时间列为datetime类型
            df[time_col] = pd.to_datetime(df[time_col])
            
            # 检查是否有非数值列（除了时间列）
            numeric_cols = []
            for col in df.columns:
                if col != time_col:
                    # 尝试转换为数值类型
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df[col] = numeric_series
                        numeric_cols.append(col)
            
            if not numeric_cols:
                raise ValueError("没有找到有效的数值列")
            
            # 返回处理后的数据和列信息
            return df, time_col, numeric_cols
            
        except Exception as e:
            raise ValueError(f"加载CSV文件失败: {str(e)}")

# 主窗口
class LifeLineAnalysisSoftware(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.time_column = None
        self.numeric_columns = []
        self.current_analysis = None
        self.data_type = "静态数据"  # 默认为静态数据
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('生命线工程设备测试数据分析软件')
        self.setGeometry(100, 100, 1500, 1000)
        
        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QTabWidget::pane {
                border: .0625rem solid #ddd;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                border: .0625rem solid #ddd;
                border-bottom-color: #ddd;
                padding: .5rem 1rem;
                margin-right: -0.0625rem;
                color: #495057;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #007bff;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: .5rem 1rem;
                border-radius: .25rem;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QGroupBox {
                font-weight: bold;
                border: .125rem solid #dee2e6;
                border-radius: .3125rem;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: .625rem;
                padding: 0 .3125rem 0 .3125rem;
            }
        """)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题区域
        title_layout = QHBoxLayout()
        title_label = QLabel('生命线工程设备测试数据分析软件')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('SimSun', 18, QFont.Bold))
        title_layout.addWidget(title_label)
        main_layout.addLayout(title_layout)
        
        # 数据类型选择
        data_type_layout = QHBoxLayout()
        self.static_radio = QRadioButton('静态数据')
        self.dynamic_radio = QRadioButton('动态数据')
        self.static_radio.setChecked(True)  # 默认选中静态数据
        self.static_radio.toggled.connect(self.on_data_type_changed)
        self.dynamic_radio.toggled.connect(self.on_data_type_changed)
        
        data_type_layout.addWidget(QLabel('数据类型:'))
        data_type_layout.addWidget(self.static_radio)
        data_type_layout.addWidget(self.dynamic_radio)
        data_type_layout.addStretch()
        main_layout.addLayout(data_type_layout)
        
        # 文件加载区域
        file_layout = QHBoxLayout()
        self.file_label = QLabel('未选择文件')
        self.file_label.setStyleSheet("color: #6c757d; font-weight: bold;")
        self.load_button = QPushButton('加载CSV数据文件')
        self.load_button.setFixedWidth(150)
        self.load_button.clicked.connect(self.load_data)
        self.clear_data_button = QPushButton('清除已加载数据')
        self.clear_data_button.setFixedWidth(150)
        self.clear_data_button.clicked.connect(self.clear_loaded_data)
        self.clear_result_button = QPushButton('清除分析结果')
        self.clear_result_button.setFixedWidth(150)
        self.clear_result_button.clicked.connect(self.clear_analysis_results)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.clear_data_button)
        file_layout.addWidget(self.clear_result_button)
        file_layout.addStretch()
        main_layout.addLayout(file_layout)
        
        # 数据信息显示
        self.data_info = QTextEdit()
        self.data_info.setMaximumHeight(100)
        self.data_info.setReadOnly(True)
        self.data_info.setFont(QFont('SimSun', 9))
        self.data_info.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: .0625rem solid #dee2e6;
                border-radius: .25rem;
            }
        """)
        main_layout.addWidget(QLabel('数据信息:'))
        main_layout.addWidget(self.data_info)
        
        # 数据格式说明
        format_label = QLabel('数据格式要求：CSV文件，第一行为列名或数值（程序自动识别），第一列为时间列')
        format_label.setStyleSheet("background-color: #e9ecef; padding: .3125rem; border: .0625rem solid #ced4da; border-radius: .25rem;")
        format_label.setFont(QFont('SimSun', 9))
        main_layout.addWidget(format_label)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont('SimSun', 10))
        self.create_tabs()
        main_layout.addWidget(self.tabs)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 版本信息标签
        self.version_label = QLabel('V1.0 | 作者: 肖图刚 | 2025年10月16日 | 广东省建筑科学研究院集团股份有限公司 路桥研究所&生命线工程技术中心')
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setFont(QFont('SimSun', 9))
        self.version_label.setStyleSheet("color: #6c757d; background-color: #f8f9fa; padding: .3125rem; border-top: .0625rem solid #dee2e6;")
        main_layout.addWidget(self.version_label)
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        
    def on_data_type_changed(self):
        """数据类型改变时的处理"""
        if self.static_radio.isChecked():
            self.data_type = "静态数据"
        else:
            self.data_type = "动态数据"
        
        # 根据数据类型设置选项卡的可用性
        self.update_tabs_enabled()
    
    def update_tabs_enabled(self):
        """根据数据类型更新选项卡的可用性"""
        if self.data_type == "动态数据":
            # 动态数据：只启用功率谱分析
            self.tabs.setTabEnabled(0, False)  # 异常数据识别与处理
            self.tabs.setTabEnabled(1, False)  # 相关性分析
            self.tabs.setTabEnabled(2, False)  # 统计分析
            self.tabs.setTabEnabled(3, False)  # 趋势分析
            self.tabs.setTabEnabled(4, True)   # 功率谱分析
        else:
            # 静态数据：只启用前四个功能，不启用功率谱分析
            self.tabs.setTabEnabled(0, True)   # 异常数据识别与处理
            self.tabs.setTabEnabled(1, True)   # 相关性分析
            self.tabs.setTabEnabled(2, True)   # 统计分析
            self.tabs.setTabEnabled(3, True)   # 趋势分析
            self.tabs.setTabEnabled(4, False)  # 功率谱分析
    
    def create_tabs(self):
        # 异常数据识别与处理
        self.outlier_tab = QWidget()
        self.setup_outlier_tab()
        self.tabs.addTab(self.outlier_tab, '异常数据识别与处理')
        
        # 相关性分析
        self.correlation_tab = QWidget()
        self.setup_correlation_tab()
        self.tabs.addTab(self.correlation_tab, '相关性分析')
        
        # 统计分析
        self.statistical_tab = QWidget()
        self.setup_statistical_tab()
        self.tabs.addTab(self.statistical_tab, '统计分析')
        
        # 趋势分析
        self.trend_tab = QWidget()
        self.setup_trend_tab()
        self.tabs.addTab(self.trend_tab, '趋势分析')
        
        # 功率谱分析
        self.psd_tab = QWidget()
        self.setup_psd_tab()
        self.tabs.addTab(self.psd_tab, '功能谱分析')
        
        # 初始化时根据默认数据类型设置选项卡可用性
        self.update_tabs_enabled()
    
    def setup_psd_tab(self):
        layout = QVBoxLayout(self.psd_tab)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QFormLayout(params_group)
        
        self.psd_nperseg = QSpinBox()
        self.psd_nperseg.setRange(32, 1024)
        self.psd_nperseg.setValue(256)
        
        # 滤波设置
        self.psd_use_filter = QCheckBox('使用带通滤波')
        self.psd_use_filter.setChecked(True)
        self.psd_low_freq = QDoubleSpinBox()
        self.psd_low_freq.setRange(0.01, 100.0)
        self.psd_low_freq.setValue(0.1)
        self.psd_high_freq = QDoubleSpinBox()
        self.psd_high_freq.setRange(0.1, 100.0)
        self.psd_high_freq.setValue(20.0)
        
        params_layout.addRow('分段长度:', self.psd_nperseg)
        params_layout.addRow('', self.psd_use_filter)
        params_layout.addRow('低频截止(Hz):', self.psd_low_freq)
        params_layout.addRow('高频截止(Hz):', self.psd_high_freq)
        
        layout.addWidget(params_group)
        
        # 信号列选择
        signal_layout = QHBoxLayout()
        self.psd_time_col = QComboBox()
        # 默认时间列为通道0
        self.psd_time_col.addItems(['通道0'] + self.numeric_columns)
        
        # 创建8个信号列选择器，支持选择"无"
        self.psd_signal_cols = []
        for i in range(8):
            combo = QComboBox()
            # 添加"无"选项，然后添加实际数据列
            combo.addItem('无')
            combo.addItems(self.numeric_columns)
            combo.setFixedWidth(120)
            combo.setPlaceholderText(f'通道{i+1}')
            self.psd_signal_cols.append(combo)
            signal_layout.addWidget(QLabel(f'信号{i+1}:'))
            signal_layout.addWidget(combo)
        
        signal_layout.addStretch()
        layout.addLayout(signal_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.psd_btn = QPushButton('执行功率谱分析')
        self.psd_btn.setFixedWidth(150)
        self.psd_btn.clicked.connect(self.run_psd_analysis)
        btn_layout.addWidget(self.psd_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示
        self.psd_result = QTextEdit()
        self.psd_result.setMaximumHeight(100)
        layout.addWidget(QLabel('分析结果:'))
        layout.addWidget(self.psd_result)
        
        # 图形显示区域
        graph_layout = QHBoxLayout()
        
        # 功率谱图
        self.psd_figure = Figure(figsize=(10, 6))
        self.psd_canvas = FigureCanvas(self.psd_figure)
        psd_group = QGroupBox('功率谱密度图')
        psd_group_layout = QVBoxLayout(psd_group)
        psd_group_layout.addWidget(self.psd_canvas)
        
        # 原始数据图
        self.raw_figure = Figure(figsize=(10, 6))
        self.raw_canvas = FigureCanvas(self.raw_figure)
        raw_group = QGroupBox('原始数据图')
        raw_group_layout = QVBoxLayout(raw_group)
        raw_group_layout.addWidget(self.raw_canvas)
        
        graph_layout.addWidget(psd_group)
        graph_layout.addWidget(raw_group)
        layout.addLayout(graph_layout)
    
    def setup_outlier_tab(self):
        layout = QVBoxLayout(self.outlier_tab)
        
        # 变量选择
        var_layout = QHBoxLayout()
        self.outlier_col_combo = QComboBox()
        self.outlier_col_combo.setFixedWidth(150)
        var_layout.addWidget(QLabel('选择分析列:'))
        var_layout.addWidget(self.outlier_col_combo)
        var_layout.addStretch()
        layout.addLayout(var_layout)
        
        # 参数设置
        params_group = QGroupBox('参数设置')
        params_layout = QFormLayout(params_group)
        
        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(1.0, 10.0)
        self.outlier_threshold.setValue(3.0)
        self.outlier_window = QSpinBox()
        self.outlier_window.setRange(3, 50)
        self.outlier_window.setValue(12)
        self.outlier_min_amplitude = QDoubleSpinBox()
        self.outlier_min_amplitude.setRange(0.0, 100.0)
        self.outlier_min_amplitude.setValue(1.0)
        self.outlier_max_iter = QSpinBox()
        self.outlier_max_iter.setRange(1, 20)
        self.outlier_max_iter.setValue(5)
        
        params_layout.addRow('异常阈值:', self.outlier_threshold)
        params_layout.addRow('滑动窗口大小:', self.outlier_window)
        params_layout.addRow('最小振幅:', self.outlier_min_amplitude)
        params_layout.addRow('最大迭代次数:', self.outlier_max_iter)
        
        layout.addWidget(params_group)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.outlier_btn = QPushButton('执行异常检测')
        self.outlier_btn.setFixedWidth(150)
        self.outlier_btn.clicked.connect(self.run_outlier_detection)
        btn_layout.addWidget(self.outlier_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示
        self.outlier_result = QTextEdit()
        self.outlier_result.setMaximumHeight(100)
        layout.addWidget(QLabel('检测结果:'))
        layout.addWidget(self.outlier_result)
        
        # 图形显示
        self.outlier_figure = Figure(figsize=(12, 6))
        self.outlier_canvas = FigureCanvas(self.outlier_figure)
        layout.addWidget(self.outlier_canvas)
    
    def setup_correlation_tab(self):
        layout = QVBoxLayout(self.correlation_tab)
        
        # 变量选择
        var_layout = QHBoxLayout()
        self.x_cols_combo = QComboBox()
        self.y_cols_combo = QComboBox()
        self.x_cols_combo.setFixedWidth(150)
        self.y_cols_combo.setFixedWidth(150)
        var_layout.addWidget(QLabel('X变量:'))
        var_layout.addWidget(self.x_cols_combo)
        var_layout.addWidget(QLabel('Y变量:'))
        var_layout.addWidget(self.y_cols_combo)
        var_layout.addStretch()
        layout.addLayout(var_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.correlation_btn = QPushButton('执行相关性分析')
        self.correlation_btn.setFixedWidth(150)
        self.correlation_btn.clicked.connect(self.run_correlation_analysis)
        btn_layout.addWidget(self.correlation_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示
        self.correlation_result = QTextEdit()
        self.correlation_result.setMaximumHeight(100)
        layout.addWidget(QLabel('相关性结果:'))
        layout.addWidget(self.correlation_result)
        
        # 图形显示
        self.correlation_figure = Figure(figsize=(12, 6))
        self.correlation_canvas = FigureCanvas(self.correlation_figure)
        layout.addWidget(self.correlation_canvas)
    
    def setup_statistical_tab(self):
        layout = QVBoxLayout(self.statistical_tab)
        
        # 传感器选择
        sensor_layout = QHBoxLayout()
        self.sensor_combo = QComboBox()
        self.sensor_combo.setFixedWidth(150)
        sensor_layout.addWidget(QLabel('选择传感器:'))
        sensor_layout.addWidget(self.sensor_combo)
        sensor_layout.addStretch()
        layout.addLayout(sensor_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.statistical_btn = QPushButton('执行统计分析')
        self.statistical_btn.setFixedWidth(150)
        self.statistical_btn.clicked.connect(self.run_statistical_analysis)
        btn_layout.addWidget(self.statistical_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示
        self.statistical_result = QTextEdit()
        self.statistical_result.setMaximumHeight(120)
        layout.addWidget(QLabel('统计结果:'))
        layout.addWidget(self.statistical_result)
        
        # 图形显示
        self.statistical_figure = Figure(figsize=(12, 6))
        self.statistical_canvas = FigureCanvas(self.statistical_figure)
        layout.addWidget(self.statistical_canvas)
    
    def setup_trend_tab(self):
        layout = QVBoxLayout(self.trend_tab)
        
        # 变量选择
        var_layout = QHBoxLayout()
        self.trend_time_col = QComboBox()
        self.trend_value_col = QComboBox()
        self.trend_time_col.setFixedWidth(150)
        self.trend_value_col.setFixedWidth(150)
        var_layout.addWidget(QLabel('时间列:'))
        var_layout.addWidget(self.trend_time_col)
        var_layout.addWidget(QLabel('数值列:'))
        var_layout.addWidget(self.trend_value_col)
        var_layout.addStretch()
        layout.addLayout(var_layout)
        
        # 参数设置
        params_group = QGroupBox('SARIMA参数设置')
        params_layout = QFormLayout(params_group)
        
        self.order_p = QSpinBox()
        self.order_p.setRange(0, 5)
        self.order_p.setValue(1)
        self.order_d = QSpinBox()
        self.order_d.setRange(0, 3)
        self.order_d.setValue(1)
        self.order_q = QSpinBox()
        self.order_q.setRange(0, 5)
        self.order_q.setValue(1)
        self.seasonal_p = QSpinBox()
        self.seasonal_p.setRange(0, 3)
        self.seasonal_p.setValue(1)
        self.seasonal_d = QSpinBox()
        self.seasonal_d.setRange(0, 2)
        self.seasonal_d.setValue(1)
        self.seasonal_q = QSpinBox()
        self.seasonal_q.setRange(0, 3)
        self.seasonal_q.setValue(1)
        self.seasonal_period = QSpinBox()
        self.seasonal_period.setRange(1, 365)
        self.seasonal_period.setValue(24)
        
        params_layout.addRow('AR(p):', self.order_p)
        params_layout.addRow('I(d):', self.order_d)
        params_layout.addRow('MA(q):', self.order_q)
        params_layout.addRow('Seasonal AR(P):', self.seasonal_p)
        params_layout.addRow('Seasonal I(D):', self.seasonal_d)
        params_layout.addRow('Seasonal MA(Q):', self.seasonal_q)
        params_layout.addRow('季节周期:', self.seasonal_period)
        
        layout.addWidget(params_group)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.trend_btn = QPushButton('执行趋势分析')
        self.trend_btn.setFixedWidth(150)
        self.trend_btn.clicked.connect(self.run_trend_analysis)
        btn_layout.addWidget(self.trend_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 结果显示
        self.trend_result = QTextEdit()
        self.trend_result.setMaximumHeight(100)
        layout.addWidget(QLabel('预测结果:'))
        layout.addWidget(self.trend_result)
        
        # 图形显示
        self.trend_figure = Figure(figsize=(12, 6))
        self.trend_canvas = FigureCanvas(self.trend_figure)
        layout.addWidget(self.trend_canvas)
    
    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择CSV数据文件', '', 'CSV Files (*.csv)'
        )
        if file_path:
            try:
                # 使用改进的数据加载器
                self.data, self.time_column, self.numeric_columns = DataLoader.load_csv_with_time_handling(file_path)
                
                # 为通道列创建映射（通道号-1）
                self.channel_mapping = {}
                for i, col in enumerate(self.numeric_columns):
                    self.channel_mapping[f'通道{i+1}'] = col
                
                self.file_label.setText(f'已加载: {os.path.basename(file_path)}')
                
                # 更新数据信息显示
                info_text = (
                    f"文件: {os.path.basename(file_path)}\n"
                    f"数据形状: {self.data.shape[0]} 行 x {self.data.shape[1]} 列\n"
                    f"时间列: {self.time_column}\n"
                    f"数值列: {', '.join(self.numeric_columns)}\n"
                    f"时间范围: {self.data[self.time_column].min()} 至 {self.data[self.time_column].max()}"
                )
                self.data_info.setPlainText(info_text)
                
                self.statusBar().showMessage(f'成功加载数据，共{len(self.data)}行')
                
                # 更新所有列选择器
                self.update_column_combos()
                
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载文件失败: {str(e)}')
    
    def clear_loaded_data(self):
        """清除已加载的数据"""
        self.data = None
        self.time_column = None
        self.numeric_columns = []
        self.file_label.setText('未选择文件')
        self.data_info.setPlainText('')
        self.statusBar().showMessage('数据已清除')
        self.update_column_combos()
    
    def clear_analysis_results(self):
        """清除分析结果"""
        # 清除各选项卡的结果显示
        self.outlier_result.setPlainText('')
        self.correlation_result.setPlainText('')
        self.statistical_result.setPlainText('')
        self.trend_result.setPlainText('')
        self.psd_result.setPlainText('')
        
        # 清除各选项卡的图形
        self.outlier_figure.clear()
        self.outlier_canvas.draw()
        self.correlation_figure.clear()
        self.correlation_canvas.draw()
        self.statistical_figure.clear()
        self.statistical_canvas.draw()
        self.trend_figure.clear()
        self.trend_canvas.draw()
        self.psd_figure.clear()
        self.psd_canvas.draw()
        self.raw_figure.clear()
        self.raw_canvas.draw()
        
        self.statusBar().showMessage('分析结果已清除')
    
    def update_column_combos(self):
        """更新所有列选择器"""
        # 清空所有选择器
        self.outlier_col_combo.clear()
        self.x_cols_combo.clear()
        self.y_cols_combo.clear()
        self.sensor_combo.clear()
        self.trend_time_col.clear()
        self.trend_value_col.clear()
        self.psd_time_col.clear()
        
        for combo in self.psd_signal_cols:
            combo.clear()
            # 为信号选择器添加"无"选项和数值列
            combo.addItem('无')
            combo.addItems(self.numeric_columns)
        
        # 添加时间列和数值列
        if self.time_column:
            self.trend_time_col.addItem(self.time_column)
            # 默认时间列为通道0
            self.psd_time_col.addItem('通道0')
        
        for col in self.numeric_columns:
            self.outlier_col_combo.addItem(col)
            self.x_cols_combo.addItem(col)
            self.y_cols_combo.addItem(col)
            self.sensor_combo.addItem(col)
            self.trend_value_col.addItem(col)
        
        # 设置默认选择
        if self.numeric_columns:
            self.y_cols_combo.setCurrentIndex(0)
            self.trend_value_col.setCurrentIndex(0)
            for i, combo in enumerate(self.psd_signal_cols):
                if i < len(self.numeric_columns):
                    combo.setCurrentIndex(i+1)  # +1 因为第一个是"无"
            if len(self.numeric_columns) > 1:
                self.x_cols_combo.setCurrentIndex(1)
    
    def run_psd_analysis(self):
        if self.data is None or not self.numeric_columns:
            QMessageBox.warning(self, '警告', '请先加载包含数值列的数据文件')
            return
            
        # 获取时间列
        time_col_text = self.psd_time_col.currentText()
        if time_col_text == '通道0':
            # 如果选择通道0，使用实际的时间列
            time_col = self.time_column
        else:
            time_col = time_col_text
        
        # 获取选中的信号列
        signal_cols = []
        for combo in self.psd_signal_cols:
            selected = combo.currentText()
            if selected != '无' and selected in self.numeric_columns and selected not in signal_cols:
                signal_cols.append(selected)
        
        if not time_col or not signal_cols:
            QMessageBox.warning(self, '警告', '请选择时间列和至少一个信号列')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            # 执行功率谱分析
            psd_analyzer = PowerSpectrumAnalysis()
            freqs, psd_all, fs = psd_analyzer.analyze_power_spectrum(
                self.data, time_col, signal_cols, 
                nperseg=self.psd_nperseg.value(),
                use_filter=self.psd_use_filter.isChecked(),
                low_freq=self.psd_low_freq.value(),
                high_freq=self.psd_high_freq.value()
            )
            
            # 更新结果显示
            result_text = f'功率谱分析结果\n'
            result_text += f'采样频率: {fs:.2f} Hz\n'
            result_text += f'滤波设置: {"已启用" if self.psd_use_filter.isChecked() else "未启用"}\n'
            if self.psd_use_filter.isChecked():
                result_text += f'滤波频段: {self.psd_low_freq.value():.2f}Hz - {self.psd_high_freq.value():.2f}Hz\n'
            result_text += f'时间列: {time_col}\n'
            result_text += f'信号列: {", ".join(signal_cols)}\n'
            
            # 找到主频
            for i, (col, f, Pxx) in enumerate(zip(signal_cols, freqs, psd_all)):
                nonzero_idx = np.where(f > 0)[0]
                if len(nonzero_idx) > 0:
                    idx_max = nonzero_idx[np.argmax(Pxx[nonzero_idx])]
                    main_freq = f[idx_max]
                    result_text += f'{col} 主频: {main_freq:.3f} Hz\n'
                else:
                    result_text += f'{col} 主频: 0.000 Hz\n'
            
            self.psd_result.setPlainText(result_text)
            
            # 绘制功率谱
            self.psd_figure.clear()
            ax = self.psd_figure.add_subplot(111)
            
            # 绘制功率谱（对数坐标）
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
            for i, (col, f, Pxx) in enumerate(zip(signal_cols, freqs, psd_all)):
                Pxx_safe = np.maximum(Pxx, 1e-20)  # 避免 log(0)
                ax.semilogy(f, Pxx_safe, label=f'{col}', linewidth=1.2, color=colors[i % len(colors)])
            
            ax.set_title('功率谱密度（Welch方法，已去直流）', fontfamily='SimSun')
            ax.set_xlabel('频率 [Hz]', fontfamily='SimSun')
            ax.set_ylabel('功率谱密度 [(m/s²)²/Hz]', fontfamily='SimSun')
            ax.legend(prop={'family': 'SimSun'}, ncol=min(4, len(signal_cols)))
            ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
            self.psd_figure.tight_layout()
            self.psd_canvas.draw()
            
            # 绘制原始数据
            self.raw_figure.clear()
            ax_raw = self.raw_figure.add_subplot(111)
            
            time = pd.to_numeric(self.data[time_col], errors='coerce').dropna().values
            if np.mean(np.diff(time)) > 100:  # 如果 dt 很大，可能单位是毫秒
                time = time / 1000.0
                
            for i, col in enumerate(signal_cols):
                signal = self.data[col].astype(float).values
                ax_raw.plot(time, signal, label=f'{col}', linewidth=1.0, color=colors[i % len(colors)])
            
            ax_raw.set_title('原始数据时程图', fontfamily='SimSun')
            ax_raw.set_xlabel('时间 [s]', fontfamily='SimSun')
            ax_raw.set_ylabel('数值', fontfamily='SimSun')
            ax_raw.legend(prop={'family': 'SimSun'}, ncol=min(4, len(signal_cols)))
            ax_raw.grid(True, ls='--', lw=0.5, alpha=0.7)
            self.raw_figure.tight_layout()
            self.raw_canvas.draw()
            
            self.statusBar().showMessage('功率谱分析完成')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'功率谱分析失败: {str(e)}')
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
    
    def run_outlier_detection(self):
        if self.data is None or not self.numeric_columns:
            QMessageBox.warning(self, '警告', '请先加载包含数值列的数据文件')
            return
            
        selected_col = self.outlier_col_combo.currentText()
        if not selected_col:
            QMessageBox.warning(self, '警告', '请选择要分析的列')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress bar
        
        try:
            # 这一步是正确的：创建一个以时间为索引的Series
            series = pd.Series(self.data[selected_col].values, index=self.data[self.time_column])
            
            # 执行异常检测
            clean_data, outliers = OutlierDetection.preprocess_series(
                series, 
                outlier_thresh=self.outlier_threshold.value(),
                window_size=self.outlier_window.value(),
                min_amplitude=self.outlier_min_amplitude.value(),
            )
            
            # 更新结果显示
            outlier_count = outliers.sum()
            
            # 【修复 1】: 直接从 series 和 outliers 中获取异常点信息
            # 因为 series 和 outliers 索引对齐，可以直接筛选
            outlier_series = series[outliers]
            
            self.outlier_result.setPlainText(
                f'异常点检测完成 - 列: {selected_col}\n'
                f'总数据点: {len(series)}\n'
                f'异常点数量: {outlier_count}\n'
                f'异常点比例: {outlier_count/len(series)*100:.2f}%\n'
                # 使用 outlier_series 的索引来获取时间范围
                f'异常点时间范围: {outlier_series.index.min()} 至 {outlier_series.index.max() if outlier_count > 0 else "N/A"}'
            )
            
            # 绘制结果
            self.outlier_figure.clear()
            ax = self.outlier_figure.add_subplot(111)
            
            # 【修复 2】: 绘图时，使用索引对齐的 series 和 clean_data
            # 使用 series.index 作为 x 轴，series.values 作为 y 轴
            ax.plot(series.index, series.values, 
                    label='原始数据', color='black', linestyle='--', alpha=0.6)
            # 使用 clean_data.index 作为 x 轴，clean_data.values 作为 y 轴
            ax.plot(clean_data.index, clean_data.values, 
                    label='清洗后数据', color='tab:orange', linewidth=1.5)
            
            # 【修复 3】: 绘制异常点时，使用前面计算好的 outlier_series
            if not outlier_series.empty:
                ax.scatter(outlier_series.index, outlier_series.values, 
                           color='red', label='异常点', zorder=5, s=30)
            
            ax.set_title(f'异常识别与处理结果 - {selected_col}', fontfamily='SimSun')
            ax.set_xlabel('时间', fontfamily='SimSun')
            ax.set_ylabel('数值', fontfamily='SimSun')
            ax.legend(prop={'family': 'SimSun'})
            ax.grid(True, linestyle='--', alpha=0.3)
            self.outlier_figure.tight_layout() # 优化布局，防止标签重叠
            self.outlier_canvas.draw()
            
            self.statusBar().showMessage('异常检测完成')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'异常检测失败: {str(e)}')
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
    
    def run_correlation_analysis(self):
        if self.data is None or not self.numeric_columns:
            QMessageBox.warning(self, '警告', '请先加载包含数值列的数据文件')
            return
            
        x_col = self.x_cols_combo.currentText()
        y_col = self.y_cols_combo.currentText()
        
        if not x_col or not y_col:
            QMessageBox.warning(self, '警告', '请选择X和Y变量')
            return
            
        if x_col == y_col:
            QMessageBox.warning(self, '警告', 'X变量和Y变量不能相同')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            # 执行相关性分析
            corr_analyzer = CorrelationAnalysis()
            x_data, y_data, model = corr_analyzer.analyze(
                self.data, [x_col], y_col
            )
            
            # 计算相关系数
            correlation = np.corrcoef(x_data.flatten(), y_data.flatten())[0, 1]
            
            # 更新结果显示
            self.correlation_result.setPlainText(
                f'相关性分析结果\n'
                f'X变量: {x_col}\n'
                f'Y变量: {y_col}\n'
                f'相关系数: {correlation:.4f}\n'
                f'回归系数: {model.coef_[0][0]:.4f}\n'
                f'截距: {model.intercept_[0]:.4f}\n'
                f'决定系数 (R²): {correlation**2:.4f}'
            )
            
            # 绘制散点图和拟合直线
            self.correlation_figure.clear()
            ax = self.correlation_figure.add_subplot(111)
            ax.scatter(x_data, y_data, color='black', s=20, alpha=0.6, label='数据点')
            
            # 绘制拟合直线
            x_range = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            ax.plot(x_range, y_pred, color='red', linewidth=2, 
                   label=f'拟合直线 (R²={correlation**2:.4f})')
            
            ax.set_xlabel(x_col, fontfamily='SimSun')
            ax.set_ylabel(y_col, fontfamily='SimSun')
            ax.set_title(f'{y_col} vs {x_col} 相关性分析', fontfamily='SimSun')
            ax.legend(prop={'family': 'SimSun'})
            ax.grid(True, alpha=0.3)
            self.correlation_canvas.draw()
            
            self.statusBar().showMessage('相关性分析完成')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'相关性分析失败: {str(e)}')
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
    
    def run_statistical_analysis(self):
        if self.data is None or not self.numeric_columns:
            QMessageBox.warning(self, '警告', '请先加载包含数值列的数据文件')
            return
            
        sensor_col = self.sensor_combo.currentText()
        if not sensor_col:
            QMessageBox.warning(self, '警告', '请选择要分析的传感器列')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            series = self.data[sensor_col]
            
            # 执行统计分析
            stat_analyzer = StatisticalAnalysis()
            stats = stat_analyzer.advanced_stats(series)
            gmm = stat_analyzer.fit_gmm(series)
            
            # 更新结果显示
            result_text = (
                f'统计分析结果 - {sensor_col}\n'
                f'数据点数: {stats["count"]}\n'
                f'均值: {stats["mean"]:.4f}\n'
                f'方差: {stats["var"]:.4f}\n'
                f'标准差: {stats["std"]:.4f}\n'
                f'中位数: {stats["median"]:.4f}\n'
                f'最大值: {stats["max"]:.4f}\n'
                f'最小值: {stats["min"]:.4f}\n'
                f'25%分位数: {stats["q25"]:.4f}\n'
                f'75%分位数: {stats["q75"]:.4f}\n'
                f'偏度: {stats["skewness"]:.4f}\n'
                f'峰度: {stats["kurtosis"]:.4f}\n'
                f'变异系数: {stats["cv"]:.4f}\n'
                f'平均绝对偏差: {stats["mad"]:.4f}\n'
                f'GMM组件数: 2\n'
                f'GMM权重: {[f"{w:.4f}" for w in gmm.weights_]}\n'
                f'GMM均值: {[f"{m[0]:.4f}" for m in gmm.means_]}\n'
                f'GMM方差: {[f"{c[0,0]:.4f}" for c in gmm.covariances_]}'
            )
            self.statistical_result.setPlainText(result_text)
            
            # 绘制直方图和GMM
            self.statistical_figure.clear()
            ax = self.statistical_figure.add_subplot(111)
            
            # 绘制直方图
            ax.hist(series.dropna(), bins=50, density=True, alpha=0.6, 
                   color='skyblue', edgecolor='black', label='数据分布')
            
            # 绘制GMM拟合曲线
            from scipy.stats import norm
            x_range = np.linspace(series.min(), series.max(), 400)
            gmm_values = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
            ax.plot(x_range, gmm_values, 'r-', linewidth=2, label='GMM拟合')
            
            ax.set_title(f'{sensor_col} 统计分布分析', fontfamily='SimSun')
            ax.set_xlabel('数值', fontfamily='SimSun')
            ax.set_ylabel('概率密度', fontfamily='SimSun')
            ax.legend(prop={'family': 'SimSun'})
            ax.grid(True, alpha=0.3)
            self.statistical_canvas.draw()
            
            self.statusBar().showMessage('统计分析完成')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'统计分析失败: {str(e)}')
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
    
    def run_trend_analysis(self):
        if self.data is None or not self.numeric_columns:
            QMessageBox.warning(self, '警告', '请先加载包含数值列的数据文件')
            return
            
        time_col = self.trend_time_col.currentText()
        value_col = self.trend_value_col.currentText()
        
        if not time_col or not value_col:
            QMessageBox.warning(self, '警告', '请选择时间列和数值列')
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            # 加载时间序列数据
            trend_analyzer = TrendAnalysis()
            series = trend_analyzer.load_series(self.data, time_col, value_col)
            
            # 训练SARIMA模型
            order = (self.order_p.value(), self.order_d.value(), self.order_q.value())
            seasonal_order = (
                self.seasonal_p.value(), 
                self.seasonal_d.value(), 
                self.seasonal_q.value(), 
                self.seasonal_period.value()
            )
            
            model_result = trend_analyzer.train_sarima(series, order, seasonal_order)
            
            # 分析趋势成分
            trend, seasonal, residual = trend_analyzer.analyze_trend_components(series)
            
            # 进行预测
            forecast_steps = 24  # 预测未来24个时间点
            forecast_result = model_result.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            # 更新结果显示
            result_text = (
                f'趋势分析结果\n'
                f'分析列: {value_col}\n'
                f'模型参数: SARIMA{order}x{seasonal_order}\n'
                f'预测步数: {forecast_steps}\n'
                f'历史数据范围: {series.min():.4f} ~ {series.max():.4f}\n'
                f'预测均值范围: {forecast_mean.min():.4f} ~ {forecast_mean.max():.4f}\n'
                f'置信区间: [{forecast_ci.iloc[0,0]:.4f}, {forecast_ci.iloc[0,1]:.4f}]\n'
                f'模型AIC: {model_result.aic:.4f}\n'
                f'模型BIC: {model_result.bic:.4f}\n'
                f'模型HQIC: {model_result.hqic:.4f}'
            )
            self.trend_result.setPlainText(result_text)
            
            # 绘制趋势图
            self.trend_figure.clear()
            ax = self.trend_figure.add_subplot(111)
            
            # 绘制原始数据（最近100个点）
            recent_data = series.tail(100)
            ax.plot(recent_data.index, recent_data.values, label='原始数据', color='black')
            
            # 绘制历史拟合值（最近100个点）
            fitted_values = model_result.fittedvalues.tail(100)
            if len(fitted_values) > 0:
                ax.plot(recent_data.index[-len(fitted_values):], fitted_values, 
                       label='模型拟合值', color='blue', linestyle='--')
            
            # 绘制预测值和置信区间
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(hours=1),
                periods=forecast_steps,
                freq='H'
            )
            ax.plot(forecast_index, forecast_mean, label='预测值', color='red')
            ax.fill_between(
                forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='red', alpha=0.2, label='95%置信区间'
            )
            
            ax.set_title(f'{value_col} 时间序列趋势分析与预测', fontfamily='SimSun')
            ax.set_xlabel('时间', fontfamily='SimSun')
            ax.set_ylabel('数值', fontfamily='SimSun')
            ax.legend(prop={'family': 'SimSun'})
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            self.trend_canvas.draw()
            
            self.statusBar().showMessage('趋势分析完成')
            
        except Exception as e:
            QMessageBox.critical(self, '错误', f'趋势分析失败: {str(e)}')
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用现代化样式
    window = LifeLineAnalysisSoftware()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()