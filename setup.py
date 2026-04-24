# setup.py
from cx_Freeze import setup, Executable
import sys

# 添加依赖包
build_exe_options = {
    "packages": [
        "pandas", 
        "numpy", 
        "matplotlib", 
        "seaborn", 
        "sklearn", 
        "statsmodels", 
        "scipy",
        "PyQt5"
    ],
    "excludes": [],
    "include_files": [("data", "data")]  # 包含data目录
}

# 可执行文件配置
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # 不显示控制台窗口

executables = [
    Executable(
        "login_window.py",
        base=base,
        target_name="生命线工程设备测试数据分析软件.exe"
    )
]

setup(
    name="生命线工程设备测试数据分析软件",
    version="1.0",
    description="面向生命线工程设备测试平台的一站式监测数据分析软件",
    options={"build_exe": build_exe_options},
    executables=executables
)