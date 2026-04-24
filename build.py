# build.py
import os
import subprocess
import sys

def build_with_pyinstaller():
    """使用PyInstaller打包"""
    cmd = [
        "pyinstaller",
        "--onefile",           # 打包成单个exe文件
        "--windowed",          # 窗口模式（不显示控制台）
        "--name", "生命线工程设备测试数据分析软件",
        "--add-data", "data;data",  # 包含data目录
        "--hidden-import=pandas",
        "--hidden-import=numpy", 
        "--hidden-import=matplotlib",
        "--hidden-import=seaborn",
        "--hidden-import=sklearn",
        "--hidden-import=statsmodels",
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=scipy.stats",
        "--hidden-import=sklearn.linear_model",
        "--hidden-import=sklearn.mixture",
        "--hidden-import=statsmodels.tsa",
        "--hidden-import=statsmodels.tsa.statespace",
        "--hidden-import=statsmodels.tsa.statespace.sarimax",
        "login_window.py"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("打包成功！")
        print("生成的exe文件在 dist 目录中")
    except subprocess.CalledProcessError as e:
        print(f"打包失败: {e}")

if __name__ == "__main__":
    build_with_pyinstaller()