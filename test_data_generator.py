# test_data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_test_data():
    """生成多种时间格式的测试数据"""
    # 生成时间序列
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(1000)]
    
    # 生成模拟传感器数据
    np.random.seed(42)
    
    # 温度数据（带季节性）
    temp_base = 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 8760)  # 年周期
    temperature = temp_base + np.random.normal(0, 2, 1000)
    
    # 位移数据（与温度相关）
    displacement = 0.5 * temperature + np.random.normal(0, 0.5, 1000)
    
    # 添加一些异常值
    outlier_indices = np.random.choice(1000, 20, replace=False)
    temperature[outlier_indices] += np.random.choice([-10, 10], 20)
    
    # 应变数据
    strain = 0.1 * temperature + 0.3 * displacement + np.random.normal(0, 0.1, 1000)
    
    # 湿度数据
    humidity = 60 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 365) + np.random.normal(0, 3, 1000)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'displacement': displacement,
        'strain': strain,
        'humidity': humidity
    })
    
    # 保存为CSV
    df.to_csv('test_data.csv', index=False)
    print("测试数据已生成: test_data.csv")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    
    # 生成另一种时间格式的数据
    df_iso = df.copy()
    df_iso['timestamp'] = df_iso['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_iso.to_csv('test_data_iso_format.csv', index=False)
    print("ISO格式测试数据已生成: test_data_iso_format.csv")
    
    # 生成其他格式的数据
    df_alt = df.copy()
    df_alt['timestamp'] = df_alt['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S')
    df_alt.to_csv('test_data_alt_format.csv', index=False)
    print("替代格式测试数据已生成: test_data_alt_format.csv")

if __name__ == "__main__":
    generate_test_data()