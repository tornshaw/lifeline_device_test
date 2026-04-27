# 测试产物说明（文本化可提交版本）

由于部分平台不支持二进制文件在线查看，本目录改为提供**可直接查看/下载的文本产物**：

- [测试汇总 JSON](./test_summary.json)
- [统计输入样例 CSV](./stats_input.csv)

> 如需图像（相关性回归图、趋势预测图、GMM图），可在本地按测试命令重新生成。

## 本地重生成命令（Headless）

```bash
python - <<'PY'
import json, types, sys, warnings
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
warnings.filterwarnings('ignore')

mod=types.ModuleType('tabulate'); mod.tabulate=lambda *a, **k: 'tabulate_unavailable'; sys.modules['tabulate']=mod
out=Path('test_outputs'); out.mkdir(exist_ok=True)
raw=pd.read_csv('test_data.csv')
summary={
 'rows': int(len(raw)),
 'columns': raw.columns.tolist(),
 'time_start': raw['timestamp'].min(),
 'time_end': raw['timestamp'].max(),
 'missing_values': raw.isna().sum().to_dict(),
 'temperature_mean': float(raw['temperature'].mean()),
 'displacement_mean': float(raw['displacement'].mean()),
 'strain_mean': float(raw['strain'].mean()),
 'humidity_mean': float(raw['humidity'].mean()),
}

corr_mod=SourceFileLoader('corr','others/correlation analysis.py').load_module()
x,y,model=corr_mod.correlation_analysis(raw,['temperature'],'displacement')
summary['corr_temperature_displacement']=float(np.corrcoef(raw['temperature'],raw['displacement'])[0,1])
summary['regression_slope']=float(model.coef_[0][0])
summary['regression_intercept']=float(model.intercept_[0])
summary['regression_r2']=float(model.score(x,y))
plt.figure(figsize=(7,5)); plt.scatter(x,y,s=8,c='black',alpha=0.6,label='samples')
xx=np.linspace(x.min(),x.max(),100).reshape(-1,1); plt.plot(xx,model.predict(xx),c='red',lw=2,label='fit')
plt.xlabel('temperature'); plt.ylabel('displacement'); plt.title('Temperature vs Displacement'); plt.legend(); plt.tight_layout(); plt.savefig(out/'correlation_regression.png',dpi=150); plt.close()

trend_mod=SourceFileLoader('trend','others/trend.py').load_module()
series=trend_mod.load_series('data/def.csv','time','value')
res=trend_mod.train_sarima(series,seasonal_period=24,order=(1,1,1),seasonal_order=(1,1,1,24))
fc=trend_mod.plot_forecast(series,res,steps_ahead=24,title='Trend Forecast Test',out_path=out/'trend_forecast.png',show=False)
summary['sarima_aic']=float(res.aic)
summary['forecast_first_3']=fc['mean'].head(3).round(6).tolist()

stats_mod=SourceFileLoader('stats','others/stats.py').load_module()
wide=raw.rename(columns={'timestamp':'time'})[['time','temperature','displacement','strain','humidity']]
wide.to_csv(out/'stats_input.csv',index=False)
df_all=stats_mod.load_monitor_data(out/'stats_input.csv')
sel=stats_mod.filter_data(df_all,['temperature','displacement'],('2023-01-01','2023-02-12'))
ser=sel.loc[sel['sensor_id']=='temperature','reading']
gmm=stats_mod.fit_gmm(ser)
stats_mod.plot_hist_gmm(ser,gmm,'temperature',fig_dir=out,show=False)
summary['gmm_weights_temperature']=[float(v) for v in gmm.weights_]
summary['gmm_means_temperature']=[float(v[0]) for v in gmm.means_]

(out/'test_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('done')
PY
```

# 已转为 Markdown 的申报文档

- [软件需求文件（MD）](../生命线工程设备测试数据分析软件v1.0-申请材料/生命线工程设备测试数据分析软件v1.0-软件需求文件.md)
- [软件设计文件（MD）](../生命线工程设备测试数据分析软件v1.0-申请材料/生命线工程设备测试数据分析软件v1.0-软件设计文件.md)
- [软件测试报告（MD）](../生命线工程设备测试数据分析软件v1.0-申请材料/生命线工程设备测试数据分析软件v1.0-软件测试报告.md)
