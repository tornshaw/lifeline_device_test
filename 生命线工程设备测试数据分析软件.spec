# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['login_window.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'statsmodels', 'scipy', 'scipy.signal', 'scipy.stats', 'sklearn.linear_model', 'sklearn.mixture', 'statsmodels.tsa', 'statsmodels.tsa.statespace', 'statsmodels.tsa.statespace.sarimax'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='生命线工程设备测试数据分析软件',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
