# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all


# Analysis for Tranquilizer
analysis_tranquilizer = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[("assets", "assets"), ("test", "test"), ("models", "models")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz_tranquilizer = PYZ(analysis_tranquilizer.pure)

exe_tranquilizer = EXE(
    pyz_tranquilizer,
    analysis_tranquilizer.scripts,
    [],
    exclude_binaries=True,
    name='Tranquilizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico"
)

# Analysis for TranqTrain
tr_datas = []
tr_binaries = []
tr_hiddenimports = ['unidecode', "text_unidecode"]
tmp_ret = collect_all('unidecode')
tr_datas += tmp_ret[0]; tr_binaries += tmp_ret[1]; tr_hiddenimports += tmp_ret[2]

analysis_tranqtrain = Analysis(
    ['tranq_train.py'],
    pathex=[],
    binaries=[] + tr_binaries,
    datas=[("assets", "."), ("test", "."), ("models", ".")] + tr_datas,  # Shared data
    hiddenimports=[] + tr_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz_tranqtrain = PYZ(analysis_tranqtrain.pure)

exe_tranqtrain = EXE(
    pyz_tranqtrain,
    analysis_tranqtrain.scripts,
    [],
    exclude_binaries=True,
    name='TranqTrain',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico"
)

# Collect both executables and shared resources
coll = COLLECT(
    exe_tranquilizer,
    exe_tranqtrain,
    analysis_tranquilizer.binaries + analysis_tranqtrain.binaries,
    analysis_tranquilizer.datas,  # Shared data resources
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Tranquilizer',
)
