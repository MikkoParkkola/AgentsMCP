# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['agentsmcp.cli']
tmp_ret = collect_all('agentsmcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/agentsmcp/cli.py'],
    pathex=['src', 'src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude heavy optional stacks to speed up build and startup
    excludes=[
        'torch','torchvision','torchaudio','tensorflow','tf_keras','jax',
        'scipy','pandas','sklearn','matplotlib','numba','llvmlite','pyarrow',
        'transformers','datasets','onnxruntime','django','yt_dlp','librosa',
        'sounddevice','grpc','google','opentelemetry','sentry_sdk','PIL.ImageTk',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='agentsmcp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['none'],
)
