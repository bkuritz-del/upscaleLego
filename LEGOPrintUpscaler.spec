# LEGOPrintUpscaler.spec

a = Analysis(
    ['gui_upscale_v2.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=['upscale_core'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LEGOPrintUpscaler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
