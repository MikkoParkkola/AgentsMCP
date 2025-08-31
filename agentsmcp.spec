# -*- mode: python ; coding: utf-8 -*-

# PyInstaller spec for AgentsMCP
# Notes:
# - Include src/ on sys.path via pathex to ensure local sources are bundled
# - Add hiddenimports for modules imported dynamically at runtime (e.g., TUI shell)
# - Disable UPX on macOS to avoid build instability

a = Analysis(
    ['src/agentsmcp/cli.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # UI modules
        'agentsmcp.ui.cli_app',
        'agentsmcp.ui.tui_shell',
        'agentsmcp.ui.modern_tui',
        'agentsmcp.ui.status_dashboard',
        'agentsmcp.ui.statistics_display',
        'agentsmcp.ui.theme_manager',
        'agentsmcp.ui.ui_components',
        'agentsmcp.ui.command_interface',
        # UI components used by ModernTUI
        'agentsmcp.ui.components.enhanced_chat',
        'agentsmcp.ui.components.chat_history',
        'agentsmcp.ui.components.realtime_input',
        # v2 TUI system
        'agentsmcp.ui.v2',
        'agentsmcp.ui.v2.main_app',
        'agentsmcp.ui.v2.fixed_working_tui',
        'agentsmcp.ui.v2.minimal_working_tui',
        'agentsmcp.ui.v2.input_handler',
        # Conversation/command routing
        'agentsmcp.conversation.command_interface_impl',
        'agentsmcp.conversation.dispatcher',
        # Optional runtime deps (safe to include if present)
        'httpx',
        'uvicorn',
        'rich',
        'click',
    ],
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
    name='agentsmcp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Ensure the built single-file binary is emitted under ./dist
# Using COLLECT with name='.' places the EXE directly in the dist root
# (avoids creating an extra subdirectory).
coll = COLLECT(
    exe,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='.',
    distpath='dist',
)
