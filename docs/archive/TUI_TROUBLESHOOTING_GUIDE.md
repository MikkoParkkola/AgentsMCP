# TUI Troubleshooting Guide

## Quick Start - Is Your TUI Working?

If you're experiencing TUI issues, run this diagnostic first:

```bash
python comprehensive_tui_diagnostic.py --quick
```

## Understanding the Diagnostic Results

### Exit Codes
- **0**: ✅ No issues - TUI should work perfectly
- **1**: ⚠️ Minor issues - TUI should work with warnings
- **2**: ❌ Major issues - TUI may not function properly  
- **3**: 💀 Critical issues - TUI will not work
- **4**: 🔧 Script error - Contact support

### Status Indicators
- **✓** (Green): Component working correctly
- **⚠** (Yellow): Warning - may cause issues
- **✗** (Red): Failed - will cause problems

## Common Issues and Solutions

### 1. "No TTY detected" Error

**Symptoms**: TUI starts but input doesn't work properly, layout is corrupted

**Solutions**:
- ✅ Run in a real terminal (Terminal.app, iTerm2, gnome-terminal)
- ✅ Avoid IDE consoles (VS Code integrated terminal, PyCharm console)
- ✅ Don't pipe output: `./agentsmcp tui` not `./agentsmcp tui | less`

### 2. "No color support detected" Warning

**Symptoms**: Text appears in black/white only, no visual highlighting

**Solutions**:
- Enable color support in your terminal settings
- Set `COLORTERM=truecolor` environment variable
- Use a modern terminal that supports 256+ colors

### 3. Import/Module Errors

**Symptoms**: "Module not found" or import errors

**Solutions**:
- Run from project root directory: `cd /path/to/AgentsMCP`
- Install dependencies: `pip install -r requirements.txt`
- Check Python path: `PYTHONPATH=/path/to/AgentsMCP/src python ...`

### 4. Terminal Size Issues

**Symptoms**: Layout appears cramped or text is cut off

**Solutions**:
- Resize terminal to at least 80x24
- Use full-screen terminal window
- Check terminal font size (smaller = more text fits)

### 5. Encoding/Unicode Issues

**Symptoms**: Strange characters, boxes instead of text, encoding errors

**Solutions**:
- Set locale: `export LANG=en_US.UTF-8`
- Use UTF-8 compatible terminal
- Check terminal encoding settings

## Diagnostic Command Options

### Standard Output (Default)
```bash
python comprehensive_tui_diagnostic.py
```
Colorized, human-readable output with all diagnostic information.

### Quick Mode (Faster)
```bash
python comprehensive_tui_diagnostic.py --quick
```
Essential checks only, faster execution for basic troubleshooting.

### Verbose Mode (Detailed)  
```bash
python comprehensive_tui_diagnostic.py --verbose
```
Detailed diagnostic information with context and technical details.

### JSON Output (Automation)
```bash
python comprehensive_tui_diagnostic.py --json
```
Machine-readable format perfect for automation, CI/CD, or support tickets.

### Combined Options
```bash
python comprehensive_tui_diagnostic.py --verbose --json > diagnostic.json
```

## Environment-Specific Issues

### macOS
- **Terminal.app**: Usually works well
- **iTerm2**: Excellent support, recommended
- **VS Code**: May have TTY issues, use external terminal

### Linux
- **gnome-terminal**: Usually works well
- **konsole**: Good support
- **tmux/screen**: May need special configuration
- **SSH**: Works if terminal forwarding is enabled

### Windows
- **Windows Terminal**: Recommended
- **PowerShell**: Basic support
- **Command Prompt**: Limited support
- **WSL**: Good support when properly configured

## Advanced Troubleshooting

### Performance Issues
If TUI is slow or laggy:
```bash
# Check memory and performance metrics
python comprehensive_tui_diagnostic.py --verbose | grep -A5 "Performance"

# Monitor during runtime
top -p $(pgrep -f "agentsmcp tui")
```

### Network Issues
If remote features don't work:
```bash
# Check network connectivity
curl -I https://api.openai.com/v1/models
ping 8.8.8.8
```

### Debug Mode
Enable debug logging:
```bash
export AGENTSMCP_DEBUG=1
./agentsmcp tui
```

## Creating Support Tickets

When reporting issues, include:

1. **Diagnostic Output**:
   ```bash
   python comprehensive_tui_diagnostic.py --verbose > diagnostic.txt
   ```

2. **System Information**:
   ```bash
   uname -a
   python --version
   which python
   echo $TERM
   echo $COLORTERM
   ```

3. **Error Logs**:
   ```bash
   ./agentsmcp tui 2>&1 | tee error.log
   ```

4. **Steps to Reproduce**: Exact commands and actions that cause the issue

## Known Limitations

### Current Limitations
- Requires terminal with TTY support
- Best performance in terminals with true color support
- Some features may not work in terminal multiplexers

### Planned Improvements
- Better fallback for non-TTY environments
- Enhanced terminal compatibility detection
- Improved error messages and recovery

## Frequently Asked Questions

**Q: TUI works but typing is invisible**
A: This was fixed in recent updates. Run the diagnostic to confirm your environment.

**Q: Enter key doesn't send messages**
A: Fixed with async/sync wrapper. Ensure you're running the latest version.

**Q: Layout gets corrupted when typing**
A: This indicates a terminal compatibility issue. Try a different terminal.

**Q: TUI exits immediately**
A: Usually a TTY or signal handling issue. Run diagnostic for specific cause.

**Q: Can I use TUI over SSH?**
A: Yes, if the SSH session has proper terminal forwarding (`ssh -t`).

## Getting Help

1. **First**: Run the comprehensive diagnostic
2. **Check**: This troubleshooting guide
3. **Search**: Existing issues in the repository
4. **Report**: New issue with diagnostic output

For immediate help, include the full diagnostic output:
```bash
python comprehensive_tui_diagnostic.py --verbose --json
```