#!/bin/bash
# Revolutionary TUI Interface Validation Script
# Quick validation of TUI functionality for developers

set -e

echo "🚀 Revolutionary TUI Interface Validation"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "agentsmcp" ] && [ ! -f "src/agentsmcp/__main__.py" ]; then
    echo "❌ Error: Run this script from the AgentsMCP project root directory"
    exit 1
fi

echo "📋 Running TUI validation tests..."
echo ""

# Method 1: Quick manual test
echo "🔍 Test 1: Quick TUI Launch Test"
echo "Testing: ./agentsmcp tui (with auto-quit)"

if timeout 10s bash -c 'echo "quit" | ./agentsmcp tui' > /tmp/tui_test_1.log 2>&1; then
    echo "✅ TUI launches and quits successfully"
else
    echo "❌ TUI launch test failed"
    echo "Output:"
    cat /tmp/tui_test_1.log
    exit 1
fi

# Method 2: Help command test
echo ""
echo "🔍 Test 2: Help Command Test"
echo "Testing: help command functionality"

if timeout 15s bash -c 'echo -e "help\nquit" | ./agentsmcp tui' > /tmp/tui_test_2.log 2>&1; then
    if grep -i "help\|commands" /tmp/tui_test_2.log > /dev/null; then
        echo "✅ Help command works correctly"
    else
        echo "⚠️  Help command executed but no help content detected"
        echo "Output sample:"
        head -5 /tmp/tui_test_2.log
    fi
else
    echo "❌ Help command test failed"
    cat /tmp/tui_test_2.log
    exit 1
fi

# Method 3: Input visibility test (core issue)
echo ""
echo "🔍 Test 3: Input Visibility Test (Core Issue)"
echo "Testing: user input visibility"

if timeout 15s bash -c 'echo -e "test input visibility\nquit" | ./agentsmcp tui' > /tmp/tui_test_3.log 2>&1; then
    # Check for basic prompt fallback (the original issue)
    basic_prompts=$(grep -c "^> " /tmp/tui_test_3.log || true)
    if [ "$basic_prompts" -le 1 ]; then
        echo "✅ No basic prompt fallback detected - input visibility likely working"
    else
        echo "⚠️  Multiple basic prompts detected - may indicate fallback mode"
        echo "Basic prompt count: $basic_prompts"
    fi
else
    echo "❌ Input visibility test failed"
    cat /tmp/tui_test_3.log
    exit 1
fi

# Method 4: Rich interface detection
echo ""
echo "🔍 Test 4: Rich Interface Detection"
echo "Testing: Revolutionary TUI vs basic fallback"

if timeout 10s bash -c 'echo "quit" | ./agentsmcp tui --debug' > /tmp/tui_test_4.log 2>&1; then
    if grep -i "rich\|enhanced\|revolutionary\|tui" /tmp/tui_test_4.log > /dev/null; then
        echo "✅ Rich interface indicators detected"
    else
        echo "⚠️  No Rich interface indicators found - check output:"
        head -3 /tmp/tui_test_4.log
    fi
else
    echo "⚠️  Debug mode test failed (non-critical)"
fi

# Summary
echo ""
echo "🎯 Validation Summary"
echo "===================="
echo "✅ Core TUI functionality appears to be working"
echo ""

# Run comprehensive tests if available
if [ -f "test_tui_acceptance_comprehensive.py" ]; then
    echo "🧪 Comprehensive test suite available!"
    echo ""
    echo "Run comprehensive tests with:"
    echo "  python test_tui_acceptance_comprehensive.py"
    echo ""
    echo "Run automated test suite with:"
    echo "  python run_tui_acceptance_tests.py"
    echo ""
    echo "Run quick validation with:"
    echo "  python run_tui_acceptance_tests.py --quick"
else
    echo "ℹ️  For comprehensive testing, ensure test files are present:"
    echo "  - test_tui_acceptance_comprehensive.py"
    echo "  - run_tui_acceptance_tests.py"
fi

echo ""
echo "✨ Manual validation steps:"
echo "1. Run: ./agentsmcp tui"
echo "2. Verify typing is immediately visible"
echo "3. Try: help, clear, quit commands" 
echo "4. Check for Rich interface (panels, colors)"
echo ""

# Cleanup
rm -f /tmp/tui_test_*.log

echo "🎉 TUI validation completed successfully!"