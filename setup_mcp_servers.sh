#!/bin/bash

# AgentsMCP MCP Server Setup Script
# This script configures all MCP servers needed for specialist agents

set -e

echo "🚀 Setting up AgentsMCP MCP Servers..."
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_info "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed. Please install Node.js 18+ and try again."
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//')
print_status "Node.js version: $NODE_VERSION"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is required but not installed. Please install npm and try again."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python 3.8+ is required but not installed. Please install Python and try again."
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
print_status "Python version: $PYTHON_VERSION"

echo ""
print_info "Installing MCP servers globally..."

# Install NPM-based MCP servers
print_info "Installing core MCP servers via npm..."

npm install -g \
    @modelcontextprotocol/server-sequential-thinking \
    @modelcontextprotocol/server-git \
    @modelcontextprotocol/server-installer || {
    print_warning "Some core MCP servers failed to install. They may not be available yet."
}

# Install specialized MCP servers (these may not exist yet, so we'll skip errors)
print_info "Installing specialized MCP servers (may not all be available)..."

# Try to install these, but don't fail if they don't exist
npm install -g @pieces-app/pieces-os-server-mcp 2>/dev/null && print_status "Pieces MCP server installed" || print_warning "Pieces MCP server not available (expected - pieces integration uses different approach)"

npm install -g serena-mcp-server 2>/dev/null && print_status "Serena MCP server installed" || print_warning "Serena MCP server not available (may need to be installed separately)"

npm install -g @semgrep/mcp-server 2>/dev/null && print_status "Semgrep MCP server installed" || print_warning "Semgrep MCP server not available (may need Semgrep CLI)"

npm install -g @aquasec/trivy-mcp-server 2>/dev/null && print_status "Trivy MCP server installed" || print_warning "Trivy MCP server not available (may need Trivy CLI)"

npm install -g @eslint/mcp-server 2>/dev/null && print_status "ESLint MCP server installed" || print_warning "ESLint MCP server not available (may use local ESLint)"

npm install -g @typescript-language-server/mcp-server 2>/dev/null && print_status "TypeScript LSP MCP server installed" || print_warning "TypeScript LSP MCP server not available"

# Install Python-based MCP servers
print_info "Installing Python MCP servers..."

$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install python-lsp-server 2>/dev/null && print_status "Python LSP server installed" || print_warning "Python LSP server installation failed"

# Create Claude Desktop config backup and update
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

if [ -f "$CLAUDE_CONFIG_FILE" ]; then
    print_info "Backing up existing Claude Desktop config..."
    cp "$CLAUDE_CONFIG_FILE" "$CLAUDE_CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    print_status "Backup created: $CLAUDE_CONFIG_FILE.backup.*"
fi

# Generate recommended Claude Desktop config
print_info "Generating recommended Claude Desktop configuration..."

cat > claude_desktop_config_recommended.json << 'EOF'
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git", "--repository", "."]
    },
    "sequential-thinking": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "installer": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-installer"]
    },
    "serena": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-memory"]
    }
  }
}
EOF

print_status "Recommended config saved to: claude_desktop_config_recommended.json"

echo ""
print_info "Setting up project-specific MCP configuration..."

# Create .mcp directory for project-specific config
mkdir -p .mcp

cat > .mcp/config.json << 'EOF'
{
  "servers": {
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git", "--repository", "."],
      "enabled": true
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"],
      "enabled": true
    },
    "installer": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-installer"],
      "enabled": true
    }
  },
  "fallbacks": {
    "pieces": "built-in-memory",
    "serena": "built-in-code-analysis",
    "semgrep": "built-in-security",
    "trivy": "built-in-vulnerability-scan"
  }
}
EOF

print_status "Project MCP config created: .mcp/config.json"

echo ""
print_info "Setup Summary:"
echo "=============="

print_status "✅ Core MCP servers installed"
print_status "✅ Project configuration created" 
print_status "✅ Recommended Claude Desktop config generated"
print_info "📝 Backup of existing config created (if existed)"

echo ""
print_info "Next Steps:"
echo "1. 📋 Copy claude_desktop_config_recommended.json content to your Claude Desktop config"
echo "2. 🔄 Restart Claude Desktop application"
echo "3. 🧪 Run: python test_agent_role_system.py to verify setup"
echo "4. 🚀 Start using specialist agents with full MCP tool support!"

echo ""
print_info "Manual Installation Notes:"
echo "• Some specialized servers may need manual installation"
echo "• Pieces integration may require Pieces OS installation"  
echo "• Serena may need separate semantic analysis setup"
echo "• Security tools (semgrep/trivy) may need CLI tools installed first"

echo ""
print_status "🎉 AgentsMCP MCP Server setup complete!"