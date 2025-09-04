# MCP Servers Setup Guide

This guide explains how to set up Model Context Protocol (MCP) servers to enable full functionality for all specialist agents in AgentsMCP.

## Quick Setup

Run the automated setup script:

```bash
./setup_mcp_servers.sh
```

This will install all available MCP servers and create recommended configurations.

## Manual Setup

### Prerequisites

- **Node.js 18+** and npm
- **Python 3.8+** and pip  
- **Git** (for version control operations)

### Core MCP Servers (Always Available)

These servers are part of the official ModelContextProtocol and should install successfully:

```bash
# Install core MCP servers
npm install -g @modelcontextprotocol/server-sequential-thinking
npm install -g @modelcontextprotocol/server-git  
npm install -g @modelcontextprotocol/server-installer
npm install -g @modelcontextprotocol/server-memory
```

### Specialized MCP Servers (May Require Additional Setup)

#### 1. Pieces (Long-term Memory)
```bash
# Option 1: If available as MCP server
npm install -g @pieces-app/pieces-os-server-mcp

# Option 2: Install Pieces OS separately
# Download from: https://pieces.app/install
# Then configure pieces integration in AgentsMCP
```

#### 2. Serena (Semantic Code Analysis) 
```bash
# Check if available as MCP server
npm install -g serena-mcp-server

# Or use built-in semantic analysis capabilities
```

#### 3. Security Tools
```bash
# Semgrep MCP server (requires Semgrep CLI)
npm install -g semgrep  # Install CLI first
npm install -g @semgrep/mcp-server

# Trivy MCP server (requires Trivy CLI)
# Install Trivy: https://aquasecurity.github.io/trivy/
npm install -g @aquasec/trivy-mcp-server
```

#### 4. Language Servers
```bash
# TypeScript Language Server
npm install -g typescript @typescript-language-server/mcp-server

# Python Language Server  
pip install python-lsp-server pylsp-mcp-server
```

#### 5. Code Quality
```bash
# ESLint MCP server
npm install -g eslint @eslint/mcp-server
```

## Configuration

### Claude Desktop Configuration

Add to your `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
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
    "memory": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-memory"]
    },
    "installer": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-installer"]
    },
    "semgrep": {
      "command": "npx",
      "args": ["@semgrep/mcp-server"],
      "env": {
        "SEMGREP_APP_TOKEN": "your-semgrep-token-here"
      }
    },
    "trivy": {
      "command": "npx",
      "args": ["@aquasec/trivy-mcp-server"]
    }
  }
}
```

### AgentsMCP Project Configuration

The project includes these MCP configuration files:

- `mcp_servers_config.json` - Complete server definitions
- `.mcp/config.json` - Project-specific settings
- `setup_mcp_servers.sh` - Automated setup script

## Agent Tool Mapping

Each specialist agent category uses specific MCP tools:

| Agent Category | MCP Tools Used |
|----------------|----------------|
| **Executive** | pieces, sequential-thinking, web_search, git |
| **Engineering** | serena, pieces, sequential-thinking, git, lsp-ts, lsp-py |
| **Security** | semgrep, trivy, pieces, serena, sequential-thinking, git |
| **Data/Analytics** | pieces, sequential-thinking, web_search, git |
| **Design/UX** | pieces, sequential-thinking, web_search |
| **Marketing/Sales** | pieces, web_search, sequential-thinking |
| **Operations** | pieces, git, serena, trivy, sequential-thinking |

## Fallback Behavior

AgentsMCP gracefully handles missing MCP servers:

- **Built-in tools** (filesystem, bash, git, web_search) always work
- **Missing MCP tools** are skipped without errors
- **Agents adapt** their capabilities based on available tools
- **Core functionality** remains intact even with minimal setup

## Verification

Test your MCP server setup:

```bash
# Run the agent system test
python test_agent_role_system.py

# Check specific MCP server availability
npx @modelcontextprotocol/server-git --help
npx @modelcontextprotocol/server-sequential-thinking --help
```

## Troubleshooting

### Common Issues

1. **Node.js version too old**
   - Upgrade to Node.js 18+
   - Use `nvm` for version management

2. **Permission errors during npm install**
   - Use `npm install -g` with proper permissions
   - Or configure npm prefix: `npm config set prefix ~/.local`

3. **Python MCP servers not working**
   - Ensure Python 3.8+
   - Use virtual environment: `python -m venv mcp-env`

4. **MCP server not found**
   - Check if server exists: some may be theoretical examples
   - Verify installation: `npm list -g | grep mcp`

5. **Claude Desktop not recognizing servers**
   - Restart Claude Desktop application
   - Check config file syntax with JSON validator
   - Verify file permissions

### Debug Commands

```bash
# Check installed MCP servers
npm list -g | grep -i mcp

# Test server directly
npx @modelcontextprotocol/server-git --version

# Check Python packages
pip list | grep -i lsp
```

## Advanced Configuration

### Custom Server Configuration

Create `.mcp/servers/` directory for custom server definitions:

```bash
mkdir -p .mcp/servers
```

Add custom server configs in JSON format:

```json
{
  "name": "custom-server",
  "command": "path/to/server",
  "args": ["--custom-arg"],
  "env": {
    "CUSTOM_VAR": "value"
  }
}
```

### Environment-Specific Settings

Use environment variables for sensitive configuration:

```bash
export SEMGREP_APP_TOKEN="your-token"
export OPENAI_API_KEY="your-key"
export PIECES_API_KEY="your-key"
```

## MCP Server Development

To create custom MCP servers for AgentsMCP:

1. Follow MCP protocol specification
2. Implement required capabilities
3. Add to `mcp_servers_config.json`
4. Update agent tool mappings
5. Test with agent system

## Updates and Maintenance

Keep MCP servers updated:

```bash
# Update all global npm packages
npm update -g

# Update Python packages
pip install --upgrade python-lsp-server

# Re-run setup script
./setup_mcp_servers.sh
```

---

## Quick Reference

**✅ Always Available:** filesystem, bash, git, web_search  
**🔧 Requires Setup:** pieces, serena, sequential-thinking, semgrep, trivy, eslint, lsp-*  
**🚀 Setup Command:** `./setup_mcp_servers.sh`  
**🧪 Test Command:** `python test_agent_role_system.py`