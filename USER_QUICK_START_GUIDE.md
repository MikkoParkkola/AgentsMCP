# 🚀 AgentsMCP Quick Start Guide

*Get up and running with AI agents in 2 minutes*

---

## What is AgentsMCP?

AgentsMCP is your personal AI assistant that can help with coding, writing, analysis, and more. Think of it as having Claude, GPT, and other AI models working together on your tasks.

---

## 📦 Installation (Pick One)

### Option 1: macOS Binary (Easiest)
1. Download the latest release from GitHub
2. Run in terminal:
   ```bash
   ./agentsmcp
   ```
   That's it!

### Option 2: Python Install
```bash
pip install -e ".[dev,rag]"
agentsmcp
```

### Option 3: Docker
```bash
docker run -p 8000:8000 agentsmcp:latest
```

---

## 🎯 First Steps

### 1. Launch AgentsMCP
```bash
agentsmcp
```

**What you'll see:**
```
🤖 Welcome to AgentsMCP! Let me help you get started.

I can help with:
• Code review and debugging
• Writing and editing
• Data analysis
• Research and summaries

What would you like to work on today?
```

### 2. Try Your First Task
Just type what you want in plain English:

```
💬 You: Help me organize the files in this directory
```

```
💬 You: Review this code for bugs: [paste your code]
```

```
💬 You: Summarize this document for me
```

---

## 🔧 Common Tasks

### Code Help
```
"Review this Python script for errors"
"Add comments to this function" 
"Create a test for this code"
"Explain what this code does"
```

### Writing & Editing  
```
"Improve this email"
"Write a README for my project"
"Fix the grammar in this document"
"Make this more professional"
```

### Analysis & Research
```
"Analyze this data file"
"Summarize the main points"
"Find patterns in these numbers"
"Research X topic for me"
```

---

## ⚡ Pro Tips

### Multi-line Input
- Paste large code blocks or documents directly
- AgentsMCP handles multi-line content automatically

### File Operations
- Drag and drop files into the terminal
- Or use: `"analyze the file at /path/to/file.txt"`

### Switch AI Models
```
💬 You: Use Claude for this task
💬 You: Switch to the local model
💬 You: What models are available?
```

---

## 🆘 Getting Help

### Built-in Help
```
💬 You: help
💬 You: what can you do?
💬 You: show examples
```

### Common Issues

**Problem**: "Command not found"
**Solution**: Make sure you're in the right directory or AgentsMCP is in your PATH

**Problem**: "No AI models available"  
**Solution**: AgentsMCP will guide you through connecting to AI services

**Problem**: "Multi-line paste not working"
**Solution**: Try typing directly instead of pasting, or restart AgentsMCP

---

## 🎯 What's Next?

Once you're comfortable with basic tasks:

1. **Explore Advanced Features**
   ```bash
   agentsmcp --help
   ```

2. **Set Up Cloud AI Models** (optional)
   - Add OpenAI API key for GPT models
   - Connect to Claude API for advanced capabilities

3. **Try Team Features**
   - Share configurations with teammates
   - Set up automated workflows

---

## 🔗 Need More Help?

- **Documentation**: Check the full README.md
- **Issues**: Report bugs on GitHub
- **Community**: Join discussions and get help

---

## 🎉 You're Ready!

AgentsMCP is designed to learn your preferences and get better over time. The more you use it, the more helpful it becomes.

**Start simple. Ask for help. Explore gradually.**

Happy building! 🚀