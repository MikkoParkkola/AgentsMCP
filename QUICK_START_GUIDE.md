# 🚀 AgentsMCP Quick Start Guide

*Get started with AgentsMCP in under 3 minutes*

---

## What is AgentsMCP?

AgentsMCP is a multi-agent orchestration platform that lets you chat with AI agents and automate complex tasks. Think of it as a powerful AI assistant that can use multiple tools and agents to solve problems.

---

## ⚡ 30-Second Quick Start

### 1. Install & Run
```bash
# Install AgentsMCP (assuming it's installed)
agentsmcp interactive --no-welcome
```

### 2. Start Chatting
```
🎼 agentsmcp ▶ hello world

🤖 AgentsMCP Response:
Hello! I'm AgentsMCP, ready to help you with tasks, automation, and multi-agent orchestration.
```

### 3. Try a Simple Task
```
🎼 agentsmcp ▶ list files in current directory

# AgentsMCP will use its tools to show you the files
```

**That's it!** You're now using AgentsMCP.

---

## 📋 What You Can Do

### Basic Chat
- Ask questions: `What's the weather like?`
- Get explanations: `Explain how machine learning works`
- Request summaries: `Summarize this text: [your text]`

### File Operations  
- `list files in /path/to/directory`
- `read the contents of file.txt`
- `search for files containing "keyword"`

### Automation Tasks
- `create a Python script that does X`
- `analyze this data and create a report`  
- `help me organize these files`

### Multi-Agent Workflows
- `use multiple agents to research topic X and create a summary`
- `coordinate between different tools to solve problem Y`

---

## 🎯 Common Use Cases

### For Developers
```
🎼 agentsmcp ▶ analyze my Python project and suggest improvements
🎼 agentsmcp ▶ create unit tests for this function
🎼 agentsmcp ▶ help me debug this error message
```

### For Content Creators
```
🎼 agentsmcp ▶ help me brainstorm blog post ideas about AI
🎼 agentsmcp ▶ proofread and improve this article
🎼 agentsmcp ▶ create social media posts from this content
```

### For Data Analysis
```
🎼 agentsmcp ▶ analyze this CSV file and create visualizations
🎼 agentsmcp ▶ find patterns in this data
🎼 agentsmcp ▶ create a dashboard for these metrics
```

### For Research
```
🎼 agentsmcp ▶ research the latest developments in quantum computing
🎼 agentsmcp ▶ summarize these research papers
🎼 agentsmcp ▶ create a comparison table of different approaches
```

---

## 🎮 Interface Modes

### Interactive Chat (Recommended for Beginners)
```bash
agentsmcp interactive --no-welcome
```
- Simple chat interface
- Type questions and get answers
- Perfect for getting started

### Dashboard Mode (For Monitoring)
```bash
agentsmcp --mode dashboard
```
- Visual dashboard with real-time stats
- Monitor agent performance
- Good for power users

### Web Interface (Coming Soon)
```bash
agentsmcp --mode web
# Opens browser interface at http://localhost:8000
```

---

## ⚙️ Basic Configuration

### Set Up API Keys (If Needed)
Some features require API keys for external services:

```bash
# For OpenAI (if using GPT models)
agentsmcp config set openai-api-key YOUR_KEY_HERE

# For other providers
agentsmcp config set anthropic-api-key YOUR_KEY_HERE
```

### Choose Your Model
```bash
# List available models
agentsmcp models list

# Set default model
agentsmcp config set default-model gpt-4
```

---

## 💡 Tips for Success

### 1. Be Specific
❌ **Vague**: "Help me with code"  
✅ **Clear**: "Review this Python function and suggest performance improvements"

### 2. Use Natural Language
❌ **Robotic**: "Execute file listing command on directory /home/user"  
✅ **Natural**: "Show me what files are in my home directory"

### 3. Break Down Complex Tasks
❌ **Overwhelming**: "Build me a complete e-commerce website with payment processing, user accounts, and inventory management"  
✅ **Manageable**: "Help me create the basic structure for an e-commerce website, starting with the product catalog"

### 4. Provide Context
❌ **No Context**: "Fix this error"  
✅ **With Context**: "I'm getting this Python error when running my web scraper: [error message]"

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### "Command not found: agentsmcp"
**Problem**: AgentsMCP not installed or not in PATH  
**Solution**: Check installation or use full path to executable

#### "Provider authentication failed"
**Problem**: Missing or invalid API keys  
**Solution**: Set up API keys using `agentsmcp config set [provider]-api-key [key]`

#### "No response from agent"
**Problem**: Agent or model not available  
**Solution**: Check model availability with `agentsmcp models list`

#### Multi-line input not working
**Problem**: Paste doesn't work correctly  
**Solution**: Type line by line or try a different terminal

#### Slow responses
**Problem**: First response takes 5+ seconds  
**Solution**: This is normal - subsequent responses are faster

---

## 📚 Next Steps

### Learn More
- **Advanced Features**: Explore orchestration and multi-agent workflows
- **Customization**: Learn about custom agents and plugins  
- **Integration**: Connect AgentsMCP to your existing tools

### Get Help
- **Documentation**: Check the full docs in the project repository
- **Examples**: Look for example scripts and use cases
- **Community**: Join discussions and ask questions

### Power User Features
Once comfortable with basics, explore:
- Custom agent creation
- Workflow automation
- API integration
- Team collaboration features

---

## ⭐ Success Stories

*"AgentsMCP helped me automate my entire data analysis pipeline - what used to take hours now takes minutes."* - Data Scientist

*"I use it to coordinate multiple AI agents for content creation. Game changer for productivity."* - Content Creator  

*"The multi-agent orchestration is perfect for complex development tasks."* - Software Developer

---

## 🎯 You're Ready!

You now know enough to start using AgentsMCP effectively. Remember:

1. **Start simple** - Basic chat and file operations
2. **Experiment** - Try different types of requests  
3. **Be patient** - First responses may be slower
4. **Ask for help** - The system can explain its own capabilities

**Happy orchestrating!** 🚀

---

*Having trouble? The system itself can help - just ask: "How do I use AgentsMCP?" or "What can you help me with?"*