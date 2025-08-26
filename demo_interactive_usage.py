#!/usr/bin/env python3
"""
Demo script showing the new AgentsMCP interactive usage with / commands
"""

def show_usage_examples():
    """Show example usage of the new command system"""
    print("🎯 AgentsMCP Interactive Mode - New Command System")
    print("=" * 60)
    print()
    
    print("📋 COMMAND SYNTAX:")
    print("  • Commands must start with '/' character")
    print("  • Everything else is treated as conversation with AI agents")
    print()
    
    print("🔧 AVAILABLE COMMANDS:")
    command_examples = [
        ("/help", "Show all available commands and usage"),
        ("/agents", "List and manage AI agents"),
        ("/status", "Show system status and health"),
        ("/execute <task>", "Execute a task using orchestration"),
        ("/theme <name>", "Change UI theme (auto/light/dark)"),
        ("/history", "Show command history"),
        ("/config", "Manage configuration settings"),
        ("/settings", "Open interactive settings dialog"),
        ("/clear", "Clear terminal screen"),
        ("/exit", "Exit the application"),
    ]
    
    for cmd, desc in command_examples:
        print(f"  {cmd:20} - {desc}")
    
    print()
    print("💬 CONVERSATION EXAMPLES:")
    conversation_examples = [
        "Write a Python function to calculate fibonacci numbers",
        "Help me debug this JavaScript code",
        "Create a REST API using FastAPI",
        "Explain how async/await works in Python",
        "Can you review this code for security issues?",
        "Generate unit tests for my function",
    ]
    
    for i, conv in enumerate(conversation_examples, 1):
        print(f"  {i}. {conv}")
    
    print()
    print("🚀 USAGE WORKFLOW:")
    print("  1. Start: PYTHONPATH=src python -m agentsmcp --mode interactive")
    print("  2. Use /commands for system control (agents, config, help)")
    print("  3. Use natural language for coding tasks and AI assistance")
    print("  4. Switch between commands and conversation seamlessly")
    
    print()
    print("✨ EXAMPLE SESSION:")
    session_example = [
        ("Command", "/help", "Get familiar with available commands"),
        ("Command", "/agents", "See what AI agents are available"),
        ("Conversation", "Create a simple web scraper in Python", "AI will help with code"),
        ("Conversation", "Add error handling and logging", "Continue the conversation"),
        ("Command", "/settings", "Adjust configuration if needed"),
        ("Command", "/exit", "End the session"),
    ]
    
    for typ, input_text, desc in session_example:
        print(f"  {typ:12} | {input_text:40} | {desc}")
    
    print()
    print("=" * 60)
    print("🎉 Ready to code with AI agents!")

if __name__ == "__main__":
    show_usage_examples()