# AgentsMCP Orchestrator Models Guide

## Overview

AgentsMCP's distributed architecture allows you to choose the best orchestrator model for your specific needs. The orchestrator is responsible for task planning, decomposition, worker coordination, and cost optimization - making the model choice critical for both performance and cost efficiency.

## 🏆 Default: GPT-5

**GPT-5 is the recommended default** based on August 2025 benchmarks:
- **Performance**: 74.9% on SWE-bench (highest score)
- **Cost**: $1.25/$10 per M tokens (excellent value)
- **Context**: 400K input, 128K output tokens
- **Best for**: Most orchestration tasks with optimal cost-performance balance

## 📊 Available Models

### Commercial Models

| Model | Performance | Context | Cost (Input/Output) | Best For |
|-------|-------------|---------|---------------------|----------|
| **GPT-5** (DEFAULT) | 74.9% | 400K/128K | $1.25/$10/M | Best overall balance |
| Claude 4.1 Opus | 74.5% | 200K/8K | $15/$75/M | Premium orchestration |
| Claude 4.1 Sonnet | 72.7% | 200K/8K | $3/$15/M | Balanced performance |
| Gemini 2.5 Pro | 63.2% | 2M/8K | $1.25/$10/M* | Massive context needs |

*Gemini pricing increases to $2.50/$15/M for >200K tokens

### Local Models (FREE)

| Model | Performance | Context | Best For |
|-------|-------------|---------|----------|
| Qwen3-235B-A22B | ~70% | 1M/32K | Privacy, offline, zero cost |
| Qwen3-32B | ~65% | 128K/8K | Local balanced option |

## 🎯 Model Selection Guide

### By Use Case

```bash
# Default (recommended for most users)
agentsmcp interactive

# Premium quality (cost no object)
agentsmcp interactive --orchestrator-model claude-4.1-opus

# Massive codebases (>400K context needed)
agentsmcp dashboard --orchestrator-model gemini-2.5-pro

# Privacy/offline deployment
agentsmcp interactive --orchestrator-model qwen3-235b-a22b

# Budget-conscious but good performance
agentsmcp interactive --orchestrator-model claude-4.1-sonnet
```

### By Budget

| Budget Level | Recommended Model | Cost per Task* | Performance |
|--------------|-------------------|----------------|-------------|
| **High Performance** | GPT-5 | $0.113 | 74.9% |
| **Premium** | Claude 4.1 Opus | $1.125 | 74.5% |
| **Balanced** | Claude 4.1 Sonnet | $0.225 | 72.7% |
| **Massive Context** | Gemini 2.5 Pro | $0.113-$0.188 | 63.2% |
| **Zero Cost** | Qwen3-235B-A22B | $0.000 | ~70% |

*Based on 50K input + 5K output tokens per orchestration task

## 🛠️ Usage Examples

### CLI Commands

```bash
# Show all available models
agentsmcp models

# Show detailed model specs
agentsmcp models --detailed

# Get recommendation for specific use case
agentsmcp models --recommend premium
agentsmcp models --recommend cost_effective
agentsmcp models --recommend massive_context
agentsmcp models --recommend local

# Use specific model
agentsmcp interactive --orchestrator-model claude-4.1-opus
agentsmcp dashboard --orchestrator-model gemini-2.5-pro
```

### Programmatic Usage

```python
from agentsmcp.distributed.orchestrator import DistributedOrchestrator

# Initialize with specific model
orchestrator = DistributedOrchestrator(
    orchestrator_model="gpt-5",
    max_workers=20,
    context_budget_tokens=200000,
    cost_budget=50.0
)

# Get model recommendations
recommended = DistributedOrchestrator.get_model_recommendation("premium")
print(f"Recommended: {recommended}")

# Get available models
models = DistributedOrchestrator.get_available_models()
for name, config in models.items():
    print(f"{name}: {config['performance_score']}% performance")

# Estimate costs
cost = orchestrator.estimate_orchestration_cost(50000, 5000)
print(f"Estimated cost: ${cost:.4f}")
```

## 📈 Performance Comparison

Based on SWE-bench Verified benchmarks (August 2025):

```
GPT-5:           ████████████████████████████████████████ 74.9%
Claude 4.1 Opus: ███████████████████████████████████████  74.5%  
Claude 4.1 Sonnet: ████████████████████████████████████   72.7%
Qwen3-235B-A22B: ███████████████████████████████████      70.0% (est)
Qwen3-32B:       ██████████████████████████████████       65.0% (est)
Gemini 2.5 Pro:  ██████████████████████████████           63.2%
```

## 💡 Best Practices

### 1. **Start with GPT-5** (Default)
- Best overall performance-to-cost ratio
- Excellent for most orchestration tasks
- 400K context handles large planning scenarios

### 2. **Scale Based on Needs**
```bash
# Small projects, cost-conscious
--orchestrator-model claude-4.1-sonnet

# Large enterprise projects
--orchestrator-model claude-4.1-opus

# Cross-repository analysis
--orchestrator-model gemini-2.5-pro

# Privacy-sensitive deployments
--orchestrator-model qwen3-235b-a22b
```

### 3. **Monitor Costs**
```bash
# Check current costs
agentsmcp costs --breakdown

# Set budget alerts
agentsmcp budget 100.0
```

### 4. **Model-Specific Optimization**
- **GPT-5**: Leverage 400K context for comprehensive planning
- **Claude 4.1**: Excellent for quality-focused orchestration
- **Gemini 2.5 Pro**: Use for massive context scenarios (>400K tokens)
- **Qwen3**: Ideal for local/offline deployments

## 🚀 Migration Guide

### From Previous Versions
If upgrading from earlier AgentsMCP versions:

```python
# Old (monolithic)
agent_manager = AgentManager(model="gpt-4")

# New (distributed)
orchestrator = DistributedOrchestrator(
    orchestrator_model="gpt-5",  # Upgraded default
    max_workers=20
)
```

### Model Switching
Models can be changed at runtime by initializing a new orchestrator:

```python
# Switch from GPT-5 to Claude 4.1 Opus for premium tasks
premium_orchestrator = DistributedOrchestrator(
    orchestrator_model="claude-4.1-opus"
)
```

## 🔧 Advanced Configuration

### Context Budget Optimization

```python
# Auto-adjust context based on model limits
orchestrator = DistributedOrchestrator(
    orchestrator_model="gpt-5",
    context_budget_tokens=500000  # Will auto-adjust to 400K max
)
```

### Cost Estimation and Budgeting

```python
# Estimate before running
estimated_cost = orchestrator.estimate_orchestration_cost(100000, 10000)
if estimated_cost < budget_threshold:
    await orchestrator.execute_request(user_request)
```

## 📚 Related Documentation

- [Distributed Architecture Guide](DISTRIBUTED_ARCHITECTURE.md)
- [Cost Optimization Guide](COST_OPTIMIZATION.md)
- [Worker Models Guide](WORKER_MODELS.md)
- [Performance Benchmarks](BENCHMARKS.md)

---

**💡 Quick Start**: Just run `agentsmcp interactive` to use the optimized GPT-5 default, or run `agentsmcp models` to explore all options!