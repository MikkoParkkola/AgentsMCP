# Providers Guide

This guide covers all supported AI providers and how to configure them with AgentsMCP.

## Default Provider: ollama-turbo

**AgentsMCP ships with `ollama-turbo` as the default provider** using the `gpt-oss:120b` model. This provides:

- ✅ Local execution (no API keys required)  
- ✅ High performance for development tasks
- ✅ Cost-effective operation
- ✅ Privacy-focused (no external API calls)

### Installation

```bash
# Install Ollama if not present
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull gpt-oss:120b

# Verify installation
agentsmcp provider test ollama-turbo
```

## Alternative Providers

### OpenAI GPT

```bash
export OPENAI_API_KEY="your-api-key"
agentsmcp --provider openai agent spawn backend-engineer "Task description"
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-api-key"  
agentsmcp --provider anthropic agent spawn architect "Design system"
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
agentsmcp --provider azure agent spawn qa-engineer "Test plan"
```

### Google Gemini

```bash
export GOOGLE_API_KEY="your-api-key"
agentsmcp --provider google agent spawn data-scientist "Analysis task"
```

## Provider Configuration

### Environment Variables

Set default provider system-wide:

```bash
export AGENTSMCP_DEFAULT_PROVIDER=ollama-turbo
export AGENTSMCP_MODEL=gpt-oss:120b
```

### Configuration File

Create `~/.agentsmcp/config.yaml`:

```yaml
providers:
  default: ollama-turbo
  ollama-turbo:
    model: gpt-oss:120b
    base_url: http://localhost:11434
  openai:
    model: gpt-4
    temperature: 0.7
  anthropic:
    model: claude-3-sonnet-20240229
    max_tokens: 4096
```

## Provider Testing & Benchmarking

```bash
# Test provider connectivity
agentsmcp provider test ollama-turbo
agentsmcp provider test openai

# Benchmark performance across providers
agentsmcp provider benchmark --task "simple code generation"

# Compare cost and performance
agentsmcp provider compare --providers ollama-turbo,openai --iterations 10
```

## Production Recommendations

### High-Volume Development Teams
- **Primary**: `ollama-turbo` for cost efficiency
- **Fallback**: `openai` for complex reasoning tasks

### Enterprise Environments  
- **On-premises**: `ollama-turbo` with custom models
- **Hybrid**: `azure` for compliance + `ollama-turbo` for development

### Resource-Constrained Environments
- **Primary**: `ollama-turbo` with smaller models
- **Configuration**: Reduce concurrent agents, increase batch sizes

## Troubleshooting

### Common Issues

**Provider not responding:**
```bash
agentsmcp provider health-check ollama-turbo
```

**Model not found:**
```bash
ollama list
ollama pull gpt-oss:120b
```

**API key issues:**
```bash
agentsmcp provider validate openai
```

### Performance Optimization

**For ollama-turbo:**
- Ensure sufficient RAM (16GB+ recommended for gpt-oss:120b)
- Use SSD storage for model files
- Consider GPU acceleration if available

**For API providers:**
- Implement request batching
- Use rate limiting to avoid quotas
- Monitor usage and costs

## Security Considerations

### Local Providers (ollama-turbo)
- ✅ No data leaves your infrastructure
- ✅ Full control over model execution
- ⚠️ Ensure Ollama service is properly secured

### API Providers  
- ⚠️ Data sent to external services
- ⚠️ API keys must be secured
- ✅ Use environment variables, not hardcoded keys
- ✅ Implement proper key rotation

### Production Deployment
- Use secrets management (AWS Secrets Manager, etc.)
- Implement network policies for external API access
- Audit API usage and costs regularly