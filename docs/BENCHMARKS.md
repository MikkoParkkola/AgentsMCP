# Benchmark Reproducibility Guide

This guide provides comprehensive benchmarking tools and methodologies for measuring AgentsMCP performance across different configurations, providers, and workloads.

## Quick Start Benchmarking

### Basic Performance Test
```bash
# Run standard benchmark suite
agentsmcp benchmark run --suite standard

# Test specific provider performance
agentsmcp benchmark run --provider ollama-turbo --iterations 10

# Compare multiple providers
agentsmcp benchmark compare --providers ollama-turbo,openai,anthropic
```

### System Resource Benchmarking
```bash
# Memory and CPU usage during agent orchestration
agentsmcp benchmark resource --agents 5,10,20 --duration 60s

# Throughput testing with concurrent agents
agentsmcp benchmark throughput --max-agents 50 --task-type code-generation
```

## Benchmark Categories

### 1. Code Generation Benchmarks
Tests agent performance on common development tasks:

```bash
# Simple function generation
agentsmcp benchmark code-gen --task simple-function --iterations 20

# Complex class implementation  
agentsmcp benchmark code-gen --task complex-class --iterations 10

# Full module creation
agentsmcp benchmark code-gen --task full-module --iterations 5
```

**Expected Results (ollama-turbo baseline):**
- Simple function: ~2.3s average, 95% success rate
- Complex class: ~8.7s average, 87% success rate  
- Full module: ~15.2s average, 78% success rate

### 2. Multi-Agent Coordination Benchmarks
Tests orchestration efficiency:

```bash
# Team coordination latency
agentsmcp benchmark coordination --team-size 5 --task feature-development

# Message passing performance
agentsmcp benchmark messaging --agents 10 --messages 100

# Resource contention handling
agentsmcp benchmark contention --concurrent-tasks 20
```

### 3. Provider Performance Benchmarks
Standardized tests across different AI providers:

```bash
# Latency comparison
agentsmcp benchmark providers --metric latency --task standard-prompt

# Cost efficiency analysis
agentsmcp benchmark providers --metric cost-per-token --iterations 100

# Quality assessment (requires human evaluation dataset)
agentsmcp benchmark providers --metric quality --dataset human-eval
```

## Reproducible Test Environment

### Hardware Requirements
**Minimum for reliable benchmarks:**
- 16GB RAM (32GB recommended)
- 8-core CPU (16-core recommended) 
- SSD storage (NVMe preferred)
- Stable internet connection (for API providers)

### Environment Setup
```bash
# Create isolated benchmark environment
python -m venv benchmark-env
source benchmark-env/bin/activate
pip install agentsmcp[benchmark]

# Configure benchmark settings
cat > ~/.agentsmcp/benchmark-config.yaml << EOF
benchmark:
  iterations: 10
  timeout_seconds: 300
  resource_monitoring: true
  providers:
    - ollama-turbo
    - openai
  baseline_provider: ollama-turbo
EOF
```

### System Preparation
```bash
# Ensure consistent system state
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.vfs_cache_pressure=50

# Close unnecessary applications
# Disable background services that may affect performance
# Ensure stable CPU/GPU clocks (disable turbo/boost if needed)
```

## Benchmark Execution

### Standard Benchmark Suite
```bash
# Full benchmark suite (30-45 minutes)
agentsmcp benchmark run --suite comprehensive --output benchmark-results.json

# Quick smoke test (5-10 minutes)
agentsmcp benchmark run --suite smoke --output smoke-test.json

# Provider comparison (15-20 minutes per provider)
agentsmcp benchmark run --suite provider-comparison
```

### Custom Benchmark Configuration
```yaml
# benchmark-config.yaml
benchmarks:
  code_generation:
    tasks:
      - simple_function: { iterations: 20, timeout: 60 }
      - api_endpoint: { iterations: 15, timeout: 120 }
      - data_model: { iterations: 10, timeout: 90 }
    
  coordination:
    team_sizes: [2, 5, 10, 15]
    task_complexity: [simple, medium, complex]
    iterations: 5
    
  resource_usage:
    max_agents: [1, 5, 10, 20, 50]
    duration_seconds: 60
    metrics: [memory, cpu, network, disk]
```

## Results Analysis

### Performance Metrics
```bash
# Generate performance report
agentsmcp benchmark analyze --input benchmark-results.json --output report.html

# Compare against baseline
agentsmcp benchmark compare --baseline baseline.json --current current.json

# Trend analysis across multiple runs
agentsmcp benchmark trend --files "results-*.json" --output trend-analysis.png
```

### Key Performance Indicators (KPIs)
1. **Task Completion Rate**: % of tasks completed successfully
2. **Average Latency**: Mean time per task completion
3. **P95 Latency**: 95th percentile task completion time
4. **Throughput**: Tasks completed per second
5. **Resource Efficiency**: Memory/CPU usage per task
6. **Cost Efficiency**: Cost per successful task completion

### Benchmark Results Interpretation
```python
# Example analysis script
import json
from agentsmcp.benchmark import BenchmarkAnalyzer

# Load results
analyzer = BenchmarkAnalyzer('benchmark-results.json')

# Performance regression detection
regression_threshold = 0.15  # 15% regression threshold
if analyzer.detect_regression(threshold=regression_threshold):
    print("⚠️  Performance regression detected!")
    analyzer.print_regression_details()

# Resource efficiency analysis
efficiency = analyzer.resource_efficiency()
print(f"Memory efficiency: {efficiency['memory']:.2f} MB/task")
print(f"CPU efficiency: {efficiency['cpu']:.2f}% utilization")
```

## CI/CD Integration

### GitHub Actions Benchmark
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks
on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[benchmark]
          
      - name: Run benchmarks
        run: |
          agentsmcp benchmark run --suite ci --output results.json
          
      - name: Analyze results
        run: |
          agentsmcp benchmark analyze --input results.json --format github-comment > comment.md
          
      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const comment = fs.readFileSync('comment.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

### Performance Regression Gates
```bash
# Set up performance gates in CI
agentsmcp benchmark gate --baseline main --current HEAD --threshold 15%

# Fail CI if performance drops more than threshold
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "❌ Performance regression detected - blocking merge"
  exit 1
fi
```

## Advanced Benchmarking

### Load Testing
```bash
# Simulate production load
agentsmcp benchmark load --concurrent-users 100 --duration 300s --ramp-up 60s

# Stress testing to find breaking points
agentsmcp benchmark stress --max-agents 1000 --increment 10 --duration 60s
```

### Memory Profiling
```bash
# Profile memory usage during benchmarks
agentsmcp benchmark profile --memory --output memory-profile.json

# Analyze memory leaks
agentsmcp benchmark analyze-memory --profile memory-profile.json
```

### Custom Benchmark Development
```python
# custom_benchmark.py
from agentsmcp.benchmark import BaseBenchmark

class CustomCodeGenBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("custom_code_gen")
    
    def setup(self):
        # Initialize test environment
        pass
    
    def run_iteration(self, iteration: int):
        # Execute single benchmark iteration
        start_time = time.time()
        
        result = self.run_agent_task(
            agent_type="backend-engineer",
            task="Generate REST API for user management"
        )
        
        end_time = time.time()
        
        return {
            'duration': end_time - start_time,
            'success': result.success,
            'quality_score': self.evaluate_quality(result)
        }
    
    def cleanup(self):
        # Clean up test environment
        pass

# Register and run custom benchmark
benchmark = CustomCodeGenBenchmark()
results = benchmark.run(iterations=10)
```

## Troubleshooting

### Common Issues

**Inconsistent results across runs:**
- Ensure system is in consistent state
- Close background applications
- Use fixed CPU/GPU clocks
- Run multiple iterations and average results

**High variance in measurements:**
- Increase iteration count
- Use longer warmup periods
- Check for resource contention
- Monitor system load during benchmarks

**Provider timeout issues:**
- Increase timeout values for slower providers
- Check network connectivity
- Monitor API rate limits
- Use exponential backoff for retries

### Debug Mode
```bash
# Run benchmarks with detailed logging
agentsmcp benchmark run --debug --log-level debug --output-logs benchmark.log

# Profile benchmark execution
agentsmcp benchmark run --profile --output-profile profile.json
```

## Best Practices

1. **Consistent Environment**: Always benchmark in the same environment
2. **Baseline Establishment**: Establish performance baselines for each release
3. **Regular Monitoring**: Run benchmarks on every significant change
4. **Multiple Iterations**: Always run multiple iterations and report statistics
5. **Resource Monitoring**: Track system resources during benchmark execution
6. **Documentation**: Document any environment changes that might affect results