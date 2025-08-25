# AgentsMCP Realigned Roadmap - Phase 0 Foundation

## Executive Summary
Prioritizing stable, production-ready foundation before advanced AI features. This realignment focuses on delivering core MCP server functionality within 6-8 weeks, establishing the infrastructure needed for future enhancements.

## Phase 0: Core Foundation (Weeks 1-4)
**Goal:** Stable MCP server with provider management, CLI, and basic orchestration

### Week 1: Provider Infrastructure
- [x] B1: Providers module skeleton (types, interfaces)
- [ ] B2: OpenAI list_models adapter
- [ ] B3: OpenRouter list_models adapter  
- [ ] B4: Ollama list_models adapter
- [ ] B5: Facade list_models router
- [ ] B6: Agent hook discover_models

### Week 2: CLI & Configuration
- [ ] C1: Command plumbing /models
- [ ] C2: Model list UI (search + select)
- [ ] C3: Provider autocomplete
- [ ] C4: Apply selection to runtime
- [ ] K1: Validation helpers
- [ ] K2: Prompt + persist API keys
- [ ] K3: Wire validation

### Week 3: MCP Gateway & Safety
- [ ] M1: negotiate_version()
- [ ] M2: downconvert_tools()
- [ ] M3: Wire negotiation
- [ ] **NEW**: Basic sandboxing layer
- [ ] **NEW**: Orchestration message bus
- [ ] **NEW**: Safety guardrails

### Week 4: Web Interface & Observability
- [ ] WUI1: SSE event bus
- [ ] WUI2: REST control & status
- [ ] WUI3: Web UI scaffold
- [ ] **NEW**: Structured logging
- [ ] **NEW**: Prometheus metrics
- [ ] **NEW**: Health checks

## Phase 1: Production Hardening (Weeks 5-6)
**Goal:** Security, performance, and reliability improvements

### Week 5: Security & Authentication
- [ ] JWT/OAuth2 authentication
- [ ] Rate limiting middleware
- [ ] CORS configuration hardening
- [ ] Secret management integration
- [ ] Container security (non-root user)

### Week 6: Performance & Testing
- [ ] X1: Token estimation + trim
- [ ] X2: Context management
- [ ] S1-S3: Streaming implementation
- [ ] Integration test suite
- [ ] Performance benchmarks
- [ ] E2E smoke tests

## Phase 2: Agent Coordination (Weeks 7-8)
**Goal:** Multi-agent discovery and basic coordination

### Week 7: Discovery & Registration
- [ ] AD1: Discovery protocol spec
- [ ] AD2: Announcer/registry daemon
- [ ] AD3: Discovery client
- [ ] AD4: Coordination handshake
- [ ] AD5: Security & config

### Week 8: Enhanced Orchestration
- [ ] Event-driven message bus
- [ ] Agent lifecycle management
- [ ] Context persistence (Redis/PostgreSQL)
- [ ] Basic workflow templates
- [ ] Monitoring dashboard

## Future Research Track (Months 3-6)
**Deferred for research spikes after stable foundation**

- LangGraph memory patterns
- AutoGen sandboxed execution
- CrewAI role specialization
- Emotional intelligence exploration
- Consciousness monitoring research
- Symphony mode coordination
- DNA evolution experiments

## Success Metrics

### Phase 0 Completion
- ✅ All provider adapters functional
- ✅ CLI fully operational
- ✅ Web UI with live updates
- ✅ 100% of backlog items B1-B6, C1-C4, K1-K3 complete

### Phase 1 Completion  
- ✅ JWT authentication implemented
- ✅ <100ms response time for all endpoints
- ✅ 80% test coverage
- ✅ Zero critical security issues

### Phase 2 Completion
- ✅ Multi-agent discovery working
- ✅ Event-driven orchestration operational
- ✅ Context persistence verified
- ✅ Production deployment ready

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Scope creep | Strict phase gates, defer advanced features |
| Integration complexity | Test each component in isolation first |
| Performance issues | Benchmark early and often |
| Security vulnerabilities | Security review at each phase gate |

## Team Allocation
- **Core Development:** 2-3 engineers
- **Testing/QA:** 1 engineer  
- **DevOps/Security:** 1 engineer
- **Research Track:** 1-2 engineers (after Phase 1)

This realigned roadmap provides a realistic path to production while preserving the vision for advanced features in a research track.