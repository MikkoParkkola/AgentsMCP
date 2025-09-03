# Task Completion Workflow

## When Task is Completed
After completing any coding task, follow this workflow:

### 1. Code Quality Checks
```bash
# Format and lint code
ruff check .
ruff format .

# Security scanning
bandit -r src/

# Dependency audit  
pip-audit
```

### 2. Testing
```bash
# Run relevant tests based on changes
pytest                                 # All tests
pytest -m "ui"                        # For UI changes
pytest -m "integration"               # For integration changes
pytest --cov=agentsmcp --cov-fail-under=80  # With coverage

# For TUI-specific changes
pytest -m "interactive"               # Interactive/TUI tests
```

### 3. Manual Testing (for TUI changes)
```bash
# Test TUI functionality
./agentsmcp tui                       # Main TUI interface
./agentsmcp tui-v2-dev                # Development TUI
./agentsmcp tui-v2-raw                # Minimal TUI test

# Verify core functionality
echo -e "help\nquit" | ./agentsmcp tui
echo -e "Hello world\nquit" | ./agentsmcp tui
```

### 4. Documentation
- Update docstrings for new/modified functions
- Update README.md if public API changes
- Add comments for complex logic
- Update type hints

### 5. Git Workflow
```bash
# Stage changes
git add .

# Commit with conventional commit format
git commit -m "feat: description of change"
git commit -m "fix: description of fix"  
git commit -m "test: description of test"

# Push changes
git push origin <branch-name>
```

## Definition of Done
- [ ] Code formatted and linted (ruff)
- [ ] Security scan passed (bandit) 
- [ ] All tests passing
- [ ] Coverage maintained (≥80%)
- [ ] Manual testing completed (for UI changes)
- [ ] Documentation updated
- [ ] Git commit with proper message
- [ ] No console errors or warnings