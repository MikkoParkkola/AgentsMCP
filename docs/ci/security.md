# CI: Security Scans

Workflows: `.github/workflows/security.yml`, `.github/workflows/codeql-python.yml`, `.github/workflows/semgrep.yml`, `.github/workflows/gitleaks.yml`, `.github/workflows/sbom.yml`.

## Purpose
- Continuous security posture checks: static analysis, dependency audit, secrets detection, code scanning, SBOM generation.

## Jobs & Steps

### Bandit
- Input: Python 3.11
- Steps: `bandit -r src -q` (or `-x tests` in CI variant)
- Output: Job status, annotations for issues.
- Acceptance: No HIGH severity findings (MED/LOW allowed per policy).

### pip-audit
- Steps: install project and `pip-audit -v`.
- Output: Vulnerability list.
- Acceptance: No known vulnerabilities with fixed versions available. Allowlist via `pip-audit.toml` (future) if needed.

### CodeQL (Python)
- Steps: `codeql-action/init` + `analyze`.
- Output: SARIF uploaded to Security tab (upload disabled in current config; acceptable).
- Acceptance: Workflow completes; triage occurs via GitHub Code Scanning (when enabled).

### Semgrep
- Steps: `returntocorp/semgrep-action@v1` with `config: p/ci`.
- Output: Findings summary in job logs.
- Acceptance: No HIGH severity true positives; otherwise PR must address or suppress with justification.

### Gitleaks
- Steps: scan repo for secrets (full history if needed).
- Output: Findings summary.
- Acceptance: No secrets detected. False positives must be ignored via config (future) or rewritten.

### SBOM (CycloneDX)
- Steps: `npx -y @cyclonedx/cdxgen -o sbom.json -r .` and upload artifact.
- Output: `sbom.json` artifact.
- Acceptance: Artifact present and valid JSON; size > 0.

## Artifacts
- `sbom.json`
- (Optional) SARIF uploads for Semgrep/CodeQL (future enhancement)

## Notes
- Security jobs run on push, PRs, and on schedule (weekly) where configured.

