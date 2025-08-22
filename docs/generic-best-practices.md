# Generic Project Best Practices

These practices apply to any software project, with or without AI agents.

## Planning and Scope
- Maintain a clear product scope and avoid fake or unimplemented features.
- Document setup and workflows so new contributors can onboard quickly.

## Coding Conventions
- Standardize the tech stack, naming, and module organization across the codebase.
- Commit lockfiles and use deterministic installs for reproducible builds.

## Testing and Quality
- Follow Test-Driven Development; require unit, integration, and end-to-end tests.
- Lint, type-check, and audit dependencies before merging.

## Continuous Integration and Deployment
- Cache dependencies and parallelize CI jobs while collecting coverage artifacts.
- Centralize deployment coordination with status tracking and rollback plans.

## Pull Request Workflow
- Keep changes small and use Conventional Commits.
- Use a consistent PR template capturing change summary, testing evidence, and documentation updates.

## Versioning and Releases
- Bump semantic versions and update the changelog for functional changes.
- Automate release tagging and changelog generation.

## Security and UX
- Run security reviews and dependency audits.
- Provide global error boundaries and ensure accessibility standards are met.
