# CI: Container Build Sanity

Workflow: `.github/workflows/python-ci.yml` (job `docker-build`).

## Purpose
- Ensure Dockerfile builds successfully on CI and tags a local image.

## Steps (Reference)
- Checkout code.
- Use `docker/build-push-action@v6` with `push: false` and tag `agentsmcp:ci`.

## Outputs
- Job status (pass/fail).
- Build logs.

## Acceptance Criteria
- Image builds without errors.
- No push to registry is attempted in CI (push disabled).

## Notes
- Future: add Hadolint scan and run a smoke command inside the image.

