# CI: Automerge & Branch Cleanup

Workflow: `.github/workflows/automerge.yml`.

## Purpose
- Allow maintainers to label a PR `automerge` and have it merge via squash automatically when checks pass; delete merged branches.

## Triggers
- `pull_request_target` with events: `labeled`, `closed`.

## Jobs

### enable-automerge
- If PR labeled `automerge`, grant automerge with squash.
- Acceptance: PR shows automerge enabled.

### delete-branch
- If PR closed and merged, delete source branch.
- Acceptance: Branch ref is removed.

## Safety
- Runs with `contents` and `pull-requests` write permissions.

