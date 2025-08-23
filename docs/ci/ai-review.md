# CI: AI Review (Opt-in)

Workflow: `.github/workflows/ai-review.yml`.

## Purpose
- When `vars.AI_REVIEW_ENABLED == 'true'`, post a structured AI review request comment on PRs.

## Trigger
- `pull_request` events.

## Output
- A PR comment prompting reviewers to provide a structured verdict template.

## Acceptance Criteria
- Comment is created on PRs when the repo var is set to `'true'`.
- Does not run or post when var not set or false.

## Notes
- This job does not approve or block PRs; it only adds guidance.

