# Open Issues Log

Track significant open questions, decision needs, and risks. Move entries to `decision-log.md` once resolved.

<!--
Template:
## [ID] - Title
- **Date Added:** YYYY-MM-DD
- **Version:** vX.Y.Z
- **Branch:** https://example.com/branch
- **Submitter:** name
- **Decision Maker:** name
- **Needed By:** YYYY-MM-DD
- **Status:** open
- **Problem:** description
- **Options:**
  - Option A - pros / cons
  - Option B - pros / cons
-->
## [0002] - Define CODEOWNERS and maintainer list
- **Date Added:** 2025-08-21
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-01
- **Status:** open
- **Problem:** Placeholder CODEOWNERS uses @REPO_OWNER; real maintainers need to be specified.
- **Options:**
  - Provide GitHub usernames of maintainers.
  - Use a GitHub team handle once defined.

## [0008] - Implement Architecture Synthesis Phase 1
- **Date Added:** 2025-08-26
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-05
- **Status:** open
- **Problem:** Core correctness and lifecycle issues (API uses multiple AgentManager instances; cleanup bug; duplicate EventBus) and lack of base metrics.
- **Options:**
  - Implement Phase 1 as proposed: single AgentManager in API, fix cleanup time, consolidate EventBus, add metrics.
  - Defer EventBus consolidation and only patch API and cleanup bug (faster, less robust).
