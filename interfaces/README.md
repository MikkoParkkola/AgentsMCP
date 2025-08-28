ICDs (Interface Control Documents)
=================================

Source of truth for public module interfaces used by agents. Versioned via semver. Any changes to public fields must go through a CHANGE_REQUEST approved by the Architect.

Conventions
- Files: one JSON per module, named by dotted module id (dots can be kept or replaced with dashes).
- Required keys: name, purpose, inputs, outputs, errors, perf, security, version.
- Keep documents minimal and machine-readable.

Self-Improvement
- Agents may update these ICDs as they learn, but only via an Architect-approved CHANGE_REQUEST. When approved, bump version and include migration notes in docs/changelog.md.

