# CI: Release Pipeline

Workflow: `.github/workflows/release.yml`.

## Triggers
- Git tag push matching `v*.*.*`.

## Jobs

### build
- Steps: checkout, setup Python 3.12, `pip install build`, `python -m build`.
- Outputs: Uploads `dist/*` as artifact `dist`.
- Acceptance: sdist and wheel are created with correct version per tag.

### github-release
- Needs: `build`.
- Steps: download `dist` artifact; create GitHub Release; upload assets `dist/*`.
- Outputs: Published GitHub Release with attached artifacts and auto release notes.
- Acceptance: Release appears under Tags/Releases; assets downloadable.

### publish-pypi
- Needs: `build`; conditional on `secrets.PYPI_API_TOKEN` existing.
- Steps: setup Python; download `dist`; `pip install twine`; `twine upload dist/*`.
- Outputs: Package published to PyPI.
- Acceptance: PyPI version matches tag; upload succeeds.

## Artifacts
- `dist/*` (wheel + sdist) attached to the release and used for PyPI publish.

## Notes
- Consider adding binary artifact upload (PyInstaller) when single-file packaging is ready.

