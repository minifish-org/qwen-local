# Public release preparation — 2026-09-22

## Validation

`python -m pytest tests -q`: 40 passed in an isolated Python 3.12 environment with mocked model backends. Portable launchd template rendering was verified without installing a service.

Gitleaks found no secrets in the fetched local Git history at preparation time.
This is a best-effort check, not a guarantee that every possible secret is detected.

## Scope and limitations

Archived reference implementation. No model downloads, real inference or launchd service activation were performed. Runtime dependency compatibility is not maintained.

See README.md for license, setup and project status; SECURITY.md describes support
and private reporting. Third-party material retains its upstream license.
