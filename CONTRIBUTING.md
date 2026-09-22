# Contributing

This repository is archived and no longer maintained. Fork it for new development; no fixes or pull-request reviews are promised.

## Local checks

See README.md for prerequisites. Run from the repository root:

```sh
python -m pip install fastapi pydantic python-multipart httpx pytest numpy
python -m pytest tests -q
```

Keep changes focused, describe the behavior and validation in your pull request,
and add regression coverage for behavior changes. Never include credentials,
private deployment settings, user data, model weights, or generated build output.

Contributions must be your own work or include compatible upstream permission and
attribution. Project contributions use the license described in LICENSE and the
README; third-party material retains its original license.

Be respectful and constructive. Harassment and disclosure of private information
are not welcome. Maintainers may moderate discussions and decline contributions.
