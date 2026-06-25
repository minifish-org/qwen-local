from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_openai_mlx_provider import server


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app, raise_server_exceptions=False)
