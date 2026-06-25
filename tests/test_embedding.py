from __future__ import annotations

from local_openai_mlx_provider import server


def test_embeddings_regression(client, monkeypatch):
    def fake_embed(inputs):
        assert inputs == ["alpha", "beta"]
        return [[0.1, 0.2], [0.3, 0.4]]

    monkeypatch.setattr(server.embedding_runtime, "embed", fake_embed)

    response = client.post(
        "/v1/embeddings",
        json={
            "model": server.settings.api_embedding_model,
            "input": ["alpha", "beta"],
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == server.settings.api_embedding_model
    assert payload["data"][0]["embedding"] == [0.1, 0.2]
    assert payload["data"][1]["embedding"] == [0.3, 0.4]
