from __future__ import annotations

from local_openai_mlx_provider import server


def test_chat_completions_regression(client, monkeypatch):
    def fake_complete(messages, *, temperature, max_tokens):
        assert messages[0].content == "ping"
        assert temperature == 0.0
        assert max_tokens == 8
        return "pong"

    monkeypatch.setattr(server.llm_runtime, "complete", fake_complete)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": server.settings.api_llm_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == server.settings.api_llm_model
    assert payload["choices"][0]["message"]["content"] == "pong"
