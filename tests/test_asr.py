from __future__ import annotations

from local_openai_mlx_provider import server


def test_audio_transcriptions_regression(client, monkeypatch):
    def fake_transcribe(
        *,
        audio_path,
        language,
        prompt,
        response_format,
        temperature,
    ):
        assert audio_path
        assert language == "en"
        assert prompt == "context"
        assert response_format == "json"
        assert temperature == 0.0
        return "hello world"

    monkeypatch.setattr(server.asr_runtime, "transcribe", fake_transcribe)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": server.settings.api_asr_model,
            "language": "en",
            "prompt": "context",
            "response_format": "json",
            "temperature": "0",
            "keep_alive": "0",
        },
        files={"file": ("speech.wav", b"not empty", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}
