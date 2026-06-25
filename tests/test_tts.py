from __future__ import annotations

from local_openai_mlx_provider import server


def test_audio_speech_regression(client, monkeypatch):
    audio = b"RIFF....WAVEfmt "

    def fake_speech(*, text, voice, response_format, speed):
        assert text == "hello"
        assert voice == "default"
        assert response_format == "wav"
        assert speed == 1.0
        return audio

    monkeypatch.setattr(server.tts_runtime, "speech", fake_speech)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": server.settings.api_tts_model,
            "input": "hello",
            "voice": "default",
            "response_format": "wav",
            "speed": 1.0,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == audio
