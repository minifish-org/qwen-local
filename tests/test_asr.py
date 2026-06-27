from __future__ import annotations

import wave

from local_openai_mlx_provider import server


def _write_wav(path, samples, *, sample_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


def test_audio_transcriptions_regression(client, monkeypatch, tmp_path):
    normalized_audio = tmp_path / "speech.wav"
    normalized_audio.write_bytes(b"normalized wav")

    def fake_transcode(audio_path: str) -> str:
        assert audio_path.endswith(".wav")
        return str(normalized_audio)

    def fake_transcribe(
        *,
        audio_path,
        language,
        prompt,
        response_format,
        temperature,
    ):
        assert audio_path == str(normalized_audio)
        assert language == "en"
        assert prompt == "context"
        assert response_format == "json"
        assert temperature == 0.0
        return "hello world"

    monkeypatch.setattr(server, "_transcode_asr_audio_to_wav", fake_transcode)
    monkeypatch.setattr(server, "_validate_asr_audio_has_signal", lambda _: None, raising=False)
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


def test_audio_transcriptions_transcodes_browser_recordings(client, monkeypatch, tmp_path):
    normalized_audio = tmp_path / "recording.wav"
    normalized_audio.write_bytes(b"normalized wav")

    def fake_transcode(audio_path: str) -> str:
        assert audio_path.endswith(".webm")
        return str(normalized_audio)

    def fake_transcribe(
        *,
        audio_path,
        language,
        prompt,
        response_format,
        temperature,
    ):
        assert audio_path == str(normalized_audio)
        assert language is None
        assert prompt is None
        assert response_format == "json"
        assert temperature == 0.0
        return "browser recording"

    monkeypatch.setattr(server, "_transcode_asr_audio_to_wav", fake_transcode, raising=False)
    monkeypatch.setattr(server, "_validate_asr_audio_has_signal", lambda _: None, raising=False)
    monkeypatch.setattr(server.asr_runtime, "transcribe", fake_transcribe)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": server.settings.api_asr_model,
            "response_format": "json",
            "temperature": "0",
            "keep_alive": "0",
        },
        files={"file": ("recording.webm", b"not empty", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "browser recording"}


def test_audio_transcriptions_rejects_silent_audio(client, monkeypatch, tmp_path):
    silent_audio = tmp_path / "silent.wav"
    _write_wav(silent_audio, [0] * 16000)

    def fake_transcode(audio_path: str) -> str:
        assert audio_path.endswith(".wav")
        return str(silent_audio)

    def fail_transcribe(**_):
        raise AssertionError("silent audio should not be sent to ASR")

    monkeypatch.setattr(server, "_transcode_asr_audio_to_wav", fake_transcode)
    monkeypatch.setattr(server.asr_runtime, "transcribe", fail_transcribe)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": server.settings.api_asr_model,
            "language": "en",
            "response_format": "json",
            "temperature": "0",
            "keep_alive": "0",
        },
        files={"file": ("speech.wav", b"not empty", "audio/wav")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["param"] == "file"
    assert "silent or too quiet" in payload["error"]["message"]


def test_audio_transcriptions_accepts_quiet_audio_with_signal(
    client, monkeypatch, tmp_path
):
    quiet_audio = tmp_path / "quiet.wav"
    _write_wav(quiet_audio, [220, -220] * 8000)

    def fake_transcode(audio_path: str) -> str:
        assert audio_path.endswith(".wav")
        return str(quiet_audio)

    def fake_transcribe(
        *,
        audio_path,
        language,
        prompt,
        response_format,
        temperature,
    ):
        assert audio_path == str(quiet_audio)
        assert language == "en"
        assert prompt is None
        assert response_format == "json"
        assert temperature == 0.0
        return "quiet speech"

    monkeypatch.setattr(server, "_transcode_asr_audio_to_wav", fake_transcode)
    monkeypatch.setattr(server.asr_runtime, "transcribe", fake_transcribe)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": server.settings.api_asr_model,
            "language": "en",
            "response_format": "json",
            "temperature": "0",
            "keep_alive": "0",
        },
        files={"file": ("speech.wav", b"not empty", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "quiet speech"}
