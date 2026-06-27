from __future__ import annotations

from local_openai_mlx_provider import server
from local_openai_mlx_provider import tts as tts_module


def test_audio_speech_regression(client, monkeypatch):
    audio = b"RIFF....WAVEfmt "

    def fake_speech(*, model_id, model_kind, text, voice, instruct, response_format, speed):
        assert model_id == server.settings.tts_model
        assert model_kind == "custom_voice"
        assert text == "hello"
        assert voice == "default"
        assert instruct is None
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


def test_audio_speech_voice_design_requires_instruct(client, monkeypatch):
    def fail_speech(**_):
        raise AssertionError("voice design should be rejected before TTS runtime")

    monkeypatch.setattr(server.tts_runtime, "speech", fail_speech)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": server.settings.api_tts_voice_design_model,
            "input": "hello",
            "voice": "vivian",
            "response_format": "wav",
            "speed": 1.0,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "instruct"


def test_audio_speech_voice_design_routes_with_instruct(client, monkeypatch):
    audio = b"RIFF....WAVEfmt "

    def fake_speech(*, model_id, model_kind, text, voice, instruct, response_format, speed):
        assert model_id == server.settings.tts_voice_design_model
        assert model_kind == "voice_design"
        assert text == "hello"
        assert voice == "ignored"
        assert instruct == "natural warm conversational voice"
        assert response_format == "wav"
        assert speed == 0.9
        return audio

    monkeypatch.setattr(server.tts_runtime, "speech", fake_speech)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": server.settings.api_tts_voice_design_model,
            "input": "hello",
            "voice": "ignored",
            "instruct": " natural warm conversational voice ",
            "response_format": "wav",
            "speed": 0.9,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == audio


def test_audio_speech_accepts_underlying_tts_model_ids(client, monkeypatch):
    calls = []

    def fake_speech(*, model_id, model_kind, text, voice, instruct, response_format, speed):
        calls.append((model_id, model_kind, instruct))
        return b"RIFF....WAVEfmt "

    monkeypatch.setattr(server.tts_runtime, "speech", fake_speech)

    custom_response = client.post(
        "/v1/audio/speech",
        json={
            "model": server.settings.tts_model,
            "input": "hello",
            "response_format": "wav",
            "keep_alive": 0,
        },
    )
    voice_design_response = client.post(
        "/v1/audio/speech",
        json={
            "model": server.settings.tts_voice_design_model,
            "input": "hello",
            "instruct": "natural voice",
            "response_format": "wav",
            "keep_alive": 0,
        },
    )

    assert custom_response.status_code == 200
    assert voice_design_response.status_code == 200
    assert calls == [
        (server.settings.tts_model, "custom_voice", None),
        (server.settings.tts_voice_design_model, "voice_design", "natural voice"),
    ]


def test_models_lists_custom_voice_and_voice_design_aliases(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {model["id"] for model in response.json()["data"]}
    assert server.settings.api_tts_model in model_ids
    assert server.settings.api_tts_voice_design_model in model_ids


def test_tts_provider_switches_model_before_loading_next_backend(monkeypatch):
    releases = []

    class FakeBackend:
        def __init__(self, settings, *, model_id, model_kind):
            self.model_id = model_id
            self.model_kind = model_kind

        def speech(self, **kwargs):
            return f"{self.model_id}:{self.model_kind}".encode()

    monkeypatch.setattr(tts_module, "MLXAudioQwenTTSBackend", FakeBackend)
    monkeypatch.setattr(tts_module, "release_mlx_memory", lambda: releases.append("released"))

    provider = tts_module.LocalTTSProvider(server.settings)

    custom_audio = provider.speech(
        model_id=server.settings.tts_model,
        model_kind="custom_voice",
        text="hello",
        voice="default",
        instruct=None,
        response_format="wav",
        speed=1.0,
    )
    voice_design_audio = provider.speech(
        model_id=server.settings.tts_voice_design_model,
        model_kind="voice_design",
        text="hello",
        voice="default",
        instruct="natural voice",
        response_format="wav",
        speed=1.0,
    )

    assert custom_audio == f"{server.settings.tts_model}:custom_voice".encode()
    assert (
        voice_design_audio
        == f"{server.settings.tts_voice_design_model}:voice_design".encode()
    )
    assert releases == ["released"]
