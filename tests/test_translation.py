from __future__ import annotations

import pytest

from local_openai_mlx_provider import server


def test_models_includes_translation_model(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["id"] for item in response.json()["data"]}
    assert server.settings.api_translation_model in model_ids


@pytest.mark.parametrize(
    ("source_language", "target_language", "text", "expected"),
    [
        ("jpn_Jpan", "eng_Latn", "こんにちは", "Hello."),
        ("kor_Hang", "eng_Latn", "안녕하세요", "Hello."),
        ("spa_Latn", "zho_Hans", "Buenos dias", "早上好。"),
        ("fra_Latn", "zho_Hans", "Bonjour", "你好。"),
    ],
)
def test_translations_for_common_language_pairs(
    client, monkeypatch, source_language, target_language, text, expected
):
    calls = []

    def fake_translate(*, text, source_language, target_language):
        calls.append((text, source_language, target_language))
        return expected

    monkeypatch.setattr(server.translation_runtime, "translate", fake_translate)

    response = client.post(
        "/v1/translations",
        json={
            "model": server.settings.api_translation_model,
            "source_language": source_language,
            "target_language": target_language,
            "text": text,
            "keep_alive": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "translated_text": expected,
        "source_language": source_language,
        "target_language": target_language,
        "model": server.settings.api_translation_model,
    }
    assert calls == [(text, source_language, target_language)]


def test_translations_rejects_unsupported_model(client):
    response = client.post(
        "/v1/translations",
        json={
            "model": "remote-translator",
            "source_language": "jpn_Jpan",
            "target_language": "eng_Latn",
            "text": "こんにちは",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "bad_request"
    assert payload["error"]["param"] == "model"
    assert "unsupported translation model" in payload["error"]["message"]


def test_translations_rejects_unsupported_language(client):
    response = client.post(
        "/v1/translations",
        json={
            "model": server.settings.api_translation_model,
            "source_language": "zzz_Zzzz",
            "target_language": "eng_Latn",
            "text": "hello",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "bad_request"
    assert payload["error"]["param"] == "source_language"
    assert "Supported NLLB language codes" in payload["error"]["message"]


def test_translations_rejects_empty_text(client):
    response = client.post(
        "/v1/translations",
        json={
            "model": server.settings.api_translation_model,
            "source_language": "fra_Latn",
            "target_language": "eng_Latn",
            "text": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "text"


def test_translations_validation_errors_are_json(client):
    response = client.post(
        "/v1/translations",
        json={
            "source_language": "fra_Latn",
            "target_language": "eng_Latn",
            "text": "Bonjour",
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "bad_request"
    assert payload["error"]["param"] == "model"
