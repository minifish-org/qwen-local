from __future__ import annotations

from local_openai_mlx_provider.config import Settings


def test_default_tts_models_use_kokoro_and_qwen3_quality_models() -> None:
    assert Settings.tts_model == "mlx-community/Kokoro-82M-bf16"
    assert Settings.tts_backend == "kokoro-mlx"
    assert Settings.tts_default_voice == "zf_xiaoxiao"
    assert (
        Settings.tts_quality_model
        == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
    )
    assert (
        Settings.tts_voice_design_model
        == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
    )
    assert Settings.api_tts_quality_model == "local-tts-quality"
    assert Settings.api_tts_voice_design_model == "local-tts-voice-design"


def test_default_asr_model_uses_large_v3_turbo() -> None:
    assert Settings.asr_model == "mlx-community/whisper-large-v3-turbo"
