from __future__ import annotations

from local_openai_mlx_provider.config import Settings


def test_default_tts_models_use_qwen3_custom_voice_0_6b_and_voice_design_1_7b() -> None:
    assert Settings.tts_model == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    assert (
        Settings.tts_voice_design_model
        == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
    )
    assert Settings.api_tts_voice_design_model == "local-tts-voice-design"


def test_default_asr_model_uses_large_v3_turbo() -> None:
    assert Settings.asr_model == "mlx-community/whisper-large-v3-turbo"
