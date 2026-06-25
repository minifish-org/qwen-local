from __future__ import annotations

from local_openai_mlx_provider.config import Settings


def test_default_asr_model_uses_large_v3_turbo() -> None:
    assert Settings.asr_model == "mlx-community/whisper-large-v3-turbo"
