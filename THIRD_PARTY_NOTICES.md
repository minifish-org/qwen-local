# Models and third-party components

The AGPL license covers this repository's adapter code, not downloaded models,
Python packages, tokenizers or voices. Dependencies retain their own licenses.
No model weights are distributed in this repository.

Review the model card and license for the exact revision you download:

- [Qwen3.5-4B MLX](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit)
- [Qwen3 Embedding](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ)
- [Kokoro](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Qwen3 TTS CustomVoice](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit)
- [Qwen3 TTS VoiceDesign](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit)
- [Whisper](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
- [NLLB-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B): CC-BY-NC-4.0. Its noncommercial restriction is separate from the adapter's AGPL terms.

Conversions and quantization do not remove upstream licensing restrictions.
The requirements file identifies runtime packages; install them from their
upstream distributions with their original notices intact.
