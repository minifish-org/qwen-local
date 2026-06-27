from __future__ import annotations

import threading

from .config import Settings
from .lifecycle import release_mlx_memory

SUPPORTED_NLLB_LANGUAGE_CODES = {
    "eng_Latn",
    "arb_Arab",
    "jpn_Jpan",
    "hin_Deva",
    "ind_Latn",
    "kor_Hang",
    "rus_Cyrl",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
    "ita_Latn",
    "por_Latn",
    "tha_Thai",
    "vie_Latn",
    "zho_Hans",
    "zho_Hant",
}


class NLLBTranslationRuntime:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._translator = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._translator is not None and self._tokenizer is not None:
            return

        with self._lock:
            if self._translator is not None and self._tokenizer is not None:
                return

            if self._settings.translation_backend != "ctranslate2":
                raise RuntimeError(
                    f"unsupported translation backend: {self._settings.translation_backend}"
                )

            try:
                import ctranslate2
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "NLLB translation requires ctranslate2 and transformers. "
                    "Install requirements.txt, then set TRANSLATION_MODEL_PATH "
                    "to a local CTranslate2 int8 NLLB-200 3.3B model directory. "
                    "For 16 GB Macs, an int8 CTranslate2 build or the distilled "
                    "1.3B NLLB model is the supported fallback."
                ) from exc

            model_path = self._settings.translation_model_path.strip()
            if not model_path:
                raise RuntimeError(
                    "TRANSLATION_MODEL_PATH must point to a local CTranslate2 "
                    "NLLB model directory. Use an int8 conversion of "
                    "facebook/nllb-200-3.3B, or use the distilled 1.3B NLLB "
                    "model as the 16 GB Mac fallback."
                )

            self._translator = ctranslate2.Translator(
                model_path,
                device=self._settings.translation_device,
                compute_type=self._settings.translation_compute_type,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._settings.translation_tokenizer_model
            )

    def is_loaded(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    def unload(self) -> None:
        with self._lock:
            self._translator = None
            self._tokenizer = None
        release_mlx_memory()

    def translate(
        self,
        *,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        _validate_language_code(source_language, field="source_language")
        _validate_language_code(target_language, field="target_language")

        self._load()
        assert self._translator is not None
        assert self._tokenizer is not None

        tokenizer = self._tokenizer
        tokenizer.src_lang = source_language
        encoded = tokenizer(text, return_attention_mask=False)
        source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        target_token = _language_token(tokenizer, target_language)

        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[[target_token]],
            max_decoding_length=self._settings.translation_max_decoding_length,
        )
        output_tokens = list(results[0].hypotheses[0])
        if output_tokens and output_tokens[0] == target_token:
            output_tokens = output_tokens[1:]

        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _validate_language_code(value: str, *, field: str) -> None:
    if value not in SUPPORTED_NLLB_LANGUAGE_CODES:
        supported = ", ".join(sorted(SUPPORTED_NLLB_LANGUAGE_CODES))
        raise ValueError(
            f"unsupported {field}: {value}. Supported NLLB language codes: {supported}"
        )


def _language_token(tokenizer, language_code: str) -> str:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", {})
    token_id = lang_code_to_id.get(language_code)
    if token_id is not None:
        return str(tokenizer.convert_ids_to_tokens(token_id))

    candidate = f"__{language_code}__"
    token_id = tokenizer.convert_tokens_to_ids(candidate)
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unknown_token_id:
        return candidate

    return language_code
