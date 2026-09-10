import os

import ctranslate2
from huggingface_hub import snapshot_download


MODEL_REPO = "Singla0009/all-indic-transliteration"


class Transliterator:
    def __init__(self):
        model_path = snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns="indicxlit_ct2_fp32/*",
        )

        model_dir = os.path.join(
            model_path,
            "indicxlit_ct2_fp32",
        )

        self.engine = ctranslate2.Translator(
            model_dir,
            device="cpu",
        )

    def roman_to_indic(self, text: str, language_code: str) -> str:
        source_tokens = [f"__{language_code}__"] + list(text)

        results = self.engine.translate_batch(
            [source_tokens],
            beam_size=4,
        )

        return "".join(results[0].hypotheses[0])