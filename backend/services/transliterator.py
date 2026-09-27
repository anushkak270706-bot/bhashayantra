import os
import re

import ctranslate2
from huggingface_hub import snapshot_download


MODEL_REPO = "Singla0009/all-indic-transliteration"

# IndicXlit is a WORD-level model. Feeding a whole sentence (spaces included) is what
# produced "मेराभारतमहानहै". We split into words, transliterate them in one batch,
# and put the spaces/punctuation back exactly where they were.
_TOKEN = re.compile(r"(\s+|[^\w\s]+)")


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

    def _words_nbest(self, words: list, language_code: str, n: int):
        batch = [[f"__{language_code}__"] + list(w) for w in words]
        results = self.engine.translate_batch(
            batch,
            beam_size=max(4, n),
            num_hypotheses=n,
            return_scores=True,
        )
        return [
            [("".join(h), float(s)) for h, s in zip(r.hypotheses, r.scores)]
            for r in results
        ]

    def roman_to_indic_detailed(self, text: str, language_code: str, n: int = 3):
        """Returns (output_text, per_word list of n-best [(candidate, log_score), ...])."""
        tokens = [t for t in _TOKEN.split(text) if t]
        word_idx = [i for i, t in enumerate(tokens) if re.search(r"\w", t) and not t.isspace()]
        if not word_idx:
            return text, []

        nbest = self._words_nbest([tokens[i] for i in word_idx], language_code, n)
        out = list(tokens)
        for i, hyps in zip(word_idx, nbest):
            out[i] = hyps[0][0]
        return "".join(out), nbest

    def roman_to_indic(self, text: str, language_code: str) -> str:
        return self.roman_to_indic_detailed(text, language_code, n=1)[0]
