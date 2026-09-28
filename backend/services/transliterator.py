import os
import re
import unicodedata

import ctranslate2
from huggingface_hub import snapshot_download

from backend.services.lexicon_ranker import needs_review, rerank
from backend.services.script_filter import in_script

WIDE_N = 10  # second, wider search for words the model is unsure about


MODEL_REPO = "Singla0009/all-indic-transliteration"

# IndicXlit is a WORD-level model. Feeding a whole sentence (spaces included) is what
# produced "मेराभारतमहानहै". We split into words, transliterate them in one batch,
# and put the spaces/punctuation back exactly where they were.
_TOKEN = re.compile(r"(\s+|[^\w\s]+)")

# Exact language tags in the model's vocabulary (checked from source_vocabulary.json)
MODEL_TAGS = {"as", "bn", "brx", "gom", "gu", "hi", "kn", "ks", "mai", "ml", "mni",
              "mr", "ne", "or", "pa", "sa", "sd", "si", "ta", "te", "ur"}
CODE_TO_TAG = {"kok": "gom", "bn": "as", "as": "bn"}  # our code -> model's tag


def model_tag(language_code: str) -> str:
    tag = CODE_TO_TAG.get(language_code, language_code)
    if tag not in MODEL_TAGS:
        raise ValueError(f"The transliteration model has no support for '{language_code}'")
    return tag


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
        batch = [[f"__{model_tag(language_code)}__"] + list(w) for w in words]
        results = self.engine.translate_batch(
            batch,
            beam_size=max(4, n),
            num_hypotheses=n,
            return_scores=True,
        )
        return [
            [(unicodedata.normalize("NFC", "".join(h)), float(s)) for h, s in zip(r.hypotheses, r.scores)]
            for r in results
        ]

    def roman_to_indic_detailed(self, text: str, language_code: str, n: int = 3, learned=None):
        """Returns (output_text, per_word ranked candidates).

        Each word's candidates are the model's n-best, re-ranked by real-word frequency.
        """
        tokens = [t for t in _TOKEN.split(text) if t]
        word_idx = [i for i, t in enumerate(tokens) if re.search(r"\w", t) and not t.isspace()]
        if not word_idx:
            return text, []

        nbest = self._words_nbest([tokens[i] for i in word_idx], language_code, n)
        ranked = [rerank(hyps, language_code) for hyps in nbest]

        # Adaptive search: most words are confident and stay fast. Unsure words get a
        # wider search, used when it finds a real word the first search missed, or when
        # the model becomes confident. Such words can still be flagged for review.
        unsure = [k for k, cands in enumerate(ranked) if needs_review(cands, language_code)]
        if unsure and n < WIDE_N:
            wide = self._words_nbest([tokens[word_idx[k]] for k in unsure], language_code, WIDE_N)
            for k, hyps in zip(unsure, wide):
                wider = rerank(hyps, language_code)
                found_real_word = wider[0]["known"] and not ranked[k][0]["known"]
                if found_real_word or not needs_review(wider, language_code):
                    ranked[k] = wider
        if learned:
            ranked = [_apply_learned(cands, [(t, c) for t, c in learned(language_code, tokens[i])
                                             if in_script(t, language_code)])
                      for i, cands in zip(word_idx, ranked)]
        out = list(tokens)
        for i, cands in zip(word_idx, ranked):
            out[i] = cands[0]["text"]
        return "".join(out), ranked

    def roman_to_indic(self, text: str, language_code: str) -> str:
        return self.roman_to_indic_detailed(text, language_code, n=3)[0]


def _apply_learned(cands: list, votes: list, min_agreement: int = 2) -> list:
    """Put a user-corrected reading first. Trusted immediately if it is one of the
    model's own candidates; otherwise only once min_agreement people chose it."""
    texts = [c["text"] for c in cands]
    for text, count in votes:
        if text in texts or count >= min_agreement:
            base = next((c for c in cands if c["text"] == text),
                        {"text": text, "model_score": None, "zipf": 0.0})
            row = {**base, "known": True, "confidence": 1.0, "learned": True, "votes": count}
            return [row] + [c for c in cands if c["text"] != text]
    return cands
