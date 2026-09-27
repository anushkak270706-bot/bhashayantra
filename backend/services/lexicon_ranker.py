"""Re-rank the model's n-best candidates using real-word frequency.

The model often has the right word as its 2nd choice (kal -> काल, alt कल).
Frequency data (open-source `wordfreq`) tells us which candidates are real, common words.

score = model_log_prob + FREQ_WEIGHT * zipf_frequency
Zipf scale: 0 = not seen, ~3 = rare, ~6 = very common, 7+ = function words.
FREQ_WEIGHT should be tuned on a held-out test set (see eval/).
"""
from wordfreq import available_languages, zipf_frequency

SUPPORTED = set(available_languages())
FREQ_WEIGHT = 2.0


def has_lexicon(language_code: str) -> bool:
    return language_code in SUPPORTED


def rerank(hyps: list, language_code: str) -> list:
    """hyps: [(candidate, log_score), ...] -> [{"text", "model_score", "zipf", "known"}], best first."""
    use_freq = has_lexicon(language_code)
    rows = []
    for text, score in hyps:
        z = zipf_frequency(text, language_code) if use_freq else 0.0
        rows.append({"text": text, "model_score": round(score, 3),
                     "zipf": round(z, 2), "known": z > 0})
    rows.sort(key=lambda r: r["model_score"] + FREQ_WEIGHT * r["zipf"], reverse=True)
    return rows
