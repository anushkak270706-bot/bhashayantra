"""Re-rank the model's candidates and decide which words need human review.

Ranking: score = model_log_prob + FREQ_WEIGHT * zipf_frequency (wordfreq, where available).
Review flag, used in every language:
  - confidence: the model's own probability share for the chosen word among its top
    CONF_TOP_K candidates. A low share means the model itself was unsure.
  - where a word list exists: also flag when no candidate is a known real word.
CONF_THRESHOLD is chosen on the Aksharantar VALIDATION split (python -m eval.evaluate
--split valid --sweep) and reported on the TEST split, so it isn't tuned on the test set.
"""
import math

from wordfreq import available_languages, zipf_frequency

from backend.services.script_filter import filter_candidates

SUPPORTED = set(available_languages())
FREQ_WEIGHT = 2.0
CONF_TOP_K = 4          # same number of candidates the live app requests
CONF_THRESHOLD = 0.29   # chosen on the validation split (flag ~20-25% of words)


def has_lexicon(language_code: str) -> bool:
    return language_code in SUPPORTED


def _shares(hyps: list) -> dict:
    """Softmax over the model's top-k log scores -> {candidate: probability share}."""
    top = hyps[:CONF_TOP_K]
    if not top:
        return {}
    m = max(s for _, s in top)
    exps = [math.exp(s - m) for _, s in top]
    total = sum(exps)
    shares = {}
    for (text, _), e in zip(top, exps):
        shares[text] = shares.get(text, 0.0) + e / total
    return shares


def rerank(hyps: list, language_code: str) -> list:
    """hyps: [(candidate, log_score), ...] best-first ->
    [{"text", "model_score", "zipf", "known", "confidence"}], best first."""
    hyps = filter_candidates(hyps, language_code)
    use_freq = has_lexicon(language_code)
    shares = _shares(hyps)
    rows = []
    for text, score in hyps:
        z = zipf_frequency(text, language_code) if use_freq else 0.0
        rows.append({"text": text, "model_score": round(score, 3), "zipf": round(z, 2),
                     "known": z > 0, "confidence": round(shares.get(text, 0.0), 3)})
    rows.sort(key=lambda r: r["model_score"] + FREQ_WEIGHT * r["zipf"], reverse=True)
    return rows


def needs_review(ranked: list, language_code: str, threshold: float = None) -> bool:
    """True when a human should check this word."""
    if not ranked:
        return False
    t = CONF_THRESHOLD if threshold is None else threshold
    unsure = ranked[0]["confidence"] < t
    unknown = has_lexicon(language_code) and not any(c["known"] for c in ranked)
    return unsure or unknown
