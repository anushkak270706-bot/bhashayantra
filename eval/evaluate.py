"""BhashaYantra evaluation: measure accuracy automatically, in any language.

Uses the Aksharantar TEST split (AI4Bharat): verified roman -> native word pairs.
The model was trained on Aksharantar's train split; the test split is held out.

Usage (from the repo root, with .venv active):
    python -m eval.evaluate --lang hi --n 1000
    python -m eval.evaluate --all --n 300

Reports, per language:
  model_top1   : model's own first choice is exactly right
  ranked_top1  : after our frequency re-ranking
  top10        : correct answer is anywhere in the 10 candidates (best ranking could reach)
  CER          : character error rate of our final output (lower is better)
  flag stats   : how trustworthy the needs_review flag is
Exact match is strict: valid spelling variants (e.g. नही / नहीं) count as wrong.
"""
import argparse
import csv
import json
import random
import time
import unicodedata
import zipfile
from pathlib import Path

from backend.services.lexicon_ranker import has_lexicon, rerank
from backend.services.transliterator import MODEL_TAGS, CODE_TO_TAG, Transliterator

# our language code -> Aksharantar file code
FILE_CODE = {"as": "asm", "bn": "ben", "brx": "brx", "gu": "guj", "hi": "hin", "kn": "kan",
             "ks": "kas", "kok": "kok", "mai": "mai", "ml": "mal", "mr": "mar", "mni": "mni",
             "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "sd": "sid", "ta": "tam",
             "te": "tel", "ur": "urd"}

RESULTS = Path(__file__).resolve().parent / "results"


def load_test_pairs(lang: str, local_file: str = None):
    if local_file:
        path = Path(local_file)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id="ai4bharat/Aksharantar",
                                    filename=f"{FILE_CODE[lang]}.zip", repo_type="dataset"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("_test.json"))
            text = z.read(name).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:  # whole-file JSON list
        rows = json.loads(text)
        rows = rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:  # JSON lines
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [(r["english word"].strip(), unicodedata.normalize("NFC", r["native word"].strip())) for r in rows
                        if r.get("english word") and r.get("native word")]


def cer(pred: str, gold: str) -> float:
    prev = list(range(len(gold) + 1))
    for i, p in enumerate(pred, 1):
        cur = [i]
        for j, g in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != g)))
        prev = cur
    return prev[-1] / max(len(gold), 1)


def evaluate(xlit: Transliterator, lang: str, n: int, seed: int, local_file: str = None):
    pairs = load_test_pairs(lang, local_file)
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n]
    lexicon = has_lexicon(lang)

    stats = dict(model_top1=0, ranked_top1=0, top10=0, cer=0.0,
                 flagged=0, flagged_wrong=0, unflagged=0, unflagged_right=0)
    errors = []
    t0 = time.time()
    for start in range(0, len(pairs), 64):
        batch = pairs[start:start + 64]
        nbest = xlit._words_nbest([r for r, _ in batch], lang, 10)
        for (roman, gold), hyps in zip(batch, nbest):
            ranked = rerank(hyps, lang)
            pred = ranked[0]["text"]
            flagged = lexicon and not any(c["known"] for c in ranked)
            right = pred == gold
            stats["model_top1"] += hyps[0][0] == gold
            stats["ranked_top1"] += right
            stats["top10"] += any(h[0] == gold for h in hyps)
            stats["cer"] += cer(pred, gold)
            if flagged:
                stats["flagged"] += 1
                stats["flagged_wrong"] += not right
            else:
                stats["unflagged"] += 1
                stats["unflagged_right"] += right
            if not right:
                errors.append([roman, gold, hyps[0][0], pred, flagged])
    total = len(pairs)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    summary = {
        "language": lang, "words": total, "lexicon_reranking": lexicon,
        "model_top1_%": pct(stats["model_top1"], total),
        "ranked_top1_%": pct(stats["ranked_top1"], total),
        "top10_%": pct(stats["top10"], total),
        "CER_%": round(100 * stats["cer"] / total, 2) if total else None,
        "flagged_%": pct(stats["flagged"], total),
        "flag_precision_%": pct(stats["flagged_wrong"], stats["flagged"]),
        "unflagged_accuracy_%": pct(stats["unflagged_right"], stats["unflagged"]),
        "seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{lang}_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["roman", "correct", "model_first_choice", "our_output", "flagged"])
        w.writerows(errors)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--all", action="store_true", help="every language the model supports")
    ap.add_argument("--n", type=int, default=500, help="test words per language")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--file", help="local zip/json instead of downloading")
    args = ap.parse_args()

    langs = [l for l in FILE_CODE if CODE_TO_TAG.get(l, l) in MODEL_TAGS] if args.all else [args.lang]
    xlit = Transliterator()
    summaries = []
    for lang in langs:
        try:
            s = evaluate(xlit, lang, args.n, args.seed, args.file)
        except Exception as exc:
            s = {"language": lang, "error": str(exc)[:120]}
        summaries.append(s)
        print(json.dumps(s, ensure_ascii=False))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    keys = ["language", "words", "model_top1_%", "ranked_top1_%", "top10_%", "CER_%",
            "flagged_%", "flag_precision_%", "unflagged_accuracy_%"]
    print("\n" + " | ".join(keys))
    for s in summaries:
        print(" | ".join(str(s.get(k, "-")) for k in keys))
    print(f"\nWrong words saved in: {RESULTS}  (one <lang>_errors.csv per language, opens in Excel)")


if __name__ == "__main__":
    main()
