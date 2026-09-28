"""BhashaYantra evaluation: measure accuracy automatically, in any language.

Uses Aksharantar (AI4Bharat) verified roman -> native word pairs. Test words share
0.0% overlap with training words, so scores measure performance on unseen words.

Usage (repo root, .venv active):
    python -m eval.evaluate --lang hi --n 1000
    python -m eval.evaluate --all --n 300
    python -m eval.evaluate --all --n 300 --split valid --sweep   # choose the flag threshold
Metrics per language:
  model_top1 / ranked_top1 : first answer exactly right, before / after re-ranking
  top10        : right answer anywhere among 10 candidates
  CER          : character error rate of our output (lower is better)
  flagged      : share of words sent for human review
  flag_precision: of flagged words, share that were really wrong
  catch_rate   : of wrong words, share that were flagged
  unflagged_accuracy : accuracy on words NOT flagged (what a user can trust)
Exact match is strict: valid spelling variants count as wrong.
"""
import argparse
import csv
import json
import random
import time
import unicodedata
import zipfile
from pathlib import Path

from backend.services.lexicon_ranker import CONF_THRESHOLD, has_lexicon, needs_review, rerank
from backend.services.transliterator import CODE_TO_TAG, MODEL_TAGS, Transliterator

FILE_CODE = {"as": "asm", "bn": "ben", "brx": "brx", "gu": "guj", "hi": "hin", "kn": "kan",
             "ks": "kas", "kok": "kok", "mai": "mai", "ml": "mal", "mr": "mar", "mni": "mni",
             "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "sd": "sid", "ta": "tam",
             "te": "tel", "ur": "urd"}
FLAG_BUDGETS = [5, 10, 20, 30, 40]  # % of least-confident words to flag in the sweep
RESULTS = Path(__file__).resolve().parent / "results"
nfc = lambda s: unicodedata.normalize("NFC", s)


def load_pairs(lang: str, split: str = "test", local_file: str = None):
    if local_file:
        path = Path(local_file)
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id="ai4bharat/Aksharantar",
                                    filename=f"{FILE_CODE[lang]}.zip", repo_type="dataset"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(f"_{split}.json"))
            text = z.read(name).decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
        rows = rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [(r["english word"].strip(), nfc(r["native word"].strip())) for r in rows
            if r.get("english word") and r.get("native word")]


def cer(pred: str, gold: str) -> float:
    prev = list(range(len(gold) + 1))
    for i, p in enumerate(pred, 1):
        cur = [i]
        for j, g in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != g)))
        prev = cur
    return prev[-1] / max(len(gold), 1)


def flag_stats(items, threshold):
    """items: [(right: bool, ranked, lang)] -> flag metrics at this threshold."""
    flagged = [needs_review(r, lang, threshold) for _, r, lang in items]
    wrong = [not right for right, _, _ in items]
    n_flag = sum(flagged)
    n_wrong = sum(wrong)
    fw = sum(f and w for f, w in zip(flagged, wrong))
    unflagged_right = sum((not f) and (not w) for f, w in zip(flagged, wrong))
    n_unflag = len(items) - n_flag
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {"flagged_%": pct(n_flag, len(items)), "flag_precision_%": pct(fw, n_flag),
            "catch_rate_%": pct(fw, n_wrong), "unflagged_accuracy_%": pct(unflagged_right, n_unflag)}


def evaluate(xlit, lang, n, seed, split="test", local_file=None):
    pairs = load_pairs(lang, split, local_file)
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n]
    stats = dict(model_top1=0, ranked_top1=0, top10=0, cer=0.0)
    items, errors = [], []
    t0 = time.time()
    for start in range(0, len(pairs), 64):
        batch = pairs[start:start + 64]
        nbest = xlit._words_nbest([r for r, _ in batch], lang, 10)
        for (roman, gold), hyps in zip(batch, nbest):
            ranked = rerank(hyps, lang)
            pred = ranked[0]["text"]
            right = pred == gold
            stats["model_top1"] += hyps[0][0] == gold
            stats["ranked_top1"] += right
            stats["top10"] += any(h[0] == gold for h in hyps)
            stats["cer"] += cer(pred, gold)
            items.append((right, ranked, lang))
            if not right:
                errors.append([roman, gold, hyps[0][0], pred, ranked[0]["confidence"],
                               needs_review(ranked, lang)])
    total = len(pairs)
    pct = lambda a: round(100 * a / total, 1) if total else None
    summary = {"language": lang, "split": split, "words": total,
               "lexicon_reranking": has_lexicon(lang),
               "model_top1_%": pct(stats["model_top1"]), "ranked_top1_%": pct(stats["ranked_top1"]),
               "top10_%": pct(stats["top10"]),
               "CER_%": round(100 * stats["cer"] / total, 2) if total else None,
               **flag_stats(items, CONF_THRESHOLD), "seconds": round(time.time() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{lang}_{split}_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["roman", "correct", "model_first_choice", "our_output", "confidence", "flagged"])
        w.writerows(errors)
    return summary, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--all", action="store_true", help="every language the model supports")
    ap.add_argument("--n", type=int, default=500, help="words per language")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--sweep", action="store_true", help="report flag metrics at several flag budgets")
    ap.add_argument("--file", help="local zip/json instead of downloading")
    args = ap.parse_args()

    langs = [l for l in FILE_CODE if CODE_TO_TAG.get(l, l) in MODEL_TAGS] if args.all else [args.lang]
    xlit = Transliterator()
    summaries, all_items = [], []
    for lang in langs:
        try:
            s, items = evaluate(xlit, lang, args.n, args.seed, args.split, args.file)
            all_items += items
        except Exception as exc:
            s = {"language": lang, "error": str(exc)[:120]}
        summaries.append(s)
        print(json.dumps({k: v for k, v in s.items() if k != "sweep"}, ensure_ascii=False))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"summary_{args.split}.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ["language", "words", "model_top1_%", "ranked_top1_%", "top10_%", "CER_%",
            "flagged_%", "flag_precision_%", "catch_rate_%", "unflagged_accuracy_%"]
    print(f"\n[{args.split} split, flag threshold {CONF_THRESHOLD}]")
    print(" | ".join(keys))
    for s in summaries:
        print(" | ".join(str(s.get(k, "-")) for k in keys))

    if args.sweep and all_items:
        confs = sorted(r[0]["confidence"] for _, r, _ in all_items)
        base_err = round(100 * sum(not right for right, _, _ in all_items) / len(all_items), 1)
        q = lambda p: confs[min(len(confs) - 1, int(len(confs) * p / 100))]
        print("\nCONFIDENCE DISTRIBUTION (all languages): "
              f"10th pct {q(10)} | median {q(50)} | 90th pct {q(90)}")
        print(f"Baseline: {base_err}% of words are wrong. Random flagging would have "
              f"flag_precision = {base_err}%; better flags beat this.")
        print("\nFLAG BUDGET SWEEP (flag the least-confident X% of words; all languages pooled)")
        print("budget_% | threshold | flagged_% | flag_precision_% | catch_rate_% | unflagged_accuracy_%")
        for b in FLAG_BUDGETS:
            t = q(b)
            st = flag_stats(all_items, t)
            print(f"{b} | {t} | {st['flagged_%']} | {st['flag_precision_%']} | {st['catch_rate_%']} | {st['unflagged_accuracy_%']}")
        print("\n(flagged_% can exceed the budget in hi/bn/ta/ur, where 'no known word' also flags.)")
    print(f"\nWrong words saved in: {RESULTS}  (<lang>_<split>_errors.csv, opens in Excel)")


if __name__ == "__main__":
    main()
