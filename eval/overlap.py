"""Stage A: before building word lists from Aksharantar's TRAIN split, measure

  - how many unique native words each language has (memory/size check), and
  - how many TEST words also appear in TRAIN (would a train word list inflate results?).

Usage (repo root, .venv active):  python -m eval.overlap
Uses the same cached downloads as eval.evaluate, so most zips are already on disk.
"""
import json
import unicodedata
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from eval.evaluate import FILE_CODE, RESULTS


def native_words(z: zipfile.ZipFile, split: str) -> set:
    name = next((n for n in z.namelist() if n.endswith(f"_{split}.json")), None)
    if name is None:
        return set()
    words = set()
    with z.open(name) as f:
        for raw in f:
            line = raw.decode("utf-8").strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = (row.get("native word") or "").strip()
            if w:
                words.add(unicodedata.normalize("NFC", w))
    return words


def main():
    rows = []
    for lang, code in FILE_CODE.items():
        try:
            path = Path(hf_hub_download(repo_id="ai4bharat/Aksharantar",
                                        filename=f"{code}.zip", repo_type="dataset"))
            with zipfile.ZipFile(path) as z:
                train, test = native_words(z, "train"), native_words(z, "test")
            overlap = len(test & train)
            rows.append({
                "language": lang,
                "zip_MB": round(path.stat().st_size / 1e6, 1),
                "train_unique_words": len(train),
                "test_unique_words": len(test),
                "test_words_also_in_train_%": round(100 * overlap / len(test), 1) if test else None,
            })
        except Exception as exc:
            rows.append({"language": lang, "error": str(exc)[:100]})
        print(json.dumps(rows[-1], ensure_ascii=False))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "overlap.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    keys = ["language", "zip_MB", "train_unique_words", "test_unique_words", "test_words_also_in_train_%"]
    print("\n" + " | ".join(keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "-")) for k in keys))
    total = sum(r.get("train_unique_words", 0) for r in rows)
    print(f"\nTotal unique training words, all languages: {total:,}")


if __name__ == "__main__":
    main()
