"""Every expert correction is saved as a labelled training pair (JSONL for the prototype)."""
import json
import time
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "corrections.jsonl"


def save_correction(record: dict) -> int:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**record, "ts": time.time()}, ensure_ascii=False) + "\n")
    with STORE.open(encoding="utf-8") as f:
        return sum(1 for _ in f)
