"""Bridge to the GPU manuscript reader running in Colab.

Colab runs the Modi vision model and opens a temporary https link (Cloudflare tunnel).
It registers that link here with a shared secret, so the link never needs pasting.
Visitors' photos are forwarded to the GPU, read, and never stored.

Render environment variable (set once): MANUSCRIPT_GPU_TOKEN = any long random string.
The same string goes into the Colab notebook.
"""
import json
import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.correction_store import DATABASE_URL, _connect

log = logging.getLogger(__name__)
router = APIRouter()

TOKEN = os.getenv("MANUSCRIPT_GPU_TOKEN", "")
FLAG_THRESHOLD = 0.326          # from the 60-page manuscript evaluation (least-confident 20% of words)
MAX_UPLOAD = 8 * 1024 * 1024    # 8 MB
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "data" / "settings.json"
_cache = {"url": None, "loaded": False}


# ---------- where the GPU link is kept (Postgres if available, so it survives restarts) ----------
def _save_url(url: str):
    _cache.update(url=url, loaded=True)
    try:
        if DATABASE_URL:
            with _connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT, "
                             "updated_at TIMESTAMPTZ DEFAULT now())")
                conn.execute("INSERT INTO app_settings (key, value) VALUES ('gpu_url', %s) "
                             "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", [url])
        else:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps({"gpu_url": url}), encoding="utf-8")
    except Exception:
        log.exception("Could not persist GPU url (kept in memory)")


def _load_url():
    if _cache["loaded"]:
        return _cache["url"]
    url = None
    try:
        if DATABASE_URL:
            with _connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT, "
                             "updated_at TIMESTAMPTZ DEFAULT now())")
                row = conn.execute("SELECT value FROM app_settings WHERE key = 'gpu_url'").fetchone()
                url = row[0] if row else None
        elif SETTINGS_FILE.exists():
            url = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("gpu_url")
    except Exception:
        log.exception("Could not load GPU url")
    _cache.update(url=url, loaded=True)
    return url


# ---------- endpoints ----------
class Registration(BaseModel):
    url: str


@router.post("/manuscript/register", include_in_schema=False)
def register(reg: Registration, x_token: str = Header(default="")):
    if not TOKEN or x_token != TOKEN:
        raise HTTPException(status_code=403, detail="Not allowed.")
    url = reg.url.strip().rstrip("/")
    if not url.startswith("https://") or len(url) > 200:
        raise HTTPException(status_code=400, detail="Expected an https link.")
    _save_url(url)
    return {"registered": True}


@router.get("/manuscript/status")
async def status():
    url = _load_url()
    if not url or not TOKEN:
        return {"online": False}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{url}/health", headers={"X-Token": TOKEN})
        return {"online": r.status_code == 200}
    except Exception:
        return {"online": False}


@router.post("/manuscript/read")
async def read_page(file: UploadFile = File(...)):
    url = _load_url()
    if not url or not TOKEN:
        raise HTTPException(status_code=503, detail="The GPU reader is offline right now.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Image too large (8 MB max).")
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=150) as client:
            r = await client.post(f"{url}/read", headers={"X-Token": TOKEN},
                                  files={"file": (file.filename or "page.jpg", data, file.content_type)})
        r.raise_for_status()
        out = r.json()
    except Exception:
        log.exception("GPU reader failed")
        raise HTTPException(status_code=503, detail="The GPU reader didn't respond. It may have gone offline.")
    words = out.get("text", "").split()
    conf = (out.get("word_conf") or []) + [1.0] * len(words)
    return {
        "text": out.get("text", ""),
        "words": [{"text": w, "confidence": round(c, 3), "flagged": c <= FLAG_THRESHOLD}
                  for w, c in zip(words, conf)],
        "variant": out.get("variant", "raw"),
        "loop_detected": bool(out.get("loop_detected")),
        "model": out.get("model"),
        "seconds": round(time.time() - t0, 1),
        "stored": False,
    }
