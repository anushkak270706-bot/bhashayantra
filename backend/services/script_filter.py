"""Drop candidates containing letters from the wrong script (e.g. a stray Kannada
letter in a Hindi word: ದिल्ली). Keeps the originals if every candidate would be dropped."""
import unicodedata

_DEVANAGARI = [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]
_BENGALI = [(0x0980, 0x09FF)]
_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
SCRIPT_RANGES = {
    **{l: _DEVANAGARI for l in ("hi", "mr", "sa", "ne", "mai", "brx", "kok", "gom", "doi")},
    "bn": _BENGALI, "as": _BENGALI, "mni": _BENGALI + [(0xABC0, 0xABFF)],
    "gu": [(0x0A80, 0x0AFF)], "pa": [(0x0A00, 0x0A7F)], "or": [(0x0B00, 0x0B7F)],
    "ta": [(0x0B80, 0x0BFF)], "te": [(0x0C00, 0x0C7F)], "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)], "si": [(0x0D80, 0x0DFF)],
    "ur": _ARABIC, "sd": _ARABIC + _DEVANAGARI, "ks": _ARABIC + _DEVANAGARI,
}
_ALWAYS_OK = {0x200C, 0x200D}  # zero-width joiners used in Indic spelling


def in_script(text: str, language_code: str) -> bool:
    ranges = SCRIPT_RANGES.get(language_code)
    if not ranges:
        return True
    for ch in text:
        cp = ord(ch)
        if cp in _ALWAYS_OK or not unicodedata.category(ch)[0] in ("L", "M"):
            continue  # punctuation, digits, joiners are fine
        if not any(lo <= cp <= hi for lo, hi in ranges):
            return False
    return True


def filter_candidates(hyps: list, language_code: str) -> list:
    kept = [h for h in hyps if in_script(h[0], language_code)]
    return kept or hyps
