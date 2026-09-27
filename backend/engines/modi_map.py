"""Verified Modi <-> Devanagari mapping over the Unicode Modi block (U+11600-1165F).

Why our own table instead of indic_transliteration's 'modi' scheme: in testing, that
scheme turns र + ी into ॠ (श्री -> श्ॠ, क्षत्रिय -> क्षत्ॠय) and merges short/long i.
This table is character-for-character and round-trips exactly.
"""

_DEVA = (
    "अआइईउऊऋॠऌॡएऐओऔ"                                   # vowels        U+11600-1160D
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"                # consonants    U+1160E-1162F
    "\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0962\u0963"  # vowel signs U+11630-11638
    "\u0947\u0948\u094b\u094c"                           #               U+11639-1163C
    "\u0902\u0903\u094d\u0901"                           # anusvara, visarga, virama, ardhacandra
    "।॥"                                                 # dandas        U+11641-11642
)

MODI_TO_DEVA = {chr(0x11600 + i): ch for i, ch in enumerate(_DEVA)}
MODI_TO_DEVA.update({chr(0x11650 + d): chr(0x0966 + d) for d in range(10)})  # digits
DEVA_TO_MODI = {v: k for k, v in MODI_TO_DEVA.items()}

# Abbreviation sign and sign huva have no single Devanagari equivalent: kept as-is.
UNMAPPED_MODI = {"\U00011643", "\U00011644"}


def modi_to_devanagari(text: str) -> str:
    return "".join(MODI_TO_DEVA.get(ch, ch) for ch in text)


def devanagari_to_modi(text: str) -> str:
    return "".join(DEVA_TO_MODI.get(ch, ch) for ch in text)


def unmapped_count(text: str) -> int:
    return sum(ch in UNMAPPED_MODI for ch in text)
