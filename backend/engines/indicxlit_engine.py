from backend.services.transliterator import Transliterator


class IndicXlitEngine:

    def __init__(self, transliterator: Transliterator):
        self.transliterator = transliterator

    def transliterate(self, text: str, language_code: str):
        return self.transliterator.roman_to_indic(
            text=text,
            language_code=language_code
        )