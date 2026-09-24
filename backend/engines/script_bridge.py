from indic_transliteration import sanscript


class ScriptBridge:

    SCRIPT_MAP = {
        "devanagari": sanscript.DEVANAGARI,
        "bengali": sanscript.BENGALI,
        "gujarati": sanscript.GUJARATI,
        "gurmukhi": sanscript.GURMUKHI,
        "kannada": sanscript.KANNADA,
        "malayalam": sanscript.MALAYALAM,
        "oriya": sanscript.ORIYA,
        "tamil": sanscript.TAMIL,
        "telugu": sanscript.TELUGU,
    }

    def convert(self, text: str, source_script: str, target_script: str):

        source = self.SCRIPT_MAP.get(source_script.lower())
        target = self.SCRIPT_MAP.get(target_script.lower())

        if source is None:
            raise ValueError(
                f"Unsupported source script: {source_script}"
            )

        if target is None:
            raise ValueError(
                f"Unsupported target script: {target_script}"
            )

        return sanscript.transliterate(
            text,
            source,
            target
        )