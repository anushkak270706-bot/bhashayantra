from indic_transliteration import sanscript

from backend.engines import modi_map


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

    # Heritage scripts handled by our own verified mapping (not the library).
    HERITAGE_SCRIPTS = {"modi"}

    def supported_scripts(self):
        return sorted(set(self.SCRIPT_MAP) | self.HERITAGE_SCRIPTS)

    def convert(self, text: str, source_script: str, target_script: str):
        source_script = source_script.lower()
        target_script = target_script.lower()

        if source_script == target_script:
            return text

        # Modi <-> anything: go through Devanagari with our own table
        if "modi" in (source_script, target_script):
            if source_script == "modi":
                deva = modi_map.modi_to_devanagari(text)
                return deva if target_script == "devanagari" else self.convert(deva, "devanagari", target_script)
            deva = text if source_script == "devanagari" else self.convert(text, source_script, "devanagari")
            return modi_map.devanagari_to_modi(deva)

        source = self.SCRIPT_MAP.get(source_script)
        target = self.SCRIPT_MAP.get(target_script)

        if source is None:
            raise ValueError(f"Unsupported source script: {source_script}")
        if target is None:
            raise ValueError(f"Unsupported target script: {target_script}")

        return sanscript.transliterate(text, source, target)
