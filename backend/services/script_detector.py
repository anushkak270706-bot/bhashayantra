class ScriptDetector:

    SCRIPT_RANGES = {
        "devanagari": (0x0900, 0x097F),
        "bengali": (0x0980, 0x09FF),
        "gurmukhi": (0x0A00, 0x0A7F),
        "gujarati": (0x0A80, 0x0AFF),
        "oriya": (0x0B00, 0x0B7F),
        "tamil": (0x0B80, 0x0BFF),
        "telugu": (0x0C00, 0x0C7F),
        "kannada": (0x0C80, 0x0CFF),
        "malayalam": (0x0D00, 0x0D7F),
        "sinhala": (0x0D80, 0x0DFF),
        "arabic": (0x0600, 0x06FF),
    }

    def detect(self, text: str):

        script_counts = {}

        for character in text:
            code_point = ord(character)

            for script, (start, end) in self.SCRIPT_RANGES.items():
                if start <= code_point <= end:
                    script_counts[script] = script_counts.get(script, 0) + 1
                    break

        if not script_counts:
            return {
                "script": "unknown",
                "confidence": 0.0
            }

        detected_script = max(
            script_counts,
            key=script_counts.get
        )

        total_script_characters = sum(script_counts.values())

        confidence = (
            script_counts[detected_script]
            / total_script_characters
        )

        return {
            "script": detected_script,
            "confidence": round(confidence, 4)
        }