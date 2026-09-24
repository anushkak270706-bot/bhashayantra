class TransliterationRouter:

    def __init__(self, indicxlit_engine, script_bridge):
        self.indicxlit_engine = indicxlit_engine
        self.script_bridge = script_bridge

    def route(self, input_type: str, language_code: str = None):
        if input_type == "text":
            if language_code:
                return self.indicxlit_engine

            raise ValueError(
                "Language code is required for text transliteration."
            )

        if input_type == "script":
            return self.script_bridge

        raise ValueError(
            f"Unsupported input type: {input_type}"
        )