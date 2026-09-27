import os


class LanguageDetector:
    """IndicLID (fastText) language detection.

    The model is ~269 MB. On small deployments (e.g. Render free, 512 MB RAM) set
    BHASHAYANTRA_LANG_DETECT=off to skip it; everything else keeps working and
    users pick the language themselves.
    """

    def __init__(self):
        self.model = None
        self.reason = None

        if os.getenv("BHASHAYANTRA_LANG_DETECT", "on").lower() == "off":
            self.reason = "Language detection is turned off on this deployment. Please choose the language."
            return

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "indiclid",
            "ftr",
            "indiclid-ftr",
            "model_baseline_roman.bin",
        )
        try:
            import fasttext  # imported here so deployments without it still start
            self.model = fasttext.load_model(model_path)
        except Exception as exc:  # missing package, or an LFS pointer instead of the real file
            self.reason = f"Language detection is unavailable ({type(exc).__name__}). Please choose the language."

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, text: str):
        if not self.available:
            raise RuntimeError(self.reason)

        labels, probabilities = self.model.predict(
            text,
            k=3
        )

        results = []

        for label, probability in zip(labels, probabilities):
            results.append({
                "language": label.replace("__label__", ""),
                "confidence": float(probability)
            })

        return results
