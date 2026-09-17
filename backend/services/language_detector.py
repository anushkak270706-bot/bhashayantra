import os
import fasttext


class LanguageDetector:
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "indiclid",
            "ftr",
            "indiclid-ftr",
            "model_baseline_roman.bin",
        )

        self.model = fasttext.load_model(model_path)

    def detect(self, text: str):
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