from backend.services.language_detector import LanguageDetector


detector = LanguageDetector()

test_inputs = [
    "mera naam arpit hai",
    "maza naav arpit aahe",
    "amar naam arpit",
]

for text in test_inputs:
    results = detector.detect(text)

    print("\nInput:", text)
    print("Predictions:")

    for result in results:
        print(
            result["language"],
            "->",
            result["confidence"]
        )