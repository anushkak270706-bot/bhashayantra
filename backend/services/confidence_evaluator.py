class ConfidenceEvaluator:

    def evaluate(self, score: float):

        if score >= 0.85:
            level = "high"
            needs_verification = False

        elif score >= 0.60:
            level = "medium"
            needs_verification = False

        else:
            level = "low"
            needs_verification = True

        return {
            "score": round(score, 4),
            "level": level,
            "needs_verification": needs_verification
        }