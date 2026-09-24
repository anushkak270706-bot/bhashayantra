class HumanVerificationService:

    def create_review(self, confidence: dict, candidates: list):

        if confidence["needs_verification"]:
            return {
                "status": "review_required",
                "reason": "Low confidence result requires human verification.",
                "candidates": candidates
            }

        return {
            "status": "auto_approved",
            "reason": "Confidence is sufficient for automatic output.",
            "candidates": candidates
        }