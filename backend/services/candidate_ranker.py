class CandidateRanker:

    def rank(self, candidates: list):

        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.get("score", 0),
            reverse=True
        )

        for index, candidate in enumerate(ranked, start=1):
            candidate["rank"] = index

        return ranked