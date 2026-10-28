class RiskAssessor:
    def assess(self, fused_score):
        fused_score = max(0.0, min(1.0, float(fused_score)))
        if fused_score >= 0.9:
            return "CRITICAL"
        if fused_score >= 0.75:
            return "HIGH"
        if fused_score >= 0.5:
            return "MEDIUM"
        return "LOW"

