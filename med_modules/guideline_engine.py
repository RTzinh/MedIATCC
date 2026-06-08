"""Clinical guideline helper scaffolding."""

from __future__ import annotations

from typing import Any, Dict, Optional


class GuidelineAdvisor:
    """Provide lightweight calculators aligned with common risk scores."""

    def cha2ds2_vasc(self, *, age: Optional[int], sex: Optional[str], history: Dict[str, bool]) -> Dict[str, Any]:
        score = 0
        factors = []
        if history.get("congestive_heart_failure"):
            score += 1
            factors.append("CHF")
        if history.get("hypertension"):
            score += 1
            factors.append("HTN")
        if age is not None:
            if age >= 75:
                score += 2
                factors.append("Age>=75")
            elif age >= 65:
                score += 1
                factors.append("Age 65-74")
        if history.get("diabetes"):
            score += 1
            factors.append("DM")
        if history.get("stroke_tia"):
            score += 2
            factors.append("Stroke/TIA")
        if history.get("vascular_disease"):
            score += 1
            factors.append("Vascular disease")
        if sex and sex.lower().startswith("f"):
            score += 1
            factors.append("Female sex")
        recommendation = "Consider oral anticoagulation." if score >= 2 else "Assess individual risk/benefit."
        return {
            "score": score,
            "factors": factors,
            "recommendation": recommendation,
        }

    def curb65(self, *, age: Optional[int], confusion: bool, urea: Optional[float], respiratory_rate: Optional[int], blood_pressure: Optional[Dict[str, int]]) -> Dict[str, Any]:
        score = 0
        if confusion:
            score += 1
        if urea is not None and urea > 7:
            score += 1
        if respiratory_rate is not None and respiratory_rate >= 30:
            score += 1
        if blood_pressure:
            if blood_pressure.get("systolic", 0) < 90 or blood_pressure.get("diastolic", 0) <= 60:
                score += 1
        if age is not None and age >= 65:
            score += 1
        recommendations = {
            0: "Outpatient treatment is usually appropriate.",
            1: "Assess comorbidities before deciding the treatment setting.",
            2: "Consider hospital admission.",
            3: "Evaluate in a higher-complexity unit.",
            4: "High risk; ICU or intensive monitoring.",
            5: "High risk; ICU or intensive monitoring.",
        }
        return {
            "score": score,
            "recommendation": recommendations.get(score, "No guideline."),
        }

    def suggest_next_steps(self, *, has_high_risk_lab: bool, imaging_flags: bool, ecg_red_flags: bool) -> Dict[str, Any]:
        plan = []
        if has_high_risk_lab:
            plan.append("Repeat essential laboratory markers in 24-48h.")
        if imaging_flags:
            plan.append("Refer images for a radiologist's second read.")
        if ecg_red_flags:
            plan.append("Refer for urgent cardiology evaluation.")
        if not plan:
            plan.append("Keep outpatient follow-up with patient education.")
        return {"plan": plan}
