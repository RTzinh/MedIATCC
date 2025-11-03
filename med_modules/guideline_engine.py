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
            factors.append("ICC")
        if history.get("hypertension"):
            score += 1
            factors.append("Ha")
        if age is not None:
            if age >= 75:
                score += 2
                factors.append("Idade>=75")
            elif age >= 65:
                score += 1
                factors.append("Idade 65-74")
        if history.get("diabetes"):
            score += 1
            factors.append("DM")
        if history.get("stroke_tia"):
            score += 2
            factors.append("AVE/AIT")
        if history.get("vascular_disease"):
            score += 1
            factors.append("Doenca vascular")
        if sex and sex.lower().startswith("f"):
            score += 1
            factors.append("Sexo feminino")
        recommendation = "Considerar anticoagulacao oral." if score >= 2 else "Avaliar risco/beneficio individual."
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
            0: "Tratamento ambulatorial geralmente adequado.",
            1: "Avaliar comorbidades antes de decidir local de tratamento.",
            2: "Considerar internacao hospitalar.",
            3: "Avaliar em unidade de maior complexidade.",
            4: "Alto risco; UTI ou monitorizacao intensiva.",
            5: "Alto risco; UTI ou monitorizacao intensiva.",
        }
        return {
            "score": score,
            "recommendation": recommendations.get(score, "Sem diretriz."),
        }

    def suggest_next_steps(self, *, has_high_risk_lab: bool, imaging_flags: bool, ecg_red_flags: bool) -> Dict[str, Any]:
        plan = []
        if has_high_risk_lab:
            plan.append("Repetir marcadores laboratoriais essenciais em 24-48h.")
        if imaging_flags:
            plan.append("Encaminhar imagens para segunda leitura de radiologista.")
        if ecg_red_flags:
            plan.append("Encaminhar para avaliacao cardiologica urgente.")
        if not plan:
            plan.append("Manter acompanhamento ambulatorial com educacao do paciente.")
        return {"plan": plan}
