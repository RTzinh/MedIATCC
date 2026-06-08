"""Hackathon triage pipeline for nursing staff.

This module keeps the core logic decoupled from the Streamlit UI so it can be
reused by scripts or APIs. The pipeline follows the stages:

1. Collection: validates and normalizes clinical inputs.
2. Processing: runs rule-based analyses with an audit trail for interpretability.
3. Report: produces a structured summary with alerts and recommendations.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]


@dataclass
class NursingTriageInput:
    systolic: Optional[float] = None
    diastolic: Optional[float] = None
    heart_rate: Optional[float] = None
    temperature: Optional[float] = None
    spo2: Optional[float] = None
    age: Optional[int] = None
    sex: str = "Not informed"
    chronic_conditions: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    blood_type: Optional[str] = None
    medications: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)

    def sanitized_symptoms(self) -> List[str]:
        return [sym.strip().lower() for sym in self.symptoms if sym.strip()]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TriagePipelineOutput:
    coleta: Dict[str, Any]
    processamento: Dict[str, Any]
    relatorio: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        relatorio = self.relatorio
        alerts = relatorio.get("alertas", [])
        recomendacoes = relatorio.get("encaminhamentos", [])
        lines = [
            f"**Priority:** {relatorio.get('prioridade', 'N/A')}  ",
            f"**Risk classification:** {relatorio.get('risco', 'N/A')}  ",
            f"**Main rationale:** {relatorio.get('justificativa', 'No rationale recorded.')}  ",
        ]
        if alerts:
            lines.append("**Identified alerts:**")
            lines.extend([f"- {alert}" for alert in alerts])
        if recomendacoes:
            lines.append("**Suggested referrals:**")
            lines.extend([f"- {item}" for item in recomendacoes])
        return "\n".join(lines)


def _score_risk(value: int, current: int) -> int:
    return max(current, min(value, len(RISK_LEVELS) - 1))


def generate_triage_report(payload: NursingTriageInput) -> TriagePipelineOutput:
    audit_trail: List[Dict[str, Any]] = []
    alerts: List[str] = []
    suggested_actions: List[str] = []
    suggested_exams: List[str] = []
    risk_score = 0

    def add_alert(metric: str, message: str, severity: int) -> None:
        nonlocal risk_score
        risk_score = _score_risk(severity, risk_score)
        alerts.append(message)
        audit_trail.append({"metric": metric, "message": message, "impact": RISK_LEVELS[severity]})

    # Blood pressure
    if payload.systolic and payload.diastolic:
        if payload.systolic >= 180 or payload.diastolic >= 120:
            add_alert("BP", "Critical blood pressure consistent with hypertensive emergency.", 3)
            suggested_actions.append("Refer immediately to the emergency room.")
        elif payload.systolic >= 160 or payload.diastolic >= 100:
            add_alert("BP", "Very high blood pressure requires a quick medical evaluation.", 2)
            suggested_actions.append("Prioritize a medical consultation within the next 2 hours.")
        elif payload.systolic <= 90 or payload.diastolic <= 60:
            add_alert("BP", "Low blood pressure detected; rule out shock or hypovolemia.", 2)
            suggested_exams.append("Repeat BP measurement and assess signs of shock.")

    # Heart rate
    if payload.heart_rate:
        if payload.heart_rate >= 130:
            add_alert("HR", "Marked tachycardia may indicate hemodynamic instability.", 3)
        elif payload.heart_rate >= 110:
            add_alert("HR", "Moderate tachycardia observed.", 2)
        elif payload.heart_rate <= 50:
            add_alert("HR", "Bradycardia requires investigation.", 2)

    # Temperature
    if payload.temperature:
        if payload.temperature >= 39:
            add_alert("Temperature", "High fever suggests a severe infectious condition.", 2)
            suggested_exams.append("Order a CBC / CRP according to protocol.")
        elif payload.temperature <= 35:
            add_alert("Temperature", "Hypothermia detected.", 3)

    # Oxygen saturation
    if payload.spo2:
        if payload.spo2 < 90:
            add_alert("SpO2", "Saturation below 90% indicates significant hypoxemia.", 3)
            suggested_actions.append("Start oxygen therapy and call the medical team.")
        elif payload.spo2 < 94:
            add_alert("SpO2", "Slightly reduced saturation, monitor.", 2)

    # Symptoms (keys match the patient's Portuguese symptom input)
    symptom_flags = {
        "dor no peito": 3,
        "falta de ar": 3,
        "confusao": 3,
        "convuls": 3,
        "sangramento": 3,
        "febre": 1,
        "tontura": 1,
        "desmaio": 2,
    }
    for symptom in payload.sanitized_symptoms():
        for keyword, severity in symptom_flags.items():
            if keyword in symptom:
                add_alert("Symptoms", f"Reported symptom: {symptom}.", severity)
                if severity >= 3:
                    suggested_actions.append("Activate the emergency protocol.")
                break

    # Chronic diseases (matched against Portuguese chronic-condition input)
    chronic = [c.lower() for c in payload.chronic_conditions]
    if any(d in chronic for d in ("diabetes", "cardiopatia", "insuficiencia cardiaca")):
        suggested_exams.append("Check capillary blood glucose and a basic ECG.")
    if "asma" in chronic or "dpoc" in chronic:
        suggested_actions.append("Assess peak expiratory flow.")

    # Allergies/meds
    if payload.allergies:
        alerts.append("Recorded allergies: " + ", ".join(payload.allergies))
    if payload.medications:
        suggested_exams.append("Check interactions with continuous-use medications.")

    prioritized_actions = list(dict.fromkeys(suggested_actions))
    prioritized_exams = list(dict.fromkeys(suggested_exams))

    risk_label = RISK_LEVELS[risk_score]
    priority = {
        "Low": "Scheduled care",
        "Moderate": "Medical evaluation within 2h",
        "High": "Immediate medical care",
        "Critical": "Emergency / red room",
    }[risk_label]
    justificativa = (
        audit_trail[-1]["message"]
        if audit_trail
        else "No significant alerts; keep standard follow-up."
    )

    coleta = payload.to_dict()
    processamento = {
        "data_processamento": _dt.datetime.utcnow().isoformat() + "Z",
        "risco": risk_label,
        "pontuacao": risk_score,
        "audit_trail": audit_trail,
        "sugestoes_exames": prioritized_exams,
    }
    relatorio = {
        "risco": risk_label,
        "prioridade": priority,
        "alertas": alerts,
        "encaminhamentos": prioritized_actions or ["Keep the patient under routine observation."],
        "justificativa": justificativa,
    }

    return TriagePipelineOutput(coleta=coleta, processamento=processamento, relatorio=relatorio)


def example_triage_payload() -> Dict[str, Any]:
    return {
        "systolic": 188,
        "diastolic": 118,
        "heart_rate": 126,
        "temperature": 38.6,
        "spo2": 91,
        "age": 64,
        "sex": "Feminino",
        "chronic_conditions": ["Hipertensao", "Diabetes"],
        "allergies": ["Dipirona"],
        "blood_type": "O+",
        "medications": ["Losartana 50mg", "Metformina 850mg"],
        "symptoms": ["Dor no peito com irradiacao", "Falta de ar em repouso"],
    }

