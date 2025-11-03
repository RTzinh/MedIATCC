"""Cross-exam validation scaffolding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MultiexamReasoner:
    """Aggregate laboratory, imaging and wearable evidence."""

    def synthesize(
        self,
        *,
        lab_alerts: Optional[List[Dict[str, Any]]] = None,
        imaging_alerts: Optional[List[Dict[str, Any]]] = None,
        wearable_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        notes: List[str] = []
        conflicting: List[str] = []

        if lab_alerts:
            for alert in lab_alerts[-5:]:
                markers = ", ".join(
                    f"{item['marker']} ({item['severity']})"
                    for item in alert.get("alerts", [])
                    if isinstance(item, dict)
                )
                if markers:
                    notes.append(f"Painel {alert.get('panel')}: {markers}.")
                for trend in alert.get("trend_summary", []):
                    notes.append(trend)

        if imaging_alerts:
            for finding in imaging_alerts[-3:]:
                cad_flags = finding.get("cad_flags") or []
                if cad_flags:
                    notes.extend(cad_flags)
                meta = finding.get("meta_summary") or []
                notes.extend(meta)

        if wearable_snapshot:
            hr = wearable_snapshot.get("heart_rate")
            spo2 = wearable_snapshot.get("spo2")
            if hr and spo2 and isinstance(hr, (int, float)) and isinstance(spo2, (int, float)):
                if hr > 110 and spo2 < 92:
                    conflicting.append(
                        "Taquicardia com queda de SpO₂ sugerem investigar causas respiratorias."
                    )
            variability = wearable_snapshot.get("hrv")
            if variability and isinstance(variability, (int, float)) and variability < 20:
                notes.append(
                    "Variabilidade da frequencia cardiaca reduzida; avaliar estresse autonomico."
                )

        return {
            "notes": notes,
            "conflicts": conflicting,
        }
