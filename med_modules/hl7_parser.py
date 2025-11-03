"""Advanced laboratory interpreter scaffolding.

This module provides a lightweight facade that can be extended to parse HL7 or
FHIR bundles, track laboratory time series, and detect clinically relevant
flags by demographic group.  The current implementation focuses on structure
and provides basic out-of-range detection using caller-supplied reference
intervals, while leaving hooks for downstream ML/analytics upgrades.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LabAlert:
    marker: str
    value: float
    reference: Tuple[float, float]
    severity: str
    rationale: str


class AdvancedLabInterpreter:
    """Parse structured lab payloads and surface longitudinal insights."""

    def __init__(self) -> None:
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def ingest_exam(
        self,
        exam: Dict[str, Any],
        *,
        lab_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
        demographics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Register an exam result and compute initial insights.

        Parameters
        ----------
        exam:
            Result dictionary produced by the streamlit sidebar pipeline. Must
            include ``id`` and either ``normalized`` or ``raw_text``.
        lab_ranges:
            Optional reference intervals keyed by analyte. Each value should be
            a mapping that includes ``low`` and ``high`` thresholds.
        demographics:
            Optional patient metadata (age, sex, pregnancy, etc.) for future
            personalization. Currently used only for labeling.
        """
        normalized = exam.get("normalized") or {}
        if not normalized:
            return None

        exam_id = exam.get("id", "")
        panel = normalized.get("panel") or exam.get("name") or "painel-desconhecido"

        alerts: List[LabAlert] = []
        if lab_ranges:
            alerts.extend(
                self._build_alerts(
                    normalized=normalized,
                    lab_ranges=lab_ranges,
                )
            )

        entry = {
            "exam_id": exam_id,
            "collected_at": normalized.get("collected_at")
            or datetime.utcnow().isoformat(),
            "panel": panel,
            "values": normalized,
            "alerts": [alert.__dict__ for alert in alerts],
        }
        self._history[panel].append(entry)

        trend_summary = self._summarize_trends(panel=panel, marker_alerts=alerts)
        demographics_note = ""
        if demographics:
            demographics_note = f"Perfil aplicado: {demographics}."

        return {
            "panel": panel,
            "exam_id": exam_id,
            "alerts": [alert.__dict__ for alert in alerts],
            "trend_summary": trend_summary,
            "demographics_note": demographics_note,
        }

    def _build_alerts(
        self,
        *,
        normalized: Dict[str, Any],
        lab_ranges: Dict[str, Dict[str, Any]],
    ) -> List[LabAlert]:
        alerts: List[LabAlert] = []
        for marker, meta in lab_ranges.items():
            raw_value = normalized.get(marker)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            low = meta.get("low")
            high = meta.get("high")
            if low is None or high is None:
                continue
            severity = "moderado"
            rationale = ""
            if value < low:
                rationale = f"Valor {value} abaixo do intervalo ({low}-{high})."
            elif value > high:
                rationale = f"Valor {value} acima do intervalo ({low}-{high})."
            if rationale:
                if abs(value - (high if value > high else low)) / (high - low + 1e-6) > 0.6:
                    severity = "alto"
                alerts.append(
                    LabAlert(
                        marker=marker,
                        value=value,
                        reference=(low, high),
                        severity=severity,
                        rationale=rationale,
                    )
                )
        return alerts

    def _summarize_trends(
        self,
        *,
        panel: str,
        marker_alerts: List[LabAlert],
    ) -> List[str]:
        if not marker_alerts:
            return []
        history = self._history.get(panel, [])
        if len(history) < 2:
            return []
        previous = history[-2]
        trends: List[str] = []
        prev_values = previous.get("values", {})
        for alert in marker_alerts:
            past_value = prev_values.get(alert.marker)
            if past_value is None:
                continue
            try:
                past_float = float(past_value)
            except (TypeError, ValueError):
                continue
            delta = alert.value - past_float
            direction = "aumentou" if delta > 0 else "reduziu"
            trends.append(
                f"{alert.marker} {direction} {abs(delta):.2f} desde o exame anterior."
            )
        return trends

    def get_history_snapshot(self) -> Dict[str, Any]:
        """Expose a condensed version of the stored trajectories."""
        snapshot: Dict[str, Any] = {}
        for panel, entries in self._history.items():
            snapshot[panel] = [
                {
                    "exam_id": entry["exam_id"],
                    "collected_at": entry["collected_at"],
                    "alerts": entry["alerts"],
                }
                for entry in entries[-5:]
            ]
        return snapshot

    def reset(self) -> None:
        self._history.clear()
