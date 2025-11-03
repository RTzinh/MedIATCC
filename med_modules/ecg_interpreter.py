"""ECG and vital-sign interpreter scaffolding."""

from __future__ import annotations

import math
import struct
from typing import Any, Dict, List, Optional

try:
    import pyedflib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pyedflib = None


class ECGInterpreter:
    """Provide placeholder ECG metrics while enabling future ML integration."""

    def analyze_bytes(self, *, exam: Dict[str, Any], content: bytes) -> Optional[Dict[str, Any]]:
        ext = exam.get("ext", "").lower()
        if ext not in {".edf", ".hl7"}:
            return None
        if ext == ".edf":
            return self._analyze_edf(exam, content)
        return self._analyze_hl7(exam, content)

    def _analyze_edf(self, exam: Dict[str, Any], content: bytes) -> Optional[Dict[str, Any]]:
        if not pyedflib:
            return {
                "exam_id": exam.get("id"),
                "summary": "pyedflib ausente; nao foi possivel extrair derivacoes EDF.",
            }
        import tempfile  # local import to avoid overhead if module missing

        try:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                reader = pyedflib.EdfReader(tmp.name)  # type: ignore[call-arg]
                signal_labels = reader.getSignalLabels()
                sample_frequency = reader.getSampleFrequency(0)
                duration = reader.getFileDuration()
                reader.close()
            return {
                "exam_id": exam.get("id"),
                "lead_count": len(signal_labels),
                "sampling_rate": sample_frequency,
                "duration": duration,
                "summary": f"{len(signal_labels)} derivacoes EDF; SR={sample_frequency}Hz; T={duration:.1f}s.",
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "exam_id": exam.get("id"),
                "summary": f"Falha ao ler EDF: {exc}",
            }

    def _analyze_hl7(self, exam: Dict[str, Any], content: bytes) -> Optional[Dict[str, Any]]:
        snapshot: List[str] = []
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            return None
        for line in text.splitlines():
            if line.startswith("OBX") and "PR" in line:
                snapshot.append("Intervalo PR reportado via HL7.")
            if "QRS" in line.upper():
                snapshot.append("Segmento QRS identificado em mensagem HL7.")
        if not snapshot:
            return None
        return {
            "exam_id": exam.get("id"),
            "summary": "; ".join(snapshot),
        }

    @staticmethod
    def derive_basic_metrics(rr_intervals: Optional[List[float]]) -> Optional[Dict[str, Any]]:
        if not rr_intervals:
            return None
        mean_rr = sum(rr_intervals) / len(rr_intervals)
        heart_rate = 60000.0 / mean_rr if mean_rr else 0.0
        sdnn = math.sqrt(
            sum((interval - mean_rr) ** 2 for interval in rr_intervals) / len(rr_intervals)
        )
        return {
            "heart_rate": round(heart_rate, 1),
            "sdnn": round(sdnn, 2),
        }
