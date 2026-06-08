"""DICOM and pseudo-CAD scaffolding module.

The goal is to provide a structured entry-point for ingesting radiology assets
submitted as raw DICOM files or zipped studies, extracting metadata, and
running placeholder heuristics that can later be replaced by a vetted CAD
model.  For now, we compute hashes and surface essential tags when pydicom is
available, while warning the caller if advanced dependencies are missing.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from typing import Any, Dict, List, Optional

try:
    import pydicom  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pydicom = None


class DICOMAnalyzer:
    """Inspect DICOM payloads and surface structured findings."""

    def __init__(self, external_handler: Optional[Any] = None) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._external_handler = external_handler

    def analyze_bytes(self, *, file_id: str, content: bytes) -> Dict[str, Any]:
        if file_id in self._cache:
            return self._cache[file_id]

        meta_summary: List[str] = []
        cad_flags: List[str] = []
        frames: int = 0

        if self._is_zip(content):
            extracted = self._process_zip(content)
            meta_summary.extend(extracted.get("meta_summary", []))
            cad_flags.extend(extracted.get("cad_flags", []))
            frames = extracted.get("frames", 0)
        elif pydicom:
            details = self._process_dicom(content)
            meta_summary.extend(details.get("meta_summary", []))
            cad_flags.extend(details.get("cad_flags", []))
            frames = details.get("frames", 0)
        else:
            meta_summary.append(
                "pydicom not installed; analysis limited to a heuristic checksum."
            )

        digest = hashlib.sha256(content).hexdigest()
        confidence = round(int(digest[:2], 16) / 255, 2)
        cad_flags.append(
            f"Prioritization heuristic: score {confidence}; review manually."
        )

        payload = {
            "id": file_id,
            "meta_summary": meta_summary,
            "cad_flags": cad_flags,
            "frames": frames,
            "hash": digest[:16],
        }
        if self._external_handler:
            try:
                external_payload = self._external_handler(
                    file_id=file_id,
                    content=content,
                    metadata=dict(payload),
                )
                if isinstance(external_payload, dict):
                    payload.update(
                        {
                            key: external_payload.get(key, payload.get(key))
                            for key in ["meta_summary", "cad_flags", "frames", "hash", "extras"]
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive
                payload.setdefault("cad_flags", []).append(f"External service failed: {exc}")
        self._cache[file_id] = payload
        return payload

    def register_external_handler(self, handler: Any) -> None:
        """Allow callers to plug an external CAD pipeline."""
        self._external_handler = handler

    def _is_zip(self, content: bytes) -> bool:
        return content[:2] == b"PK"

    def _process_zip(self, content: bytes) -> Dict[str, Any]:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.namelist()
            meta_summary = [f"ZIP study with {len(members)} objects."]
            cad_flags: List[str] = []
            frames = 0
            if pydicom:
                for name in members[:5]:
                    if not name.lower().endswith((".dcm", ".dicom")):
                        continue
                    try:
                        raw = archive.read(name)
                        details = self._process_dicom(raw)
                        meta_summary.extend(details.get("meta_summary", []))
                        cad_flags.extend(details.get("cad_flags", []))
                        frames += details.get("frames", 0)
                    except Exception as exc:  # pragma: no cover - defensive
                        cad_flags.append(f"Failed to parse {name}: {exc}")
            else:
                cad_flags.append("Install `pydicom` for detailed DICOM reading.")
            return {
                "meta_summary": meta_summary,
                "cad_flags": cad_flags,
                "frames": frames,
            }

    def _process_dicom(self, content: bytes) -> Dict[str, Any]:
        meta_summary: List[str] = []
        cad_flags: List[str] = []
        frames = 0
        if not pydicom:
            return {
                "meta_summary": ["pydicom unavailable; native reading disabled."],
                "cad_flags": [],
                "frames": 0,
            }
        try:
            dataset = pydicom.dcmread(io.BytesIO(content))
            patient_position = getattr(dataset, "PatientPosition", "N/A")
            study_desc = getattr(dataset, "StudyDescription", "No description")
            series_desc = getattr(dataset, "SeriesDescription", "No series")
            modality = getattr(dataset, "Modality", "N/A")
            rows = getattr(dataset, "Rows", 0)
            cols = getattr(dataset, "Columns", 0)
            frames = getattr(dataset, "NumberOfFrames", 1) or 1

            meta_summary.extend(
                [
                    f"Mod: {modality} | Study: {study_desc} | Series: {series_desc}",
                    f"Positioning: {patient_position} | {rows}x{cols}px | frames={frames}",
                ]
            )

            if modality == "CR" and rows and cols:
                if rows < 1500 or cols < 1500:
                    cad_flags.append("Resolution below the chest X-ray standard (1500px).")
                if rows > cols:
                    cad_flags.append("Projection suggests AP/PA; correlate with the clinical report.")
            if modality in {"CT", "MR"}:
                cad_flags.append("Volumetric study detected; CAD not implemented.")
        except Exception as exc:  # pragma: no cover - defensive
            cad_flags.append(f"DICOM parsing failed: {exc}")
        return {
            "meta_summary": meta_summary,
            "cad_flags": cad_flags,
            "frames": frames,
        }
