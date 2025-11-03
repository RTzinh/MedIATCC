"""Helper modules for MedIA advanced analytics.

These modules encapsulate optional, higher-complexity medical analysis
workflows (laboratory interpretation, imaging CAD stubs, guideline engines,
etc.) so that `app.py` remains focused on orchestration.
"""

from .hl7_parser import AdvancedLabInterpreter
from .dicom_analyzer import DICOMAnalyzer
from .cross_validator import MultiexamReasoner
from .ecg_interpreter import ECGInterpreter
try:
    from .guideline_engine import GuidelineAdvisor
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    GuidelineAdvisor = None  # type: ignore

__all__ = [
    name
    for name, obj in {
        "AdvancedLabInterpreter": AdvancedLabInterpreter,
        "DICOMAnalyzer": DICOMAnalyzer,
        "MultiexamReasoner": MultiexamReasoner,
        "ECGInterpreter": ECGInterpreter,
        "GuidelineAdvisor": GuidelineAdvisor,
    }.items()
    if obj is not None
]
