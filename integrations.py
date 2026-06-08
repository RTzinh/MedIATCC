"""Centralize optional integrations with external CAD and HL7/FHIR services.

Customize `get_lab_pipeline` and `get_cad_handler` to return callables that
implement your production logic. Each function should return either `None`
(meaning the default heuristics remain active) or a callable with the
signature documented below.

- Lab pipeline signature::

      def pipeline(*, exam, normalized, alerts, demographics) -> Dict[str, Any]

  * `exam`: raw exam payload created by the upload pipeline.
  * `normalized`: dict with structured lab values.
  * `alerts`: list of previously detected alerts (each item a dict).
  * `demographics`: dict with patient data (age, sex, etc.).
  * return: dict with optional fields `notes`, `alerts`, `extras`.

- CAD handler signature::

      def handler(*, file_id, content, metadata) -> Dict[str, Any]

  * `file_id`: file/image identifier.
  * `content`: original binary bytes of the DICOM/zip.
  * `metadata`: heuristic payload computed by the app.
  * return: dict with keys `meta_summary`, `cad_flags`, `frames`, `hash`,
    and optionally `extras`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

LabPipeline = Callable[..., Dict[str, Any]]
CadHandler = Callable[..., Dict[str, Any]]


def get_lab_pipeline() -> Optional[LabPipeline]:
    """Return a custom HL7/FHIR pipeline callable or ``None``."""
    return None


def get_cad_handler() -> Optional[CadHandler]:
    """Return a custom CAD handler callable or ``None``."""
    return None
