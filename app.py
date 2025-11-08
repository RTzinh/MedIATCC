import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
try:
    from groq import BadRequestError, Groq
except ModuleNotFoundError as exc:
    Groq = Any  # type: ignore
    BadRequestError = Exception  # type: ignore
    _groq_import_error: Optional[Exception] = exc
else:
    _groq_import_error = None

try:
    import PyPDF2  # type: ignore
except ImportError:  # pragma: no cover
    PyPDF2 = None

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

try:
    from gtts import gTTS  # type: ignore
except ImportError:  # pragma: no cover
    gTTS = None

try:
    from google import genai as google_genai  # type: ignore
    from google.genai import types as google_genai_types  # type: ignore
except ImportError:  # pragma: no cover
    google_genai = None
    google_genai_types = None

try:
    import google.generativeai as legacy_genai  # type: ignore
except ImportError:  # pragma: no cover
    legacy_genai = None

try:
    import qrcode  # type: ignore
except ImportError:  # pragma: no cover
    qrcode = None

try:
    from med_modules import (
        AdvancedLabInterpreter,
        DICOMAnalyzer,
        MultiexamReasoner,
        ECGInterpreter,
        GuidelineAdvisor,
    )
except ModuleNotFoundError:  # pragma: no cover - defensive fallback
    AdvancedLabInterpreter = DICOMAnalyzer = MultiexamReasoner = ECGInterpreter = GuidelineAdvisor = None  # type: ignore

try:
    from integrations import get_lab_pipeline, get_cad_handler
except ModuleNotFoundError:  # pragma: no cover - optional integrations
    def get_lab_pipeline() -> Optional[Any]:
        return None

    def get_cad_handler() -> Optional[Any]:
        return None

try:
    from hackathon import (
        NursingTriageInput,
        example_triage_payload,
        generate_triage_report,
    )
except ModuleNotFoundError:  # pragma: no cover - optional module
    NursingTriageInput = Any  # type: ignore

    def example_triage_payload() -> Dict[str, Any]:  # type: ignore
        return {}

    def generate_triage_report(_: Any) -> Dict[str, Any]:  # type: ignore
        return {}

def safe_show_image(image: Any, caption: Optional[str] = None, **kwargs: Any) -> None:
    """Wrapper to avoid Streamlit media cache errors when the file is missing."""
    if image is None:
        return
    if isinstance(image, str) and not os.path.exists(image):
        st.warning("Imagem não está mais disponível nesta sessão.")
        return
    try:
        st.image(image, caption=caption, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive fallback
        st.warning(f"Não foi possível exibir a imagem: {exc}")


def default_hl7_pipeline(
    *,
    exam: Dict[str, Any],
    normalized: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    demographics: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub external HL7/FHIR pipeline hook; replace with production logic."""
    notes: List[str] = []
    extra_alerts: List[Dict[str, Any]] = []
    age = demographics.get("age")
    troponin = normalized.get("troponina")
    if troponin is not None:
        try:
            troponin_value = float(troponin)
            threshold = 34.0 if demographics.get("sex") == "M" else 16.0
            if troponin_value > threshold:
                notes.append("Troponina acima do limite específico por sexo.")
                extra_alerts.append(
                    {
                        "marker": "troponina",
                        "value": troponin_value,
                        "severity": "alto",
                        "reference": [0, threshold],
                        "rationale": "Elevação detectada pela pipeline externa.",
                    }
                )
        except (TypeError, ValueError):
            pass
    hemoglobina = normalized.get("hemoglobina")
    if hemoglobina is not None and age and age > 65:
        try:
            hb_value = float(hemoglobina)
            if hb_value < 11:
                notes.append("Hemoglobina baixa em idoso; sugerir investigação de anemia.")
        except (TypeError, ValueError):
            pass
    return {"notes": notes, "alerts": extra_alerts}


def default_cad_handler(
    *,
    file_id: str,
    content: bytes,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub CAD handler; replace with integration to imaging service."""
    updated_meta = list(metadata.get("meta_summary") or [])
    updated_meta.append("Laudo CAD externo não configurado; exibindo heurística local.")
    updated_flags = list(metadata.get("cad_flags") or [])
    updated_flags.append("Integração CAD aguardando serviço externo.")
    return {
        "meta_summary": updated_meta,
        "cad_flags": updated_flags,
        "frames": metadata.get("frames"),
        "hash": metadata.get("hash"),
        "extras": {"handler": "default_stub"},
    }


def register_external_services() -> None:
    """Configure CAD/HL7 integrations, preferindo handlers externos quando presentes."""
    lab_engine = st.session_state.get("lab_interpreter")
    if lab_engine:
        custom_pipeline = get_lab_pipeline()
        if custom_pipeline:
            lab_engine.register_external_pipeline(custom_pipeline)
            st.session_state.lab_pipeline_registered = True
        elif not st.session_state.get("lab_pipeline_registered"):
            lab_engine.register_external_pipeline(default_hl7_pipeline)
            st.session_state.lab_pipeline_registered = True

    cad_engine = st.session_state.get("dicom_analyzer")
    if cad_engine:
        custom_handler = get_cad_handler()
        if custom_handler:
            cad_engine.register_external_handler(custom_handler)
            st.session_state.cad_handler_registered = True
        elif not st.session_state.get("cad_handler_registered"):
            cad_engine.register_external_handler(default_cad_handler)
            st.session_state.cad_handler_registered = True


DEFAULT_GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or "AIzaSyCag_eYIGTTZfw-xSUw8iERcNuroOZO7G4"
)
DEFAULT_GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL_NAME",
    "models/gemini-2.5-flash",
)
VOICE_AGENT_INSTRUCTIONS = (
    "Voce e o agente de voz do MedIA. Transcreva falas em portugues e ofereca orientacoes medicas de apoio, "
    "reforcando que o paciente deve buscar atendimento presencial em situacoes criticas. Use linguagem simples, "
    "acolhedora e sem jargoes. Quando nao tiver informacoes suficientes, faca perguntas abertas para entender melhor."
)
VOICE_AGENT_RESPONSE_SCHEMA = (
    "Retorne um JSON valido com as chaves 'transcription' e 'assistant_reply'. "
    "Em 'transcription' escreva a transcricao literal do paciente. Em 'assistant_reply' forneca a resposta do agente."
)

QUICK_PROMPTS = [
    {
        "label": "🌡️ Febre alta",
        "text": "Estou com febre acima de 39 °C ha dois dias mesmo tomando antitermicos. Quais sinais indicam emergencia?",
    },
    {
        "label": "❤️ Dor no peito",
        "text": "Sinto dor aguda no peito que irradia para o braco esquerdo acompanhada de suor frio. O que devo observar?",
    },
    {
        "label": "💊 Medicacoes",
        "text": "Quais cuidados preciso ter ao combinar ibuprofeno com dipirona e um anti-hipertensivo?",
    },
    {
        "label": "📝 Resumo",
        "text": "Pode resumir nossa conversa e listar exames ou orientacoes que devo seguir nas proximas 24h?",
    },
]

COMMON_CHRONIC_CONDITIONS = [
    "Hipertensão",
    "Diabetes",
    "Asma",
    "DPOC",
    "Cardiopatia",
    "Insuficiencia cardiaca",
]
COMMON_ALLERGIES = [
    "Dipirona",
    "Ibuprofeno",
    "Amoxicilina",
    "Penicilina",
    "Losartana",
]
COMMON_MEDICATIONS = [
    "Losartana 50mg",
    "Metformina 850mg",
    "AAS 100mg",
    "Atorvastatina 20mg",
]


def parse_free_text_list(value: str) -> List[str]:
    tokens = re.split(r"[,;\n]+", value or "")
    return [item.strip() for item in tokens if item.strip()]


def sanitize_numeric(value: float) -> Optional[float]:
    if value and value > 0:
        return float(value)
    return None


def gemini_sdk_available() -> bool:
    return google_genai is not None or legacy_genai is not None


CLINICAL_DISCLAIMER = (
    "Aviso: interpretacoes automatizadas complementam o parecer medico humano e nao substituem atendimento presencial."
)

CRITICAL_KEYWORDS = {
    "dor toracica intensa",
    "falta de ar severa",
    "tiro",
    "golpe de faca",
    "sangramento intenso",
    "desmaio prolongado",
    "infarto",
    "avc",
    "convulsao continua",
}

MEDICATION_DATABASE = {
    "ibuprofeno": {
        "interactions": ["Pode reduzir o efeito de anti-hipertensivos"],
        "contra": ["Evitar em doenca renal avancada", "Usar com cautela em gastrite ativa"],
    },
    "dipirona": {
        "interactions": ["Potencializa anti-hipertensivos e anticoagulantes"],
        "contra": ["Historico de agranulocitose", "Alergia a pirazolonas"],
    },
    "paracetamol": {
        "interactions": ["Uso combinado com alcool aumenta risco hepatico"],
        "contra": ["Doenca hepatica grave sem acompanhamento"],
    },
    "amoxicilina": {
        "interactions": ["Pode reduzir eficacia de contraceptivos orais"],
        "contra": ["Alergia a penicilinas"],
    },
}

STOPWORDS = {
    "com", "pra", "para", "nos", "das", "dos", "uma", "num", "que", "qual", "pelo", "pela",
    "das", "dos", "das", "e", "de", "do", "da", "um", "uma", "em", "no", "na", "os", "as",
    "por", "sem", "ser", "ter", "vai", "vou", "tem", "dor", "estou", "mais", "menos",
}

SYMPTOM_HINTS = {
    "dor",
    "febre",
    "vomito",
    "náusea",
    "nausea",
    "ansia",
    "ansiedade",
    "calafrio",
    "calafrios",
    "tosse",
    "fadiga",
    "fraqueza",
    "tontura",
    "diarreia",
    "diarre",
    "coceira",
    "mancha",
    "inchaco",
    "inchaco",
    "ardor",
    "queima",
    "quente",
    "frio",
    "resfriado",
    "gripado",
    "sangramento",
    "dor de cabeça",
    "cefaléia",
    "cefaleia",
}

EDUCATION_LIBRARY = {
    "hipertensao": [
        {
            "title": "Guia pratico de controle da pressao arterial",
            "type": "video",
            "url": "https://www.youtube.com/watch?v=93dnup_pressao",
        },
        {
            "title": "Infografico: reducao de sodio na dieta",
            "type": "infografico",
            "url": "https://www.saude.gov/infografico-sodio.pdf",
        },
    ],
    "diabetes": [
        {
            "title": "Video educativo sobre autocuidado em diabetes tipo 2",
            "type": "video",
            "url": "https://www.youtube.com/watch?v=84edu_diabetes",
        },
        {
            "title": "Checklist de monitoramento glicemico domiciliar",
            "type": "folheto",
            "url": "https://www.consultorio.ai/assets/folheto-monitoramento.pdf",
        },
    ],
    "saude mental": [
        {
            "title": "Tecnicas de respiracao para ansiedade",
            "type": "audio",
            "url": "https://www.saude.gov/respiracao-guiada.mp3",
        },
        {
            "title": "Cartilha de sinais de alerta em depressao",
            "type": "folheto",
            "url": "https://www.saude.gov/cartilha-depressao.pdf",
        },
    ],
    "puerperio": [
        {
            "title": "Cuidados pos-parto imediato",
            "type": "video",
            "url": "https://www.youtube.com/watch?v=84puerperio",
        },
        {
            "title": "Infografico: amamentacao segura",
            "type": "infografico",
            "url": "https://www.saude.gov/infografico-amamentacao.pdf",
        },
    ],
}

THEME_STYLES = {
    "Claro": {
        "bg": "#f7fbff",
        "panel": "#ffffff",
        "accent": "#2b6cb0",
        "text": "#1a202c",
        "accent_soft": "#dbeafe",
    },
    "Escuro": {
        "bg": "#0f172a",
        "panel": "#1e293b",
        "accent": "#38bdf8",
        "text": "#f8fafc",
        "accent_soft": "#1e3a8a",
    },
    "Clinico": {
        "bg": "#edf7f6",
        "panel": "#ffffff",
        "accent": "#0f766e",
        "text": "#0b1120",
        "accent_soft": "#c8f3ed",
    },
}

ICON_EMERGENCY = "[URG]"
ICON_EXAM = "[EXAME]"
ICON_INTERACTION = "[FARMA]"
ICON_WEARABLE = "[WEAR]"
ICON_EDUCATION = "[EDU]"
ICON_STEP = "[TRIAGEM]"
ICON_IMAGING = "[IMAGEM]"

MEDICATION_DISCLAIMER = (
    "Estas informacoes nao substituem orientacao medica presencial. Consulte seu medico ou farmacêutico."
)

MEDICATION_MONOGRAPHS = {
    "roacutan": {
        "aliases": ["isotretinoina", "isotretinoína", "acutane"],
        "class": "Retinoide oral para tratamento de acne nodular grave",
        "indications": [
            "Acne nodular, conglobata ou resistente a outros tratamentos",
        ],
        "contra": [
            "Gravidez e lactacao (teratogenico)",
            "Insuficiencia hepatica grave",
            "Hiperlipidemia nao controlada",
            "Uso concomitante de tetraciclinas",
        ],
        "warnings": [
            "Obrigatorio programa de contracepcao eficaz em pacientes em idade fertil",
            "Monitorar transaminases e lipideos periodicamente",
            "Pode causar ressecamento cutaneo, labial e ocular",
            "Risco de alteracoes psiquiatricas (depressao, ideacao suicida)",
        ],
        "dose": "0,5 a 1 mg/kg/dia, via oral, fracionada em 1 a 2 tomadas com alimentos; duracao media de 16 a 24 semanas",
        "references": [
            "Bula oficial: https://www.anvisa.gov.br/datavisa/fila_bula/frmVisualizarBula.asp?pNuTransacao=xxxxx",
            "Sociedade Brasileira de Dermatologia - Diretrizes de Acne",
        ],
    },
    "ibuprofeno": {
        "aliases": ["ibupr", "brufen", "advil", "motrin"],
        "class": "Anti-inflamatorio nao esteroide (AINE)",
        "indications": [
            "Dor leve a moderada (cefaleia, dor muscular, pos-operatorio, dismenorreia)",
            "Processos inflamatorios agudos",
            "Febre",
        ],
        "contra": [
            "Ulcera peptica ativa ou historico recorrente",
            "Insuficiencia renal grave",
            "Insuficiencia hepatica grave",
            "Terceiro trimestre de gestacao",
            "Asma sensivel a AINEs",
        ],
        "warnings": [
            "Usar a menor dose eficaz pelo menor tempo possivel",
            "Pode elevar pressao arterial e reduzir eficacia de anti-hipertensivos",
            "Associado a risco de eventos cardiovasculares e gastrointestinais",
            "Monitorar funcao renal em pacientes idosos ou com doencas pre-existentes",
        ],
        "dose": "200 a 400 mg por via oral a cada 6-8 horas conforme necessidade; dose maxima usual 1200 mg/dia sem supervisao medica",
        "references": [
            "Bula oficial: https://consultas.anvisa.gov.br/#/bulario/q/?nomeProduto=IBUPROFENO",
            "Diretrizes Sociedade Brasileira de Pediatria - Uso de AINEs",
        ],
    },
    "dipirona": {
        "aliases": ["metamizol", "novalgina", "dipirona sodica", "dipirona sódica"],
        "class": "Analgésico e antipirético não opioide",
        "indications": [
            "Dor leve a moderada (cefaleia, dor muscular, pos-operatorio)",
            "Febre refrataria a outros antipireticos",
        ],
        "contra": [
            "Hipersensibilidade a pirazolonas ou pirazolidinas",
            "Historico de agranulocitose induzida por dipirona",
            "Deficiencia de glicose-6-fosfato desidrogenase",
            "Terceiro trimestre de gestacao (risco de fechamento precoce do ducto arterioso)",
        ],
        "warnings": [
            "Monitorar sinais de reacoes hematologicas (agranulocitose, leucopenia)",
            "Pode causar reacoes anafilaticas graves; usar com cautela em pacientes asmáticos",
            "Evitar uso cronico sem supervisao medica",
        ],
        "dose": "500 mg a 1 g por via oral a cada 6-8 horas conforme necessidade; dose maxima diaria 4 g em adultos",
        "references": [
            "Bula oficial: https://consultas.anvisa.gov.br/#/bulario/q/?nomeProduto=DIPIRONA",
            "Manual de condutas clínicas - Ministério da Saúde",
        ],
    },
}


def ensure_session_defaults() -> None:
    defaults = {
        "history": [],
        "exam_findings": [],
        "exam_ids": set(),
        "imaging_findings": [],
        "imaging_ids": set(),
        "medication_alerts": [],
        "critical_events": [],
        "wearable_payload": {},
        "wearable_raw_input": "",
        "consultation_log": [],
        "pending_voice_input": "",
        "audio_responses": [],
        "education_recommendations": [],
        "epidemiology_snapshot": {},
        "active_learning_queue": [],
        "planner_state": {},
        "multimodal_signature": {},
        "confidence_history": [],
        "explainability_notes": [],
        "symptom_log": [],
        "theme": "Claro",
        "font_scale": 1.0,
        "advanced_lab_findings": [],
        "advanced_lab_ids": set(),
        "dicom_findings": [],
        "dicom_ids": set(),
        "cross_validation_notes": [],
        "cross_conflicts": [],
        "ecg_insights": [],
        "ecg_ids": set(),
        "guideline_recommendations": [],
        "guideline_plan": [],
        "guideline_cha2ds2": None,
        "guideline_curb65": None,
        "demographics": {},
        "lab_pipeline_registered": False,
        "cad_handler_registered": False,
        "question_progress": {
            "total": 10,
            "answered": 0,
            "current": 1,
            "history": [],
        },
        "education_checklist": {},
        "memory_window": 60,
        "memory_messages": [],
        "printable_summary": "",
        "dashboard_tab": "Sintomas",
        "triage_mode": False,
        "voice_conversation": [],
        "voice_agent_status": "",
        "hackathon_triage_report": None,
        "hackathon_triage_payload": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "exam_pipeline" not in st.session_state:
        st.session_state.exam_pipeline = ExamPipeline()
    if "radiography_service" not in st.session_state:
        st.session_state.radiography_service = RadiographyService()
    if "validator" not in st.session_state:
        st.session_state.validator = ClinicalValidator()
    if "med_checker" not in st.session_state:
        st.session_state.med_checker = MedicationInteractionChecker()
    if "emergency_router" not in st.session_state:
        st.session_state.emergency_router = EmergencyRouter()
    if "history_manager" not in st.session_state:
        st.session_state.history_manager = HistoryManager()
    if "education_manager" not in st.session_state:
        st.session_state.education_manager = EducationResourceManager(EDUCATION_LIBRARY)
    if "epidemiology_monitor" not in st.session_state:
        st.session_state.epidemiology_monitor = EpidemiologyMonitor()
    if "active_learning_tracker" not in st.session_state:
        st.session_state.active_learning_tracker = ActiveLearningTracker()
    if "fusion_engine" not in st.session_state:
        st.session_state.fusion_engine = MultimodalFusionEngine()
    if "confidence_calibrator" not in st.session_state:
        st.session_state.confidence_calibrator = ConfidenceCalibrator()
    if "conversation_planner" not in st.session_state:
        st.session_state.conversation_planner = ConversationPlanner()
    if "explainability_engine" not in st.session_state:
        st.session_state.explainability_engine = ExplainabilityEngine()
    if "lab_interpreter" not in st.session_state and AdvancedLabInterpreter:
        st.session_state.lab_interpreter = AdvancedLabInterpreter()
    if "dicom_analyzer" not in st.session_state and DICOMAnalyzer:
        st.session_state.dicom_analyzer = DICOMAnalyzer()
    if "cross_reasoner" not in st.session_state and MultiexamReasoner:
        st.session_state.cross_reasoner = MultiexamReasoner()
    if "ecg_interpreter" not in st.session_state and ECGInterpreter:
        st.session_state.ecg_interpreter = ECGInterpreter()
    if "guideline_advisor" not in st.session_state and GuidelineAdvisor:
        st.session_state.guideline_advisor = GuidelineAdvisor()


class ExamPipeline:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def process(self, uploaded_file) -> Optional[Dict[str, Any]]:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if file_id in self._cache:
            return None
        content = uploaded_file.read()
        uploaded_file.seek(0)
        ext = Path(uploaded_file.name).suffix.lower()
        raw_text, notes = self._extract_content(content, ext)
        normalized = self._normalize(raw_text, ext)
        result = {
            "id": file_id,
            "name": uploaded_file.name,
            "ext": ext,
            "notes": notes,
            "raw_text": raw_text[:5000],
            "normalized": normalized,
            "binary_preview": content[:200000]
            if ext in {".edf", ".hl7", ".zip", ".dcm"}
            else b"",
        }
        self._cache[file_id] = result
        return result

    def render_for_prompt(self, findings: List[Dict[str, Any]]) -> str:
        if not findings:
            return ""
        lines: List[str] = []
        for item in findings:
            lines.append(f"{item['name']} ({item['ext']}):")
            if item["normalized"]:
                serialized = json.dumps(item["normalized"])[:800]
                lines.append(serialized)
            elif item["raw_text"]:
                lines.append(item["raw_text"][:800])
            if item["notes"]:
                lines.append(f"Notas: {'; '.join(item['notes'])}")
        return "\n".join(lines)

    def _extract_content(self, content: bytes, ext: str) -> tuple[str, List[str]]:
        notes: List[str] = []
        text = ""
        if ext == ".csv":
            text = content.decode("utf-8", errors="ignore")
        elif ext == ".json":
            text = content.decode("utf-8", errors="ignore")
        elif ext == ".hl7":
            text = content.decode("utf-8", errors="ignore")
        elif ext == ".pdf":
            if PyPDF2:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
            else:
                notes.append("PyPDF2 ausente; instale para extrair texto de PDF.")
        elif ext in {".png", ".jpg", ".jpeg"}:
            if Image and pytesseract:
                image = Image.open(io.BytesIO(content))
                text = pytesseract.image_to_string(image, lang="por+eng")
            else:
                notes.append("OCR indisponivel; instale pillow e pytesseract.")
        else:
            text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            notes.append("Conteudo nao extraido; revise manualmente.")
        return text, notes

    def _normalize(self, text: str, ext: str) -> Dict[str, Any]:
        if not text.strip():
            return {}
        if ext == ".json":
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                pass
        if ext == ".hl7":
            return self._parse_hl7(text)
        return self._parse_key_values(text.splitlines())

    def _parse_key_values(self, lines: List[str]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key_norm = re.sub(r"[^a-z0-9_]+", "_", key.lower().strip())
                data[key_norm] = value.strip()
        return data

    def _parse_hl7(self, text: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for segment in text.split("\n"):
            parts = segment.split("|")
            if not parts:
                continue
            tag = parts[0].strip()
            data[tag] = parts[1:]
        return data


class RadiographyService:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("RADIOLOGY_SERVICE_URL")
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, uploaded_file) -> Optional[Dict[str, Any]]:
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if file_id in self._cache:
            return None
        content = uploaded_file.read()
        uploaded_file.seek(0)
        payload, notes = self._call_service(content, uploaded_file.type)
        result = {
            "id": file_id,
            "name": uploaded_file.name,
            "payload": payload,
            "notes": notes,
        }
        self._cache[file_id] = result
        return result

    def render_for_prompt(self, findings: List[Dict[str, Any]]) -> str:
        if not findings:
            return ""
        sections: List[str] = []
        for item in findings:
            sections.append(f"Radiografia {item['name']}:")
            payload = item["payload"]
            impressions = payload.get("impressions") or []
            probabilities = payload.get("probabilities") or {}
            recommendations = payload.get("recommendations") or []
            if impressions:
                sections.append("Achados: " + ", ".join(impressions))
            if probabilities:
                formatted = ", ".join(
                    f"{label}={value:.2f}" for label, value in probabilities.items()
                )
                sections.append("Probabilidades: " + formatted)
            if recommendations:
                sections.append("Recomendacoes: " + "; ".join(recommendations))
            if item["notes"]:
                sections.append(f"Notas: {'; '.join(item['notes'])}")
        return "\n".join(sections)

    def _call_service(self, content: bytes, mime: Optional[str]) -> tuple[Dict[str, Any], List[str]]:
        notes: List[str] = []
        if self.base_url and requests:
            try:
                response = requests.post(
                    self.base_url,
                    files={"file": ("radiografia", content, mime or "application/octet-stream")},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                return payload, notes
            except Exception as exc:  # pragma: no cover
                notes.append(f"Falha no microservico: {exc}")
        fallback_score = int(hashlib.md5(content).hexdigest()[:2], 16) / 255
        payload = {
            "impressions": ["Classificacao heuristica - modo demonstrativo"],
            "probabilities": {"achado_indeterminado": round(fallback_score, 2)},
            "recommendations": [
                "Encaminhar laudo para radiologista humano.",
                "Correlacionar achados com exame clinico.",
            ],
        }
        if not self.base_url:
            notes.append("Microservico externo nao configurado; usando heuristica local.")
        elif not requests:
            notes.append("Biblioteca requests ausente; instale para chamar o microservico.")
        return payload, notes


class ClinicalValidator:
    def attach_disclaimer(self, response: str, context: Dict[str, Any]) -> str:
        disclaimers = [CLINICAL_DISCLAIMER]
        if context.get("critical_flags"):
            disclaimers.append(
                "Sinais de alerta detectados: " + ", ".join(sorted(set(context["critical_flags"])))
            )
        if context.get("medication_alerts"):
            disclaimers.append(
                "Alerta farmacologico: " + "; ".join(sorted(set(context["medication_alerts"])))
            )
        return f"{response}\n\n{' '.join(disclaimers)}"


class MedicationInteractionChecker:
    def check(self, text: str) -> List[str]:
        found: List[str] = []
        lowered = text.lower()
        for name, data in MEDICATION_DATABASE.items():
            if name in lowered:
                found.extend(f"{name}: {msg}" for msg in data["interactions"])
                found.extend(f"{name} contraindicacao: {msg}" for msg in data["contra"])
        return found


class EmergencyRouter:
    def detect(self, text: str) -> Optional[str]:
        lowered = text.lower()
        triggered = [kw for kw in CRITICAL_KEYWORDS if kw in lowered]
        if triggered:
            return (
                "Possivel emergencia identificada ("
                + ", ".join(triggered)
                + "). Orientar paciente a acionar 192 (SAMU) ou 190 imediatamente."
            )
        return None


class HistoryManager:
    def save(self, history: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._summarize(history)
        record = {
            "summary": summary,
            "exam_ids": [item["id"] for item in context.get("exam_findings", [])],
            "imaging_ids": [item["id"] for item in context.get("imaging_findings", [])],
            "medication_alerts": context.get("medication_alerts", []),
        }
        st.session_state.consultation_log.append(record)
        return record

    def _summarize(self, history: List[str]) -> str:
        if not history:
            return "Sem mensagens registradas."
        text = "\n".join(self._strip_html(item) for item in history[-20:])
        if len(text) > 800:
            return text[:800] + "..."
        return text

    def _strip_html(self, value: str) -> str:
        clean = re.sub(r"<[^>]+>", "", value)
        return clean.replace("&nbsp;", " ").strip()


class EducationResourceManager:
    def __init__(self, library: Dict[str, List[Dict[str, str]]]) -> None:
        self.library = library

    def recommend_from_text(self, text: str, limit: int = 6) -> List[Dict[str, str]]:
        lowered = text.lower()
        recommendations: List[Dict[str, str]] = []
        already: set = set()
        for condition, resources in self.library.items():
            if condition in lowered:
                for item in resources:
                    key = (item["title"], item["url"])
                    if key not in already:
                        recommendations.append(item)
                        already.add(key)
        return recommendations[:limit]

    def list_categories(self) -> List[str]:
        return sorted(self.library.keys())


class EpidemiologyMonitor:
    def __init__(self) -> None:
        self.terms = Counter()

    def ingest(self, text: str) -> None:
        tokens = re.findall(r"[a-zA-Zà-úÀ-Ú]{3,}", text.lower())
        for token in tokens:
            if token in STOPWORDS or len(token) < 4:
                continue
            self.terms[token] += 1

    def export(self, limit: int = 20) -> Dict[str, int]:
        return dict(self.terms.most_common(limit))


class ActiveLearningTracker:
    FAILURE_PATTERNS = (
        "nao sei",
        "nao tenho informacao",
        "nao consigo responder",
        "procure um profissional imediatamente",
        "sem dados suficientes",
        "modelo configurado foi descontinuado",
    )

    def should_flag(self, user_text: str, bot_text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
        metadata = metadata or {}
        if metadata.get("type") == "model_decommissioned":
            return None
        for pattern in self.FAILURE_PATTERNS:
            if pattern in bot_text.lower():
                return {
                    "user": user_text,
                    "bot": bot_text[:500],
                    "reason": f"Resposta com baixa cobertura: '{pattern}'",
                }
        if user_text.strip().endswith("?") and "?" not in bot_text:
            return {
                "user": user_text,
                "bot": bot_text[:500],
                "reason": "Pergunta direta sem resposta correspondente.",
            }
        return None


class MultimodalFusionEngine:
    def fuse(
        self,
        user_text: str,
        exam_findings: List[Dict[str, Any]],
        imaging_findings: List[Dict[str, Any]],
        wearable_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "user_text": user_text,
            "exam_count": len(exam_findings),
            "imaging_count": len(imaging_findings),
            "wearable_payload": wearable_payload,
        }
        blob = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        vector = [
            round(int(digest[idx : idx + 2], 16) / 255, 4) for idx in range(0, 10, 2)
        ]
        summary = (
            f"Fusao multimodal: {len(exam_findings)} exames, "
            f"{len(imaging_findings)} imagens, dados wearable={'sim' if wearable_payload else 'nao'}."
        )
        return {"signature": digest[:16], "vector": vector, "summary": summary}


class ConfidenceCalibrator:
    def score(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.45
        explanation: List[str] = ["Base inicial de 0.45."]
        if context.get("exam_findings"):
            score += 0.15
            explanation.append("Dados de exames estruturados fornecidos (+0.15).")
        if context.get("imaging_findings"):
            score += 0.15
            explanation.append("Achados radiologicos presentes (+0.15).")
        if context.get("medication_alerts"):
            score += 0.05
            explanation.append("Consideracao de interacoes medicamentosas (+0.05).")
        if "Aviso" in response or "orientar" in response.lower():
            score -= 0.05
            explanation.append("Resposta destaca limitacoes (+/-).")
        score = min(max(score, 0.1), 0.95)
        confidence_label = "alta" if score >= 0.75 else "moderada" if score >= 0.55 else "baixa"
        explanation.append(f"Nivel de confianca: {confidence_label}.")
        return {
            "score": round(score, 2),
            "label": confidence_label,
            "rationale": " ".join(explanation),
        }


class ConversationPlanner:
    def plan(
        self,
        user_text: str,
        state: Dict[str, Any],
    ) -> Dict[str, str]:
        lowered = user_text.lower()
        stage = "triagem"
        next_action = "Coletar sintomas adicionais."
        if any(term in lowered for term in ["resultado", "exame", "laudo"]):
            stage = "analise-exames"
            next_action = "Cruzar sintomas com exames enviados."
        if state.get("imaging_findings"):
            stage = "integracao-imagem"
            next_action = "Conectar achados de imagem com relato clinico."
        if any(term in lowered for term in ["dor intensa", "emergencia", "urgente"]):
            stage = "emergencia"
            next_action = "Priorizar orientacao de socorro imediato."
        if "imc" in lowered:
            next_action = "Executar calculo de IMC e aguardar nova instrucao."
        plan_prompt = (
            f"Plano atual: {stage}. Proxima acao: {next_action}. "
            "Caso ja tenha as 10 perguntas respondidas, sintetizar diagnosticos diferenciais."
        )
        return {"stage": stage, "next_action": next_action, "plan_prompt": plan_prompt}


class ExplainabilityEngine:
    def generate(self, finding: Dict[str, Any]) -> str:
        payload = finding.get("payload", {})
        probs = payload.get("probabilities") or {}
        if not probs:
            return (
                f"Nenhuma probabilidade detalhada informada para {finding.get('name')}."
            )
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        top = ", ".join(f"{label}: {value:.2f}" for label, value in ranked[:3])
        return (
            f"Mapa de calor sugerido para {finding.get('name')}: principais hipoteses {top}. "
            "Solicitar grad-CAM ao microservico para suporte visual definitivo."
        )


def truncate_text(value: str, limit: int = 6000) -> str:
    if len(value) <= limit:
        return value
    return (
        value[:limit]
        + "\n\n[Contexto reduzido automaticamente para manter a compatibilidade com o modelo.]"
    )


def predict_with_fallback(client: Groq, model_name: str, messages: List[Dict[str, str]]) -> str:
    limits = [6000, 4500, 3000]
    last_error: Optional[BadRequestError] = None
    for limit in limits:
        trimmed_messages = _limit_last_user_message(messages, limit)
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=trimmed_messages,
            )
            choice = completion.choices[0]
            content = getattr(choice.message, "content", "") if hasattr(choice, "message") else ""
            return content or ""
        except BadRequestError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise BadRequestError("Falha ao gerar resposta apos tentativas de reducao de contexto.")


def _limit_last_user_message(messages: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    trimmed: List[Dict[str, str]] = []
    for idx, message in enumerate(messages):
        if message.get("role") == "user" and idx == len(messages) - 1:
            trimmed.append(
                {
                    "role": message["role"],
                    "content": truncate_text(message.get("content", ""), limit=limit),
                }
            )
        else:
            trimmed.append(message)
    return trimmed


def extract_symptom_candidates(text: str) -> List[str]:
    lowered = text.lower()
    if not any(hint in lowered for hint in SYMPTOM_HINTS):
        return []
    tokens = re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+", lowered)
    symptoms: List[str] = []
    for token in tokens:
        token = token.strip()
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if token not in symptoms:
            symptoms.append(token)
    return symptoms[:12]


def get_unique_symptoms(limit: Optional[int] = None) -> List[str]:
    collected: List[str] = []
    for entry in st.session_state.symptom_log:
        for token in entry.get("symptoms", []):
            if token not in collected:
                collected.append(token)
    if limit is not None:
        return collected[:limit]
    return collected


def summarize_symptom_log(items: List[Dict[str, Any]]) -> str:
    tokens = get_unique_symptoms()
    if not tokens:
        return "Ainda nao registrei sintomas anteriores nesta conversa."
    prettified = [token.replace("_", " ").capitalize() for token in tokens]
    return "Sintomas mencionados anteriormente: " + ", ".join(prettified)


def build_symptom_report() -> str:
    tokens = get_unique_symptoms()
    if not tokens:
        return "Sem sintomas registrados no momento."
    prettified = [token.replace("_", " ").capitalize() for token in tokens]
    return "Sintomas relatados: " + ", ".join(prettified)


def reset_question_progress() -> None:
    st.session_state.question_progress = {
        "total": 10,
        "answered": 0,
        "current": 1,
        "history": [],
    }


def find_medication_query(text: str) -> Optional[str]:
    lowered = text.lower()
    trigger_terms = (
        "bula",
        "posologia",
        "dose",
        "dosagem",
        "como tomar",
        "como usar",
        "indicacao",
        "efeito colateral",
    )
    if not any(term in lowered for term in trigger_terms):
        return None
    for key, data in MEDICATION_MONOGRAPHS.items():
        aliases = [key] + data.get("aliases", [])
        if any(alias in lowered for alias in aliases):
            return key
    return None


def generate_medication_response(med_key: str) -> str:
    data = MEDICATION_MONOGRAPHS.get(med_key)
    if not data:
        return ""
    lines = [
        f"[FARMA] Informacoes essenciais sobre {med_key.capitalize()} ({data.get('class', 'Medicamento')})",
        "",
        "**Indicacoes principais:**",
    ]
    for item in data.get("indications", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Contraindicacoes:**")
    for item in data.get("contra", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**Alertas e precaucoes:**")
    for item in data.get("warnings", []):
        lines.append(f"- {item}")
    lines.append("")
    dose = data.get("dose")
    if dose:
        lines.append(f"**Posologia orientativa:** {dose}")
        lines.append("")
    references = data.get("references", [])
    if references:
        lines.append("**Referencias confiaveis:**")
        for ref in references:
            lines.append(f"- {ref}")
        lines.append("")
    lines.append(MEDICATION_DISCLAIMER)
    return "\n".join(lines)


LAB_RANGES = {
    "hemoglobina": {"low": 12.0, "high": 17.5, "unit": "g/dL", "label": "[hemoglobina]"},
    "hemacias": {"low": 4.0, "high": 6.0, "unit": "milhoes/mm3", "label": "[hemacias]"},
    "eritrocitos": {"low": 4.0, "high": 6.0, "unit": "milhoes/mm3", "label": "[eritrocitos]"},
    "hematocrito": {"low": 37.0, "high": 52.0, "unit": "%", "label": "[hematocrito]"},
    "leucocitos": {"low": 4000.0, "high": 11000.0, "unit": "/mm3", "label": "[leucocitos]"},
    "leucócitos": {"low": 4000.0, "high": 11000.0, "unit": "/mm3", "label": "[leucocitos]"},
    "plaquetas": {"low": 150000.0, "high": 450000.0, "unit": "/mm3", "label": "[plaquetas]"},
}


def parse_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        for part in value:
            parsed = parse_numeric_value(part)
            if parsed is not None:
                return parsed
        return None
    text = str(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if match:
        return float(match.group(0).replace(",", "."))
    return None


def clear_conversation_memory() -> None:
    st.session_state.memory_messages = []


def adjust_memory_window(window: int) -> None:
    messages = st.session_state.get("memory_messages")
    if not isinstance(messages, list):
        messages = []
    max_messages = max(window * 2, 2)
    if len(messages) > max_messages:
        st.session_state.memory_messages = messages[-max_messages:]
    else:
        st.session_state.memory_messages = messages


def append_memory_message(role: str, content: str) -> None:
    messages = st.session_state.get("memory_messages")
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": role, "content": content})
    st.session_state.memory_messages = messages
    adjust_memory_window(int(st.session_state.get("memory_window", 60)))


def pop_last_memory_message() -> None:
    messages = st.session_state.get("memory_messages")
    if isinstance(messages, list) and messages:
        messages.pop()
        st.session_state.memory_messages = messages


def build_model_messages(system_prompt: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    stored = st.session_state.get("memory_messages")
    if isinstance(stored, list):
        messages.extend(
            {"role": item.get("role", "user"), "content": item.get("content", "")}
            for item in stored
            if isinstance(item, dict) and "content" in item
        )
    return messages


def analyze_exam_item(item: Dict[str, Any]) -> List[str]:
    findings: List[str] = []
    normalized = item.get("normalized") or {}
    raw_text = item.get("raw_text", "")
    for key, meta in LAB_RANGES.items():
        value = normalized.get(key)
        if value is None and raw_text:
            pattern = rf"{key}[^0-9-]*([-]?\d+(?:[.,]\d+)?)"
            match = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if match:
                value = match.group(1)
        if value is None:
            continue
        numeric = parse_numeric_value(value)
        if numeric is None:
            continue
        low, high = meta["low"], meta["high"]
        label = meta["label"]
        if numeric < low:
            findings.append(
                f"{label}: valor {numeric} abaixo do intervalo ideal ({low}-{high} {meta['unit']}). Sugerir avaliacao clinica."
            )
        elif numeric > high:
            findings.append(
                f"{label}: valor {numeric} acima do intervalo ideal ({low}-{high} {meta['unit']}). Sugerir correlacionar com sintomas."
            )
    return findings


def apply_theme_settings() -> None:
    theme = st.session_state.get("theme", "Claro")
    font_scale = st.session_state.get("font_scale", 1.0)
    palette = THEME_STYLES.get(theme, THEME_STYLES["Claro"])
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        background-color: {palette['bg']} !important;
        color: {palette['text']} !important;
        font-size: {font_scale}em;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }}
    
    /* Header moderno */
    h1 {{
        font-weight: 700 !important;
        font-size: 2.5em !important;
        margin-bottom: 0.3em !important;
        background: linear-gradient(135deg, {palette['accent']} 0%, #5b21b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    h2, h3 {{
        font-weight: 600 !important;
        color: {palette['text']} !important;
    }}
    
    /* Botões modernos */
    .stButton button {{
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        border: 1px solid {palette['accent_soft']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
    }}
    
    /* Botão de voz especial */
    .voice-button {{
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        font-size: 1.1em !important;
        padding: 0.8em 1.5em !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3) !important;
    }}
    
    .voice-button:hover {{
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.4) !important;
        transform: scale(1.05) !important;
    }}
    
    /* Panels melhorados */
    .themed-panel {{
        background: {palette['panel']};
        border: 1px solid {palette['accent_soft']};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        transition: all 0.3s ease;
    }}
    
    .themed-panel:hover {{
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
        transform: translateY(-2px);
    }}
    
    .step-card {{
        background: {palette['panel']};
        border-left: 6px solid {palette['accent']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: {palette['text']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    
    .step-card.inactive {{
        border-left-color: {palette['accent_soft']};
        opacity: 0.6;
    }}
    
    /* Badges modernos */
    .badge-accent {{
        background: {palette['accent']};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    
    /* Alertas melhorados */
    .emergency-banner {{
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #dc2626;
        border-radius: 16px;
        padding: 16px;
        color: #7f1d1d;
        margin-bottom: 12px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.15);
    }}
    
    .warning-banner {{
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 16px;
        color: #92400e;
        margin-bottom: 12px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
    }}
    
    /* Cards de educação */
    .education-card {{
        border: 1px solid {palette['accent_soft']};
        border-radius: 14px;
        padding: 16px;
        background: {palette['panel']};
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }}
    
    .education-card:hover {{
        border-color: {palette['accent']};
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }}
    
    /* Tabs modernos */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent;
        border-bottom: 2px solid {palette['accent_soft']};
        margin-bottom: 16px;
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 12px 12px 0 0;
        padding: 12px 24px !important;
        font-weight: 500;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {palette['accent_soft']};
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        background: transparent !important;
        border: none !important;
        padding-top: 16px !important;
    }}
    
    /* Progress bar */
    .stProgress > div {{
        background: {palette['accent_soft']} !important;
        border-radius: 20px !important;
        height: 12px !important;
    }}
    
    .stProgress > div > div {{
        background: linear-gradient(90deg, {palette['accent']} 0%, #8b5cf6 100%) !important;
        border-radius: 20px !important;
    }}
    
    /* Input de chat aprimorado */
    div[data-testid="stChatInput"] {{
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        width: min(750px, 92vw);
        background: {palette['panel']};
        border: 2px solid {palette['accent_soft']};
        border-radius: 24px;
        box-shadow: 0 20px 48px rgba(15, 23, 42, 0.15);
        z-index: 1010;
    }}
    
    div[data-testid="stChatInput"] textarea {{
        min-height: 60px !important;
        border-radius: 20px !important;
        font-size: 1em !important;
    }}

    #chat-history {{
        display: flex;
        flex-direction: column;
        gap: 12px;
    }}

    .message {{
        border-radius: 16px;
        padding: 14px 18px;
        line-height: 1.5;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }}

    .message strong {{
        font-weight: 600;
    }}

    .message.user-message {{
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        align-self: flex-end;
    }}

    .message.ai-message {{
        background: {palette['panel']};
        border: 1px solid {palette['accent_soft']};
    }}

    .message.ai-message.error {{
        border-color: #dc2626;
        background: rgba(248, 113, 113, 0.12);
    }}

    .message.ai-message.emergency {{
        border-color: #dc2626;
        background: rgba(248, 113, 113, 0.15);
    }}

    .message.ai-message.alert {{
        border-color: #f59e0b;
        background: rgba(251, 191, 36, 0.12);
    }}
    
    /* Sidebar moderna */
    section[data-testid="stSidebar"] {{
        background: {palette['panel']} !important;
        border-right: 1px solid {palette['accent_soft']};
    }}
    
    section[data-testid="stSidebar"] > div {{
        background: {palette['panel']} !important;
    }}
    
    /* Animações suaves */
    * {{
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }}
    
    /* Espaçamento */
    .stChatMessageContainer {{
        padding-bottom: 240px !important;
    }}
    
    main .block-container {{
        padding-bottom: 280px !important;
        padding-top: 2rem !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def update_question_progress(response: str) -> None:
    if not st.session_state.get("triage_mode", False):
        return
    data = st.session_state.question_progress
    total = data.get("total", 10)
    lower = response.lower()
    match = re.search(r"pergunt[ao]*\s*(\d+)", lower)
    if match:
        number = int(match.group(1))
        number = min(max(number, 1), total)
        data["current"] = number
        data["answered"] = max(data["answered"], min(number - 1, total))
    prompt_snippet = None
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for line in lines:
        if "pergunta" in line.lower():
            prompt_snippet = line
            break
    history = data.setdefault("history", [])
    if len(history) < total:
        history.extend([""] * (total - len(history)))
    if prompt_snippet:
        history[data["current"] - 1] = prompt_snippet[:120]
    st.session_state.question_progress = data


def generate_qr_code(content: str) -> Optional[bytes]:
    if not qrcode:
        return None
    try:
        qr = qrcode.QRCode(box_size=2, border=2)
        qr.add_data(content[:500])
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        return None



def render_patient_dashboard() -> None:
    st.markdown(f"### {ICON_STEP} Painel do paciente", unsafe_allow_html=True)
    tabs = st.tabs(["Perfil", "Sintomas", "Exames", "Alertas", "Emergencia", "Insights"])
    with tabs[0]:
        render_patient_profile()
    with tabs[1]:
        tokens = get_unique_symptoms()
        if tokens:
            prettified = [token.replace("_", " ").capitalize() for token in tokens]
            for token in prettified:
                st.markdown(f"- {token}")
        else:
            st.caption("Nenhum sintoma registrado ainda.")
    with tabs[2]:
        if st.session_state.exam_findings or st.session_state.imaging_findings:
            for item in st.session_state.exam_findings[-5:]:
                st.markdown(f"- {ICON_EXAM} {item['name']}")
            for item in st.session_state.imaging_findings[-5:]:
                st.markdown(f"- {ICON_IMAGING} {item['name']}")
        else:
            st.caption("Nenhum exame anexado.")
    with tabs[3]:
        if st.session_state.medication_alerts:
            for alert in st.session_state.medication_alerts[-5:]:
                st.markdown(f"- {ICON_INTERACTION} {alert}")
        else:
            st.caption("Nenhum alerta farmacologico no momento.")
    with tabs[4]:
        if st.session_state.critical_events:
            st.markdown(
                f"<div class='emergency-banner'>{ICON_EMERGENCY} {st.session_state.critical_events[-1]}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "[Ligar 192 (SAMU)](tel:192) | [Ligar 190 (Policia)](tel:190)",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Nenhum evento critico detectado.")
    with tabs[5]:
        render_advanced_insights()


def render_wearable_insights() -> None:
    if not st.session_state.wearable_payload:
        st.caption("Sem dados recentes de wearables.")
        return
    payload = st.session_state.wearable_payload
    st.markdown(f"#### {ICON_WEARABLE} Insights de wearables", unsafe_allow_html=True)
    for key, value in payload.items():
        if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            st.line_chart(value, height=120)
            st.caption(f"{key.title()} com {len(value)} pontos coletados.")
        else:
            st.text(f"{key}: {value}")



def render_explainability_panel() -> None:
    st.markdown("#### Detalhes do raciocinio", unsafe_allow_html=True)
    bullets: List[str] = []
    if st.session_state.exam_findings:
        bullets.append(f"{ICON_EXAM} Exames interpretados: {len(st.session_state.exam_findings)}")
    if st.session_state.imaging_findings:
        bullets.append(f"{ICON_IMAGING} Imagens analisadas: {len(st.session_state.imaging_findings)}")
    if st.session_state.symptom_log:
        tokens = get_unique_symptoms(limit=4)
        if tokens:
            bullets.append(f"{ICON_STEP} Sintomas chave: {', '.join(tokens)}")
    if bullets:
        for bullet in bullets:
            st.markdown(f"- {bullet}")
    else:
        st.caption("Ainda coletando informacoes para explicabilidade.")
    if st.session_state.explainability_notes:
        st.markdown("Notas de imagem recentes:")
        for note in st.session_state.explainability_notes[-3:]:
            st.markdown(f"- {note}")


def render_education_cards() -> None:

    if not st.session_state.education_recommendations:
        st.caption("Recomende exames ou descreva sintomas para ativar materiais educativos.")
        return
    checklist = st.session_state.education_checklist
    for idx, rec in enumerate(st.session_state.education_recommendations):
        key = rec["url"]
        checked = checklist.get(key, False)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"""
                <div class="education-card">
                    <strong>{rec['title']}</strong> <span class='badge-accent'>{rec['type']}</span><br/>
                    <a href="{rec['url']}" target="_blank">Abrir material</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            checklist[key] = st.checkbox("Lido", value=checked, key=f"edu_{idx}")
    st.session_state.education_checklist = checklist


def refresh_multiexam_reasoning() -> None:
    reasoner = st.session_state.get("cross_reasoner")
    labs = st.session_state.get("advanced_lab_findings", [])
    imaging = st.session_state.get("dicom_findings", [])
    wearable_snapshot = st.session_state.get("wearable_payload", {})
    if reasoner:
        outcome = reasoner.synthesize(
            lab_alerts=labs,
            imaging_alerts=imaging,
            wearable_snapshot=wearable_snapshot if isinstance(wearable_snapshot, dict) else {},
        )
        st.session_state.cross_validation_notes = outcome.get("notes", [])
        st.session_state.cross_conflicts = outcome.get("conflicts", [])
    advisor = st.session_state.get("guideline_advisor")
    if advisor:
        has_high_risk_lab = any(
            any(sub.get("severity") == "alto" for sub in entry.get("alerts", []))
            for entry in labs
            if isinstance(entry, dict)
        )
        imaging_flags = any(
            bool(entry.get("cad_flags"))
            for entry in imaging
            if isinstance(entry, dict)
        )
        ecg_flags = any(
            "falha" in (insight.get("summary", "").lower())
            for insight in st.session_state.get("ecg_insights", [])
            if isinstance(insight, dict)
        )
        plan_payload = advisor.suggest_next_steps(
            has_high_risk_lab=has_high_risk_lab,
            imaging_flags=imaging_flags,
            ecg_red_flags=ecg_flags,
        )
        st.session_state.guideline_plan = plan_payload.get("plan", [])
        demographics = st.session_state.get("demographics", {})
        history = {}
        if isinstance(demographics, dict):
            history = demographics.get("history") or {}
            if not isinstance(history, dict):
                history = {}
            st.session_state.guideline_cha2ds2 = advisor.cha2ds2_vasc(
                age=demographics.get("age"),
                sex=demographics.get("sex"),
                history=history,
            )
            wearable_rr = wearable_snapshot.get("respiratory_rate")
            labs_payload = demographics.get("labs") if isinstance(demographics.get("labs"), dict) else {}
            st.session_state.guideline_curb65 = advisor.curb65(
                age=demographics.get("age"),
                confusion=bool(demographics.get("confusion")),
                urea=labs_payload.get("urea") if isinstance(labs_payload, dict) else None,
                respiratory_rate=int(wearable_rr) if isinstance(wearable_rr, (int, float)) else None,
                blood_pressure=demographics.get("blood_pressure")
                if isinstance(demographics.get("blood_pressure"), dict)
                else None,
            )


def render_advanced_insights() -> None:
    st.markdown("#### Insights laboratoriais avancados", unsafe_allow_html=True)
    labs = st.session_state.get("advanced_lab_findings", [])
    if labs:
        for entry in labs[-3:]:
            alerts = entry.get("alerts", [])
            alert_text = ", ".join(
                f"{item.get('marker')} ({item.get('severity')})"
                for item in alerts
                if isinstance(item, dict)
            )
            trend = "; ".join(entry.get("trend_summary", []) or [])
            line = f"{entry.get('panel')}: {alert_text}" if alert_text else entry.get("panel")
            st.markdown(f"- {line}")
            if trend:
                st.caption(trend)
    else:
        st.caption("Sem flags laboratoriais avancados no momento.")

    st.markdown("#### CAD e metadados de DICOM", unsafe_allow_html=True)
    dicom_items = st.session_state.get("dicom_findings", [])
    if dicom_items:
        for record in dicom_items[-3:]:
            meta = "; ".join(record.get("meta_summary", []) or [])
            cad_flags = "; ".join(record.get("cad_flags", []) or [])
            st.markdown(f"- {record.get('id')}: {meta or 'Sem metadados'}")
            if cad_flags:
                st.caption(cad_flags)
    else:
        st.caption("Nenhum estudo DICOM detalhado registrado.")

    st.markdown("#### Correlacao multiexames", unsafe_allow_html=True)
    cross_notes = st.session_state.get("cross_validation_notes", [])
    cross_conflicts = st.session_state.get("cross_conflicts", [])
    if cross_notes:
        for note in cross_notes[-5:]:
            st.markdown(f"- {note}")
    else:
        st.caption("Sem notas combinadas ate o momento.")
    if cross_conflicts:
        st.warning("Conflitos detectados: " + " | ".join(cross_conflicts))

    st.markdown("#### ECG / sinais vitais", unsafe_allow_html=True)
    ecg_insights = st.session_state.get("ecg_insights", [])
    if ecg_insights:
        for insight in ecg_insights[-3:]:
            st.markdown(f"- {insight.get('summary', 'Sem resumo')}")
    else:
        st.caption("Nenhuma analise de ECG processada.")

    st.markdown("#### Diretrizes clinicas sugeridas", unsafe_allow_html=True)
    plan = st.session_state.get("guideline_plan", [])
    if plan:
        for step in plan:
            st.markdown(f"- {step}")
    else:
        st.caption("Sem alertas de diretrizes no momento.")
    cha2ds2 = st.session_state.get("guideline_cha2ds2")
    if isinstance(cha2ds2, dict):
        st.caption(
            f"CHA2DS2-VASc: {cha2ds2.get('score')} pontos ({', '.join(cha2ds2.get('factors', []))}) - "
            f"{cha2ds2.get('recommendation')}"
        )
    curb65 = st.session_state.get("guideline_curb65")
    if isinstance(curb65, dict):
        st.caption(f"CURB-65: {curb65.get('score')} - {curb65.get('recommendation')}")


def render_patient_profile() -> None:
    demographics = st.session_state.get("demographics")
    if not isinstance(demographics, dict) or not any(demographics.values()):
        st.caption("Perfil demografico ainda nao informado.")
        return

    age = demographics.get("age")
    sex = demographics.get("sex")
    bmi = demographics.get("bmi")
    weight = demographics.get("weight_kg")
    height = demographics.get("height_cm")
    pregnancy_status = demographics.get("pregnancy_status")
    smoking_status = demographics.get("smoking_status")

    if age:
        st.markdown(f"- Idade: **{age}** anos")
    if sex:
        st.markdown(f"- Sexo: **{sex}**")
    if weight:
        st.markdown(f"- Peso: **{weight:.1f} kg**")
    if height:
        st.markdown(f"- Altura: **{height:.1f} cm**")
    if bmi:
        st.markdown(f"- IMC calculado: **{bmi:.1f}**")
    if pregnancy_status:
        st.markdown(f"- Gestacao: **{pregnancy_status}**")
    if smoking_status:
        st.markdown(f"- Tabagismo: **{smoking_status}**")

    blood_pressure = demographics.get("blood_pressure")
    if isinstance(blood_pressure, dict):
        sys_bp = blood_pressure.get("systolic")
        dia_bp = blood_pressure.get("diastolic")
        if sys_bp and dia_bp:
            st.markdown(f"- Pressao arterial: **{sys_bp}/{dia_bp} mmHg**")

    history = demographics.get("history")
    if isinstance(history, dict) and any(history.values()):
        st.markdown("##### Condicoes clinicas registradas")
        for key, value in history.items():
            if value:
                st.markdown(f"- {key.replace('_', ' ').title()}")

    notes = demographics.get("notes")
    if notes:
        st.markdown("##### Observacoes adicionais")
        st.write(notes)


def build_context_sections() -> tuple[str, Dict[str, Any]]:
    exam_context = st.session_state.exam_pipeline.render_for_prompt(st.session_state.exam_findings)
    imaging_context = st.session_state.radiography_service.render_for_prompt(
        st.session_state.imaging_findings
    )
    wearable_payload = st.session_state.wearable_payload or {}
    wearable_context = ""
    if wearable_payload:
        wearable_context = "Dados de wearables: " + json.dumps(wearable_payload)[:800]
    medication_alerts = st.session_state.medication_alerts
    demographics = st.session_state.get("demographics", {})
    pieces = []
    if exam_context:
        pieces.append("Resumo de exames estruturados:\n" + exam_context)
        analyses = []
        for item in st.session_state.exam_findings:
            analyses.extend(analyze_exam_item(item))
        if analyses:
            pieces.append("Analise automatica de exames:\n" + "\n".join(f"- {msg}" for msg in analyses))
    if imaging_context:
        pieces.append("Resumo de radiografias:\n" + imaging_context)
    if wearable_context:
        pieces.append(wearable_context)
    if medication_alerts:
        pieces.append("Alertas farmacologicos ativos: " + "; ".join(sorted(set(medication_alerts))))
    if isinstance(demographics, dict) and any(
        value for value in demographics.values() if value not in (None, "", [], {})
    ):
        demo_lines: List[str] = []
        age = demographics.get("age")
        if age not in (None, "", 0):
            demo_lines.append(f"Idade: {age} anos")
        sex = demographics.get("sex")
        if sex:
            demo_lines.append(f"Sexo: {sex}")
        weight = demographics.get("weight_kg")
        if weight not in (None, "", 0):
            try:
                demo_lines.append(f"Peso: {float(weight):.1f} kg")
            except (TypeError, ValueError):
                demo_lines.append(f"Peso: {weight} kg")
        height = demographics.get("height_cm")
        if height not in (None, "", 0):
            try:
                demo_lines.append(f"Altura: {float(height):.1f} cm")
            except (TypeError, ValueError):
                demo_lines.append(f"Altura: {height} cm")
        bmi = demographics.get("bmi")
        if bmi not in (None, "", 0):
            try:
                demo_lines.append(f"IMC: {float(bmi):.1f}")
            except (TypeError, ValueError):
                demo_lines.append(f"IMC: {bmi}")
        pregnancy_status = demographics.get("pregnancy_status")
        if pregnancy_status:
            demo_lines.append(f"Gestacao: {pregnancy_status}")
        smoking_status = demographics.get("smoking_status")
        if smoking_status:
            demo_lines.append(f"Tabagismo: {smoking_status}")
        blood_pressure = demographics.get("blood_pressure")
        if isinstance(blood_pressure, dict):
            systolic = blood_pressure.get("systolic")
            diastolic = blood_pressure.get("diastolic")
            if systolic and diastolic:
                demo_lines.append(f"Pressao arterial: {systolic}/{diastolic} mmHg")
        history = demographics.get("history")
        if isinstance(history, dict):
            conditions = [
                key.replace("_", " ").title()
                for key, value in history.items()
                if value
            ]
            if conditions:
                demo_lines.append("Historico: " + ", ".join(conditions))
        notes = demographics.get("notes")
        if notes:
            demo_lines.append("Observacoes: " + str(notes))
        if demo_lines:
            pieces.append("Perfil do paciente:\n" + "\n".join(f"- {line}" for line in demo_lines))
    advanced_labs = st.session_state.get("advanced_lab_findings", [])
    if advanced_labs:
        lab_lines = []
        for entry in advanced_labs[-3:]:
            alerts = entry.get("alerts", [])
            markers = ", ".join(
                f"{item.get('marker')}({item.get('severity')})"
                for item in alerts
                if isinstance(item, dict)
            )
            summary = entry.get("panel")
            if markers:
                summary = f"{summary}: {markers}"
            if entry.get("trend_summary"):
                summary += " | " + "; ".join(entry.get("trend_summary", []))
            lab_lines.append(summary)
        if lab_lines:
            pieces.append("Lab avancado:\n" + "\n".join(lab_lines))
    dicom_findings = st.session_state.get("dicom_findings", [])
    if dicom_findings:
        dicom_lines = []
        for record in dicom_findings[-3:]:
            meta = "; ".join(record.get("meta_summary", []) or [])
            cad = "; ".join(record.get("cad_flags", []) or [])
            dicom_lines.append(f"{record.get('id')}: {meta} {cad}")
        if dicom_lines:
            pieces.append("Resumo DICOM detalhado:\n" + "\n".join(dicom_lines))
    cross_notes = st.session_state.get("cross_validation_notes", [])
    if cross_notes:
        pieces.append("Notas combinadas multiexames:\n" + "\n".join(f"- {note}" for note in cross_notes[-5:]))
    if st.session_state.get("cross_conflicts"):
        pieces.append("Conflitos detectados: " + "; ".join(st.session_state.cross_conflicts))
    ecg_insights = st.session_state.get("ecg_insights", [])
    if ecg_insights:
        pieces.append(
            "Resumo ECG:\n" + "\n".join(f"- {item.get('summary', '')}" for item in ecg_insights[-3:])
        )
    guideline_plan = st.session_state.get("guideline_plan", [])
    if guideline_plan:
        pieces.append("Plano sugerido por diretrizes:\n" + "\n".join(f"- {step}" for step in guideline_plan))
    triage_state = "ativo" if st.session_state.get("triage_mode", False) else "inativo"
    pieces.append(f"Status da triagem: {triage_state}")
    symptom_report = build_symptom_report()
    if "Sem sintomas" not in symptom_report:
        pieces.append(symptom_report)
    plan_state = st.session_state.planner_state or {}
    if plan_state.get("plan_prompt"):
        pieces.append("Plano conversacional:\n" + plan_state["plan_prompt"])
    fusion_data = st.session_state.multimodal_signature or {}
    if fusion_data.get("summary"):
        pieces.append(
            "Assinatura multimodal: "
            + fusion_data["summary"]
            + f" Vetor={fusion_data.get('vector')}"
        )
    context_payload = truncate_text("\n\n".join(pieces), limit=4000)
    context_meta = {
        "critical_flags": st.session_state.critical_events,
        "medication_alerts": medication_alerts,
        "exam_findings": st.session_state.exam_findings,
        "imaging_findings": st.session_state.imaging_findings,
        "demographics": demographics if isinstance(demographics, dict) else {},
        "advanced_lab_findings": advanced_labs,
        "dicom_findings": dicom_findings,
        "cross_notes": cross_notes,
        "ecg_insights": ecg_insights,
        "guideline_plan": guideline_plan,
        "planner_state": plan_state,
        "multimodal_signature": fusion_data,
    }
    return context_payload, context_meta


def generate_tts_audio(text: str, language: str = "pt") -> Optional[bytes]:
    if not gTTS:
        return None
    try:
        tts = gTTS(text=text, lang=language)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


class GeminiVoiceAgent:
    """Wrapper simples para acionar o Gemini AI Studio com audio."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self.api_key = api_key or DEFAULT_GEMINI_API_KEY
        self.model_name = model_name
        self._client: Optional[Any] = None
        self._legacy_model: Optional[Any] = None
        self._using_files_api = False
        self.last_error: Optional[str] = None

    def available(self) -> bool:
        return bool(self.api_key and gemini_sdk_available())

    def _legacy_model_name(self) -> str:
        if self.model_name.startswith("models/"):
            return self.model_name.split("/", 1)[1]
        return self.model_name

    def _ensure_backend(self) -> None:
        if not self.available():
            raise RuntimeError(
                "Gemini Voice Agent indisponivel (biblioteca ou chave ausente)."
            )
        if self._client or self._legacy_model:
            return
        if google_genai is not None:
            self._client = google_genai.Client(api_key=self.api_key)
        elif legacy_genai is not None:
            legacy_genai.configure(api_key=self.api_key)
            self._legacy_model = legacy_genai.GenerativeModel(
                model_name=self._legacy_model_name(),
                system_instruction=VOICE_AGENT_INSTRUCTIONS,
            )
        else:  # pragma: no cover
            raise RuntimeError("SDK Gemini nao instalado.")

    def transcribe_and_respond(
        self,
        *,
        audio: bytes,
        mime_type: str = "audio/wav",
    ) -> Dict[str, Any]:
        self._ensure_backend()
        start = time.time()
        try:
            if self._client is not None and google_genai_types is not None:
                prompt = (
                    f"{VOICE_AGENT_INSTRUCTIONS}\n\n{VOICE_AGENT_RESPONSE_SCHEMA}\n"
                    "Quando nao houver fala, informe explicitamente em portugues."
                )
                audio_part = google_genai_types.Part.from_bytes(
                    data=audio,
                    mime_type=mime_type,
                )
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, audio_part],
                    config={
                        "temperature": 0.4,
                        "top_p": 0.9,
                        "max_output_tokens": 1024,
                        "response_mime_type": "application/json",
                    },
                )
                raw_text = getattr(response, "text", "") or ""
            elif self._legacy_model is not None:
                response = self._legacy_model.generate_content(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {"mime_type": mime_type, "data": audio},
                                {"text": VOICE_AGENT_RESPONSE_SCHEMA},
                            ],
                        }
                    ],
                    generation_config={
                        "temperature": 0.5,
                        "top_p": 0.95,
                        "max_output_tokens": 1024,
                        "response_mime_type": "application/json",
                    },
                    request_options={"timeout": 60},
                )
                raw_text = getattr(response, "text", "") or ""
            else:  # pragma: no cover
                raise RuntimeError("Nenhum backend Gemini configurado.")
        except Exception as exc:
            self.last_error = str(exc)
            raise
        latency = time.time() - start
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            payload = {
                "transcription": raw_text.strip(),
                "assistant_reply": raw_text.strip(),
            }
        transcription = (payload.get("transcription") or "").strip()
        assistant_reply = (payload.get("assistant_reply") or "").strip()
        return {
            "transcription": transcription,
            "assistant_reply": assistant_reply,
            "latency": latency,
            "raw": raw_text,
        }


def get_voice_agent() -> Optional[GeminiVoiceAgent]:
    if not gemini_sdk_available():
        return None
    agent = st.session_state.get("voice_agent")
    if not isinstance(agent, GeminiVoiceAgent):
        st.session_state.voice_agent = GeminiVoiceAgent()
        agent = st.session_state.voice_agent
    return agent


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Fluxos auxiliares")

        theme_choice = st.radio(
            "Tema visual",
            options=list(THEME_STYLES.keys()),
            index=list(THEME_STYLES.keys()).index(st.session_state.theme)
            if st.session_state.theme in THEME_STYLES
            else 0,
            horizontal=True,
        )
        st.session_state.theme = theme_choice
        font_scale = st.slider(
            "Tamanho da fonte",
            min_value=0.9,
            max_value=1.4,
            value=float(st.session_state.font_scale),
            step=0.05,
        )
        st.session_state.font_scale = font_scale
        memory_window = st.slider(
            "Memoria da IA (trocas consideradas)",
            min_value=10,
            max_value=120,
            value=int(st.session_state.memory_window),
            step=5,
        )
        if memory_window != st.session_state.memory_window:
            st.session_state.memory_window = memory_window
            adjust_memory_window(memory_window)

        st.markdown("---")
        with st.expander("Dados do paciente", expanded=True):
            demographics = st.session_state.get("demographics")
            if not isinstance(demographics, dict):
                demographics = {}
            current_history = demographics.get("history")
            if not isinstance(current_history, dict):
                current_history = {}

            age_default = demographics.get("age")
            age_input = st.number_input(
                "Idade (anos)",
                min_value=0,
                max_value=120,
                value=int(age_default) if isinstance(age_default, (int, float)) and age_default > 0 else 0,
                step=1,
            )

            sex_options = ["Selecione", "F", "M", "Outro"]
            current_sex = demographics.get("sex")
            sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
            sex_input = st.selectbox("Sexo", options=sex_options, index=sex_index)

            weight_default = demographics.get("weight_kg") or 0.0
            height_default = demographics.get("height_cm") or 0.0
            weight_input = st.number_input(
                "Peso (kg)",
                min_value=0.0,
                max_value=400.0,
                value=float(weight_default),
                step=0.5,
                format="%.1f",
            )
            height_input = st.number_input(
                "Altura (cm)",
                min_value=0.0,
                max_value=250.0,
                value=float(height_default),
                step=0.5,
                format="%.1f",
            )
            bmi_value: Optional[float] = None
            if weight_input > 0 and height_input > 0:
                bmi_value = weight_input / ((height_input / 100) ** 2)
                st.caption(f"IMC calculado: {bmi_value:.1f}")

            pregnancy_status = demographics.get("pregnancy_status")
            if sex_input == "F":
                pregnancy_options = ["Nao", "Primeiro trimestre", "Segundo trimestre", "Terceiro trimestre"]
                pregnancy_index = pregnancy_options.index(pregnancy_status) if pregnancy_status in pregnancy_options else 0
                pregnancy_status = st.selectbox("Gestacao atual", options=pregnancy_options, index=pregnancy_index)
            else:
                pregnancy_status = None

            smoking_options = ["Nao fumante", "Ex-fumante", "Fumante atual"]
            current_smoke = demographics.get("smoking_status")
            smoke_index = smoking_options.index(current_smoke) if current_smoke in smoking_options else 0
            smoking_status = st.selectbox("Tabagismo", options=smoking_options, index=smoke_index)

            st.caption("Condicoes clinicas relevantes")
            history_labels = {
                "congestive_heart_failure": "Insuficiencia cardiaca",
                "hypertension": "Hipertensao",
                "stroke_tia": "AVE/AIT previo",
                "diabetes": "Diabetes",
                "vascular_disease": "Doenca vascular",
            }
            updated_history: Dict[str, bool] = {}
            for key, label in history_labels.items():
                updated_history[key] = st.checkbox(
                    label,
                    value=bool(current_history.get(key)),
                    key=f"history_{key}",
                )

            st.caption("Pressao arterial (opcional)")
            blood_pressure = demographics.get("blood_pressure")
            if not isinstance(blood_pressure, dict):
                blood_pressure = {}
            systolic_input = st.number_input(
                "Pressao sistolica",
                min_value=0,
                max_value=250,
                value=int(blood_pressure.get("systolic", 0) or 0),
                step=1,
            )
            diastolic_input = st.number_input(
                "Pressao diastolica",
                min_value=0,
                max_value=160,
                value=int(blood_pressure.get("diastolic", 0) or 0),
                step=1,
            )

            notes_input = st.text_area(
                "Outras informacoes clinicas (opcional)",
                value=str(demographics.get("notes") or ""),
                height=80,
            )

            processed_demo = {
                key: value
                for key, value in demographics.items()
                if key
                not in {
                    "age",
                    "sex",
                    "history",
                    "blood_pressure",
                    "notes",
                    "weight_kg",
                    "height_cm",
                    "bmi",
                    "pregnancy_status",
                    "smoking_status",
                }
            }
            processed_demo["history"] = updated_history
            processed_demo["notes"] = notes_input.strip() or None
            processed_demo["age"] = int(age_input) if age_input > 0 else None
            processed_demo["sex"] = sex_input if sex_input != "Selecione" else None
            processed_demo["weight_kg"] = round(weight_input, 1) if weight_input > 0 else None
            processed_demo["height_cm"] = round(height_input, 1) if height_input > 0 else None
            processed_demo["smoking_status"] = smoking_status
            if pregnancy_status and pregnancy_status != "Nao":
                processed_demo["pregnancy_status"] = pregnancy_status
            else:
                processed_demo.pop("pregnancy_status", None)
            if bmi_value:
                processed_demo["bmi"] = round(bmi_value, 1)
            else:
                processed_demo.pop("bmi", None)

            if systolic_input > 0 and diastolic_input > 0:
                processed_demo["blood_pressure"] = {
                    "systolic": int(systolic_input),
                    "diastolic": int(diastolic_input),
                }
            else:
                processed_demo.pop("blood_pressure", None)

            if not processed_demo["notes"]:
                processed_demo.pop("notes", None)
            if not processed_demo["weight_kg"]:
                processed_demo.pop("weight_kg", None)
            if not processed_demo["height_cm"]:
                processed_demo.pop("height_cm", None)

            if processed_demo != demographics:
                st.session_state.demographics = processed_demo
                refresh_multiexam_reasoning()
                st.success("Dados demograficos atualizados.")

        if st.session_state.symptom_log:
            with st.expander("Resumo rapido de sintomas", expanded=False):
                st.caption("Sintomas mais recentes registrados automaticamente.")
                st.write(summarize_symptom_log(st.session_state.symptom_log))
                if st.button("Limpar sintomas", key="clear_symptoms_sidebar"):
                    st.session_state.symptom_log = []
                    st.success("Sintomas resetados para esta sessao.")

        exam_files = st.file_uploader(
            "Envie exames estruturados (PDF, CSV, HL7, imagem)",
            type=["pdf", "csv", "hl7", "json", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if exam_files:
            for exam in exam_files:
                raw_bytes = exam.getvalue()
                exam.seek(0)
                result = st.session_state.exam_pipeline.process(exam)
                if result and result["id"] not in st.session_state.exam_ids:
                    st.session_state.exam_findings.append(result)
                    st.session_state.exam_ids.add(result["id"])
                    lab_engine = st.session_state.get("lab_interpreter")
                    if lab_engine and result["id"] not in st.session_state.advanced_lab_ids:
                        advanced = lab_engine.ingest_exam(
                            result,
                            lab_ranges=LAB_RANGES,
                            demographics=st.session_state.get("demographics"),
                        )
                        if advanced:
                            st.session_state.advanced_lab_findings.append(advanced)
                            st.session_state.advanced_lab_ids.add(result["id"])
                    ecg_engine = st.session_state.get("ecg_interpreter")
                    if (
                        ecg_engine
                        and result["id"] not in st.session_state.ecg_ids
                        and raw_bytes
                    ):
                        insight = ecg_engine.analyze_bytes(exam=result, content=raw_bytes)
                        if insight:
                            st.session_state.ecg_insights.append(insight)
                            st.session_state.ecg_ids.add(result["id"])

        if st.session_state.exam_findings:
            with st.expander("Exames processados", expanded=False):
                for item in st.session_state.exam_findings:
                    st.markdown(f"**{item['name']}**")
                    if item["normalized"]:
                        st.json(item["normalized"])
                    elif item["raw_text"]:
                        st.text(item["raw_text"])
                    if item["notes"]:
                        st.caption("; ".join(item["notes"]))

        imaging_files = st.file_uploader(
            "Radiografias (JPG, PNG, DICOM zip)",
            type=["png", "jpg", "jpeg", "zip", "dcm"],
            accept_multiple_files=True,
        )
        if imaging_files:
            for img in imaging_files:
                raw_bytes = img.getvalue()
                img.seek(0)
                result = st.session_state.radiography_service.analyze(img)
                if result and result["id"] not in st.session_state.imaging_ids:
                    st.session_state.imaging_findings.append(result)
                    st.session_state.imaging_ids.add(result["id"])
                    explain_note = st.session_state.explainability_engine.generate(result)
                    st.session_state.explainability_notes.append(explain_note)
                    dicom_engine = st.session_state.get("dicom_analyzer")
                    if dicom_engine and result["id"] not in st.session_state.dicom_ids:
                        dicom_payload = dicom_engine.analyze_bytes(
                            file_id=result["id"],
                            content=raw_bytes,
                        )
                        if dicom_payload:
                            st.session_state.dicom_findings.append(dicom_payload)
                            st.session_state.dicom_ids.add(result["id"])
        refresh_multiexam_reasoning()

        if st.session_state.imaging_findings:
            with st.expander("Achados de radiografia", expanded=False):
                for item in st.session_state.imaging_findings:
                    st.markdown(f"**{item['name']}**")
                    st.json(item["payload"])
                    if item["notes"]:
                        st.caption("; ".join(item["notes"]))

        if st.session_state.explainability_notes:
            with st.expander("Explicabilidade visual", expanded=False):
                for note in st.session_state.explainability_notes[-5:]:
                    st.write("- " + note)

        st.markdown("---")
        st.subheader("Educacao personalizada")
        render_education_cards()
        selected_category = st.selectbox(
            "Explorar categoria manualmente",
            options=["Selecione"] + st.session_state.education_manager.list_categories(),
            index=0,
        )
        if selected_category != "Selecione":
            for rec in EDUCATION_LIBRARY[selected_category]:
                st.markdown(
                    f"* `{selected_category}` -> **{rec['title']}** ({rec['type']})"
                )

        st.markdown("---")
        st.subheader("Ficha para imprimir")
        summary_text = st.session_state.printable_summary or build_symptom_report()
        if summary_text and "Sem sintomas registrados" not in summary_text:
            st.text_area(
                "Resumo de sintomas",
                value=summary_text,
                height=140,
                disabled=True,
            )
            qr_bytes = generate_qr_code(summary_text)
            if qr_bytes:
                safe_show_image(qr_bytes, caption="Compartilhe via QR Code", use_column_width=False)
            st.download_button(
                "Baixar sintomas (.txt)",
                summary_text.encode("utf-8"),
                file_name="medIA_sintomas.txt",
                mime="text/plain",
            )
        else:
            st.caption("Nenhum sintoma suficiente para gerar resumo imprimivel.")
        st.markdown("---")
        st.subheader("Monitoramento epidemiologico")
        if st.session_state.epidemiology_snapshot:
            st.json(st.session_state.epidemiology_snapshot)
        else:
            st.caption("Ainda sem dados agregados suficientes.")

        st.markdown("---")
        if st.session_state.active_learning_queue:
            with st.expander("Fila de aprendizado ativo", expanded=False):
                if st.button("Limpar fila", key="clear_active_learning"):
                    st.session_state.active_learning_queue = []
                    st.success("Fila de aprendizado esvaziada.")
                for idx, item in enumerate(st.session_state.active_learning_queue[-5:], 1):
                    st.markdown(f"**Caso {idx}**")
                    st.caption(item["reason"])
                    st.text(f"Pergunta: {item['user']}")
                    st.text(f"Resposta: {item['bot']}")
        else:
            st.caption("Sem itens na fila de aprendizado ativo no momento.")

        if st.session_state.confidence_history:
            last_conf = st.session_state.confidence_history[-1]
            st.metric(
                "Confianca da ultima resposta",
                f"{last_conf['label']} ({last_conf['score']})",
            )
            if len(st.session_state.confidence_history) > 3:
                scope = st.session_state.confidence_history[-3:]
                average = sum(item["score"] for item in scope) / len(scope)
                st.caption(f"Media das ultimas respostas: {average:.2f}")

        if st.session_state.planner_state:
            st.caption(
                "Plano atual: "
                + st.session_state.planner_state.get("stage", "")
                + " -> "
                + st.session_state.planner_state.get("next_action", "")
            )

        st.markdown("---")
        wearable_input = st.text_area(
            "Dados de wearables (cole JSON com batimentos, saturacao, etc.)",
            value=st.session_state.wearable_raw_input,
        )
        if wearable_input != st.session_state.wearable_raw_input:
            st.session_state.wearable_raw_input = wearable_input
            if wearable_input.strip():
                try:
                    st.session_state.wearable_payload = json.loads(wearable_input)
                except json.JSONDecodeError:
                    st.warning("JSON invalido para dados de wearables.")
            else:
                st.session_state.wearable_payload = {}
            refresh_multiexam_reasoning()

        st.markdown("---")
        st.subheader("Multimodal")
        voice_text = st.text_area(
            "Transcricao de voz (cole resultado do STT)",
            key="voice_transcript_area",
        )
        if st.button("Enviar transcricao para o chat"):
            if voice_text.strip():
                st.session_state.pending_voice_input = voice_text.strip()
                st.success("Transcricao enviada para o chat.")
            else:
                st.warning("Informe um texto antes de enviar.")

        st.checkbox("Gerar audio das respostas (usa gTTS se disponivel)", key="audio_toggle")

        st.markdown("---")
        if st.button("Salvar resumo da consulta"):
            record = st.session_state.history_manager.save(
                st.session_state.history,
                {
                    "exam_findings": st.session_state.exam_findings,
                    "imaging_findings": st.session_state.imaging_findings,
                    "medication_alerts": st.session_state.medication_alerts,
                },
            )
            st.success("Resumo armazenado.")
            st.json(record)

        if st.session_state.consultation_log:
            with st.expander("Historico inteligente", expanded=False):
                for idx, record in enumerate(st.session_state.consultation_log, start=1):
                    st.markdown(f"**Consulta {idx}**")
                    st.text(record["summary"])


def render_voice_agent_panel() -> None:
    st.markdown(
        """
        <div class='themed-panel' style='text-align:center; background: linear-gradient(120deg, rgba(99,102,241,0.12), rgba(6,182,212,0.12));'>
            <h3 style='margin-bottom:0.3em;'>🎤 MedIA Voz com Gemini</h3>
            <p style='font-size:0.95em; color:#475569; margin:0 auto; max-width:520px;'>
                Grave uma mensagem curta e deixe o Gemini AI Studio transcrever e responder em tempo real.
                A transcrição é enviada para o chat principal para manter todo mundo sincronizado.
            </p>
            <p style='font-size:0.8em; color:#94a3b8; margin:0 auto; max-width:520px;'>
                Usa o modelo <strong>gemini-2.5-flash</strong> (1,048,576 tokens de entrada, 65,536 de saída) compatível com texto, imagem, vídeo e áudio.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not gemini_sdk_available():
        st.warning(
            "Instale `google-genai` (SDK oficial) ou `google-generativeai` para habilitar o agente de voz."
        )
        return

    agent = get_voice_agent()
    if agent is None or not agent.available():
        st.warning(
            "Configure a variavel `GEMINI_API_KEY` (ou use a integrada no codigo) para habilitar o agente de voz."
        )
        return

    audio_file = st.audio_input(
        "Grave sua pergunta (max. 30s). Toque no microfone, aguarde e clique em 'Enviar voz'.",
        key="voice_audio_input",
    )
    send_col, clear_col = st.columns([3, 1])
    with send_col:
        send_voice = st.button(
            "Enviar mensagem de voz",
            key="send_voice_button",
            use_container_width=True,
        )
    with clear_col:
        if st.button(
            "Limpar voz",
            key="clear_voice_history",
            use_container_width=True,
        ):
            st.session_state.voice_conversation = []
            st.session_state.voice_agent_status = "Histórico de voz reiniciado."
            st.success("Histórico do agente de voz limpo.")

        if send_voice:
            if audio_file is None:
                st.warning("Grave uma mensagem antes de enviar.")
            else:
                audio_bytes = audio_file.getvalue()
                mime_type = audio_file.type or "audio/wav"
                result: Optional[Dict[str, Any]] = None
                transcript = ""
                reply = ""
                try:
                    with st.spinner("Conversando com o Gemini..."):
                        result = agent.transcribe_and_respond(
                            audio=audio_bytes,
                            mime_type=mime_type,
                        )
                except Exception as exc:
                    error_text = str(exc)
                    st.session_state.voice_agent_status = f"Erro ao acionar Gemini: {error_text}"
                    if "404" in error_text and "models" in error_text:
                        st.error(
                            "Modelo Gemini não encontrado para esta API. Confira os nomes suportados executando "
                            "`client.models.list()` ou defina `GEMINI_MODEL_NAME=models/gemini-1.5-flash`."
                        )
                    else:
                        st.error(
                            "Não foi possível entender o áudio agora. Verifique a conexão e tente novamente."
                        )
                if result:
                    transcript = result.get("transcription", "")
                    reply = result.get("assistant_reply", "")
                    latency = result.get("latency", 0.0)
                    entry: Dict[str, Any] = {
                        "patient": transcript,
                        "agent": reply,
                        "latency": latency,
                    }
                    audio_answer = generate_tts_audio(reply) if reply else None
                    if audio_answer:
                        entry["audio"] = audio_answer
                    history = list(st.session_state.voice_conversation)
                    history.append(entry)
                    st.session_state.voice_conversation = history[-6:]
                    st.session_state.voice_agent_status = (
                        f"Última resposta em {latency:.1f}s usando {agent.model_name}."
                    )
                    if transcript:
                        st.session_state.pending_voice_input = transcript
                        st.info("Transcrição enviada automaticamente para o chat principal.")

    if st.session_state.voice_agent_status:
        st.caption(st.session_state.voice_agent_status)

    if not st.session_state.voice_conversation:
        st.caption("Interaja por voz para ver transcrições e respostas aqui.")
        return

    st.markdown("#### Histórico de voz")
    for idx, item in enumerate(reversed(st.session_state.voice_conversation), start=1):
        st.markdown(
            f"""
            <div class='themed-panel' style='padding:16px; margin-bottom:8px; text-align:left;'>
                <div style='font-size:0.85em; color:#94a3b8;'>Interação #{len(st.session_state.voice_conversation) - idx + 1}</div>
                <p><strong>Paciente:</strong> {item.get('patient', '---')}</p>
                <p><strong>MedIA Voz:</strong> {item.get('agent', '---')}</p>
                <p style='font-size:0.85em; color:#94a3b8;'>Latência: {item.get('latency', 0.0):.1f}s</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if item.get("audio"):
            st.audio(item["audio"], format="audio/mp3")


def render_quick_prompt_bar() -> None:
    if not QUICK_PROMPTS:
        return
    st.markdown("#### Sugestoes rapidas")
    st.caption("Clique em um atalho para preencher o chat com uma pergunta comum.")
    columns = st.columns(len(QUICK_PROMPTS))
    triggered_label: Optional[str] = None
    for idx, prompt in enumerate(QUICK_PROMPTS):
        label = prompt["label"]
        text = prompt["text"]
        if columns[idx].button(label, key=f"quick_prompt_{idx}"):
            st.session_state.pending_voice_input = text
            triggered_label = label
    if triggered_label:
        st.success(f"Atalho '{triggered_label}' enviado para o chat. Ajuste o texto antes de enviar se desejar.")


def render_hackathon_triage_tab() -> None:
    st.header("🧠 Hackathon Saude Inteligente")
    st.caption(
        "Fluxo dedicado para o desafio de triagem de enfermagem: coleta estruturada → processamento interpretavel → relatorio automatizado."
    )
    st.markdown(
        """
        **Diretrizes principais**

        - Enfermeiros registram sinais vitais e antecedentes básicos em poucos toques.
        - O motor de regras do MedIA gera alertas transparentes e sugere encaminhamentos.
        - O relatório pode ser compartilhado com o time médico ou exportado como JSON.
        """,
    )

    if st.button("Carregar exemplo oficial do Hackathon", key="load_hackathon_sample"):
        st.session_state.hackathon_triage_payload = example_triage_payload()
        st.experimental_rerun()

    payload_hint = st.session_state.get("hackathon_triage_payload") or example_triage_payload()
    chronic_default = payload_hint.get("chronic_conditions", [])
    allergies_default = payload_hint.get("allergies", [])
    meds_default = payload_hint.get("medications", [])
    symptoms_default = "\n".join(payload_hint.get("symptoms", []))

    with st.form("hackathon_triage_form"):
        col_left, col_right = st.columns(2)
        systolic = col_left.number_input(
            "Pressao sistolica (mmHg)",
            min_value=0,
            max_value=300,
            value=int(payload_hint.get("systolic") or 0),
        )
        diastolic = col_left.number_input(
            "Pressao diastolica (mmHg)",
            min_value=0,
            max_value=200,
            value=int(payload_hint.get("diastolic") or 0),
        )
        heart_rate = col_left.number_input(
            "Frequencia cardiaca (bpm)",
            min_value=0,
            max_value=250,
            value=int(payload_hint.get("heart_rate") or 0),
        )
        temperature = col_left.number_input(
            "Temperatura corporal (°C)",
            min_value=0.0,
            max_value=45.0,
            step=0.1,
            value=float(payload_hint.get("temperature") or 0.0),
        )
        spo2 = col_left.number_input(
            "Saturacao de O2 (%)",
            min_value=0,
            max_value=100,
            value=int(payload_hint.get("spo2") or 0),
        )

        age = col_right.number_input(
            "Idade",
            min_value=0,
            max_value=120,
            value=int(payload_hint.get("age") or 0),
        )
        sex_options = ["Não informado", "Feminino", "Masculino"]
        sex_default = payload_hint.get("sex", "Não informado")
        if sex_default not in sex_options:
            sex_default = "Não informado"
        sex = col_right.selectbox(
            "Sexo biologico",
            options=sex_options,
            index=sex_options.index(sex_default),
        )
        blood_options = ["Não informado", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        blood_default = payload_hint.get("blood_type") or "Não informado"
        if blood_default not in blood_options:
            blood_default = "Não informado"
        blood_type = col_right.selectbox(
            "Tipo sanguineo (opcional)",
            options=blood_options,
            index=blood_options.index(blood_default),
        )
        chronic_conditions = col_right.multiselect(
            "Doencas cronicas",
            options=COMMON_CHRONIC_CONDITIONS,
            default=[c for c in chronic_default if c in COMMON_CHRONIC_CONDITIONS],
            help="Acrescente outras no campo texto abaixo.",
        )
        chronic_extra = col_right.text_input(
            "Outras doencas cronicas",
            value=", ".join([c for c in chronic_default if c not in COMMON_CHRONIC_CONDITIONS]),
        )

        allergies = st.multiselect(
            "Alergias registradas",
            options=COMMON_ALLERGIES,
            default=[a for a in allergies_default if a in COMMON_ALLERGIES],
        )
        allergy_extra = st.text_input(
            "Outras alergias",
            value=", ".join([a for a in allergies_default if a not in COMMON_ALLERGIES]),
        )
        medications = st.multiselect(
            "Medicacoes de uso continuo",
            options=COMMON_MEDICATIONS,
            default=[m for m in meds_default if m in COMMON_MEDICATIONS],
        )
        meds_extra = st.text_input(
            "Outras medicacoes",
            value=", ".join([m for m in meds_default if m not in COMMON_MEDICATIONS]),
        )
        symptoms_text = st.text_area(
            "Sintomas relatados",
            value=symptoms_default,
            placeholder="Ex.: dor no peito, febre acima de 39°C, falta de ar ao repouso...",
        )

        submitted = st.form_submit_button(
            "Gerar relatorio automatizado",
            use_container_width=True,
        )

    if submitted:
        payload = NursingTriageInput(
            systolic=sanitize_numeric(systolic),
            diastolic=sanitize_numeric(diastolic),
            heart_rate=sanitize_numeric(heart_rate),
            temperature=sanitize_numeric(temperature),
            spo2=sanitize_numeric(spo2),
            age=int(age) if age else None,
            sex=sex,
            chronic_conditions=chronic_conditions + parse_free_text_list(chronic_extra),
            allergies=allergies + parse_free_text_list(allergy_extra),
            blood_type=None if blood_type == "Não informado" else blood_type,
            medications=medications + parse_free_text_list(meds_extra),
            symptoms=parse_free_text_list(symptoms_text),
        )
        report = generate_triage_report(payload)
        st.session_state.hackathon_triage_report = report
        st.session_state.hackathon_triage_payload = payload.to_dict()
        st.success("Relatorio gerado com sucesso! Veja os detalhes abaixo.")

    report_data = st.session_state.get("hackathon_triage_report")
    if report_data:
        report_dict = (
            report_data.to_dict()
            if hasattr(report_data, "to_dict")
            else report_data
        )
        summary = (
            report_data.summary_markdown()
            if hasattr(report_data, "summary_markdown")
            else ""
        )
        if summary:
            st.markdown(summary)

        score = report_dict.get("processamento", {}).get("pontuacao", 0)
        st.progress(min(score / 3, 1.0))

        download_payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        st.download_button(
            "Baixar JSON da triagem",
            data=download_payload.encode("utf-8"),
            file_name="hackathon_triage_report.json",
            mime="application/json",
        )

        with st.expander("Camada: Coleta"):
            st.json(report_dict.get("coleta", {}))
        with st.expander("Camada: Processamento e trilha de auditoria"):
            st.json(report_dict.get("processamento", {}))
        with st.expander("Camada: Relatorio final"):
            st.json(report_dict.get("relatorio", {}))
    else:
        st.info("Preencha o formulario e clique em 'Gerar relatorio automatizado' para visualizar a triagem.")


def render_history() -> None:
    st.subheader("Respostas do MedIA")
    if st.session_state.history:
        st.markdown(
            "<div id='chat-history' style='display: flex; flex-direction: column;'>"
            + "<hr>".join(st.session_state.history)
            + "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="MedIA",
        page_icon="🩺",
        layout="centered",
    )

    st.title("MedIA")
    if _groq_import_error is not None:
        st.error(
            "O SDK `groq` nao foi encontrado. Instale-o com `pip install groq` para usar a integracao com o modelo."
        )
        st.code(str(_groq_import_error))
        return

    ensure_session_defaults()
    register_external_services()
    apply_theme_settings()
    render_sidebar()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        try:
            groq_api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not groq_api_key:
        st.error("GROQ_API_KEY nao encontrado. Configure a chave nas variaveis de ambiente ou em st.secrets.")
        return
    default_model = "llama-3.3-70b-versatile"
    model_aliases = {
        "llama3-8b-8192": "llama-3.1-8b-instant",
        "llama3-70b-8192": "llama-3.3-70b-versatile",
    }
    model = os.environ.get("GROQ_MODEL_NAME", "")
    if not model:
        try:
            if "GROQ_MODEL_NAME" in st.secrets:
                model = st.secrets["GROQ_MODEL_NAME"]
        except Exception:
            pass
    original_model = model or default_model
    resolved_model = model_aliases.get(original_model, original_model)
    if resolved_model != original_model:
        st.warning(
            f"Modelo '{original_model}' foi substituido automaticamente por '{resolved_model}'. "
            "Atualize GROQ_MODEL_NAME para evitar esta mensagem."
        )
    model = resolved_model
    try:
        groq_client = Groq(api_key=groq_api_key)
    except Exception as exc:
        st.error("Falha ao inicializar o cliente Groq. Verifique a chave de API.")
        st.code(str(exc))
        return

    st.write(
        "Sou um sistema de apoio medico com analise de exames e radiografias. "
        "Sempre consulte um profissional de saude para confirmacao."
    )

    system_prompt = (
        "Voce e um assistente medico virtual flexivel. Quando o usuario relatar sintomas e desejar suporte clinico, "
        "ofereca realizar uma triagem estruturada com ate 10 perguntas numeradas, mas permita que ele interrompa ou "
        "pule etapas a qualquer momento. Nao limite a conversa a triagem: responda imediatamente perguntas diretas "
        "sobre doencas, sintomas, exames, medicamentos, prevencao ou orientacoes gerais, mesmo durante a triagem. "
        "Use os conteudos de exames, radiografias e dados de wearables enviados para enriquecer a resposta quando possivel. "
        "Sempre forneca diagnosticos diferenciais provaveis, exames complementares sugeridos e orientacoes de cuidado, "
        "indicando quando um especialista humano deve ser consultado. Caso o usuario solicite calculo de IMC ou outras "
        "informacoes especificas, atenda prontamente e retome a triagem apenas se ele quiser prosseguir. "
        "Quando detectar sinais criticos, priorize orientacao emergencial. "
        "Mantenha o dialogo aberto apos concluir as perguntas, dando continuidade a duvidas ou novas solicitacoes sem forcar reinicio."
    )

    triage_tab, patient_tab, insights_tab = st.tabs(
        ["Triagem", "Painel do paciente", "Insights"]
    )

    with triage_tab:
        render_voice_agent_panel()
        render_quick_prompt_bar()
        st.markdown("<br>", unsafe_allow_html=True)

        render_hackathon_triage_tab()
        st.markdown("<br>", unsafe_allow_html=True)

        render_history()
        st.markdown("""<script>
const chat = document.getElementById(\'chat-history\');
if (chat) { chat.scrollTop = chat.scrollHeight; }
</script>""", unsafe_allow_html=True)

        user_input = st.chat_input("💬 Digite seus sintomas ou faça uma pergunta médica...", key="user_input")
        if not user_input and st.session_state.pending_voice_input:
            user_input = st.session_state.pending_voice_input
            st.session_state.pending_voice_input = ""

        if user_input:
            st.session_state.history.append(
                f"<div class='message user-message'><strong>Voce:</strong> {user_input}</div>"
            )

            normalized_input = user_input.strip().lower()
            if normalized_input in {"limpar conversa", "reset", "reiniciar"}:
                st.session_state.history = []
                clear_conversation_memory()
                st.session_state.symptom_log = []
                st.session_state.education_recommendations = []
                st.session_state.medication_alerts = []
                st.session_state.active_learning_queue = []
                st.session_state.confidence_history = []
                st.session_state.planner_state = {}
                st.session_state.multimodal_signature = {}
                st.session_state.pending_voice_input = ""
                st.session_state.audio_responses = []
                reset_question_progress()
                st.session_state.printable_summary = ""
                st.session_state.education_checklist = {}
                st.session_state.epidemiology_snapshot = {}
                st.session_state.critical_events = []
                st.session_state.triage_mode = False
                st.session_state.advanced_lab_findings = []
                st.session_state.advanced_lab_ids = set()
                st.session_state.dicom_findings = []
                st.session_state.dicom_ids = set()
                st.session_state.cross_validation_notes = []
                st.session_state.cross_conflicts = []
                st.session_state.ecg_insights = []
                st.session_state.ecg_ids = set()
                st.session_state.guideline_recommendations = []
                st.session_state.guideline_plan = []
                st.session_state.guideline_cha2ds2 = None
                st.session_state.guideline_curb65 = None
                refresh_multiexam_reasoning()
                st.success("Conversa e contexto reiniciados.")
                st.rerun()

            start_keywords = [
                "iniciar triagem",
                "começar triagem",
                "comecar triagem",
                "fazer triagem",
                "continuar triagem",
                "retomar triagem",
            ]
            stop_keywords = [
                "parar triagem",
                "encerrar triagem",
                "cancelar triagem",
                "finalizar triagem",
                "sem triagem",
            ]
            start_requested = any(keyword in normalized_input for keyword in start_keywords)
            stop_requested = any(keyword in normalized_input for keyword in stop_keywords)

            general_keywords = [
                "remedio",
                "remédio",
                "medicamento",
                "bula",
                "posologia",
                "dose",
                "efeito",
                "tratamento",
                "doenca",
                "doença",
                "diagnostico",
                "diagnóstico",
                "exame",
                "resultado",
                "sintoma",
                "prevenir",
            ]
            is_direct_query = any(keyword in normalized_input for keyword in general_keywords)

            symptom_candidates = extract_symptom_candidates(user_input)
            if symptom_candidates:
                st.session_state.symptom_log.append(
                    {"raw": user_input, "symptoms": symptom_candidates}
                )

            if stop_requested:
                if st.session_state.triage_mode:
                    st.session_state.triage_mode = False
                    reset_question_progress()
            elif start_requested:
                if not st.session_state.triage_mode:
                    st.session_state.triage_mode = True
                    reset_question_progress()
            elif (
                not st.session_state.triage_mode
                and symptom_candidates
                and any(trigger in normalized_input for trigger in ["dor", "sinto", "tenho", "estou com", "sentindo", "sintomas"])
            ):
                st.session_state.triage_mode = True
                reset_question_progress()

            if st.session_state.triage_mode and is_direct_query and not start_requested:
                st.session_state.triage_mode = False
                reset_question_progress()

            med_key = find_medication_query(user_input)
            if med_key:
                st.session_state.triage_mode = False
                reset_question_progress()
                med_response = generate_medication_response(med_key)
                if med_response:
                    st.session_state.history.append(
                        f"<div class='message ai-message'><strong>MedIA:</strong> {med_response}</div>"
                    )
                    st.session_state.printable_summary = build_symptom_report()
                    st.rerun()
                    return

            if re.search(r"\bquais?\b.*\bsintoma", normalized_input):
                summary = summarize_symptom_log(st.session_state.symptom_log)
                response_text = (
                    summary
                    + "\n\nSempre consulte um profissional de saude para interpretacao completa."
                )
                st.session_state.history.append(
                    f"<div class='message ai-message'><strong>MedIA:</strong> {response_text}</div>"
                )
                st.rerun()
                return

            st.session_state.epidemiology_monitor.ingest(user_input)
            st.session_state.epidemiology_snapshot = st.session_state.epidemiology_monitor.export()

            emergency_message = st.session_state.emergency_router.detect(user_input)
            if emergency_message:
                st.session_state.history.append(
                    f"<div class='message ai-message emergency'><strong>MedIA:</strong> {emergency_message}</div>"
                )
                st.session_state.critical_events.append(emergency_message)
                st.rerun()

            medication_alerts = st.session_state.med_checker.check(user_input)
            if medication_alerts:
                st.session_state.medication_alerts.extend(medication_alerts)
                st.session_state.history.append(
                    "<div class='message ai-message alert'><strong>MedIA:</strong> "
                    + " ".join(medication_alerts)
                    + "</div>"
                )

            planner_state = st.session_state.conversation_planner.plan(
                user_text=user_input,
                state={
                    "exam_findings": st.session_state.exam_findings,
                    "imaging_findings": st.session_state.imaging_findings,
                    "wearable_payload": st.session_state.wearable_payload,
                },
            )
            st.session_state.planner_state = planner_state

            fusion_data = st.session_state.fusion_engine.fuse(
                user_text=user_input,
                exam_findings=st.session_state.exam_findings,
                imaging_findings=st.session_state.imaging_findings,
                wearable_payload=st.session_state.wearable_payload,
            )
            st.session_state.multimodal_signature = fusion_data

            if st.session_state.triage_mode and not is_direct_query:
                progress = st.session_state.question_progress
                history = progress.setdefault("history", [])
                if len(history) < progress["total"]:
                    history.extend([""] * (progress["total"] - len(history)))
                current = progress.get("current", 1)
                if 1 <= current <= progress["total"] and history[current - 1]:
                    progress["answered"] = max(progress["answered"], current)
                progress["answered"] = min(progress["answered"], progress["total"])
                progress["current"] = min(progress["answered"] + 1, progress["total"])
                st.session_state.question_progress = progress
            elif not st.session_state.triage_mode:
                reset_question_progress()

            context_payload, context_meta = build_context_sections()
            request_type = (
                "pergunta direta"
                if is_direct_query
                else ("fluxo de triagem" if st.session_state.triage_mode else "conversa geral")
            )
            meta_lines = [
                f"Status da triagem: {'ativo' if st.session_state.triage_mode else 'inativo'}",
                f"Tipo de interacao: {request_type}",
            ]
            meta_block = "\n".join(meta_lines)
            composed_input = meta_block + "\n\nEntrada do paciente: " + user_input
            if context_payload:
                composed_input = f"{context_payload}\n\n{meta_block}\n\nEntrada do paciente: {user_input}"

            append_memory_message("user", composed_input)
            model_messages = build_model_messages(system_prompt)
            model_error_context = {}
            try:
                response = predict_with_fallback(groq_client, model, model_messages)
                append_memory_message("assistant", response)
            except BadRequestError as exc:
                pop_last_memory_message()
                error_text = str(exc)
                model_error_context = {}
                if "model_decommissioned" in error_text:
                    friendly = (
                        "MedIA: o modelo configurado foi descontinuado. "
                        "Defina `GROQ_MODEL_NAME` para um modelo suportado como "
                        "`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, ou outro da lista atual da Groq "
                        "e tente novamente."
                    )
                    st.error(
                        "Modelo Groq configurado foi descontinuado. Atualize `GROQ_MODEL_NAME` para um modelo suportado "
                        "(ex.: llama-3.3-70b-versatile ou llama-3.1-8b-instant)."
                    )
                    model_error_context["type"] = "model_decommissioned"
                    model_error_context["detail"] = error_text
                else:
                    friendly = (
                        "MedIA: nao foi possivel gerar uma resposta agora porque o pedido excedeu os limites "
                        "do modelo. Remova alguns anexos ou reduza o texto e tente novamente."
                    )
                    st.error("Falha ao acionar o modelo Groq (BadRequest). Ajuste o contexto e tente de novo.")
                    model_error_context["type"] = "context_limit"
                    model_error_context["detail"] = error_text
                st.session_state.history.append(
                    f"<div class='message ai-message error'><strong>MedIA:</strong> {friendly}</div>"
                )
                st.session_state.active_learning_queue.append(
                    {
                        "user": user_input,
                        "bot": friendly,
                        "reason": f"BadRequestError: {exc}",
                    }
                )
                return
            except Exception as exc:  # pragma: no cover - resiliencia
                pop_last_memory_message()
                friendly = (
                    "MedIA encontrou um erro inesperado ao gerar a resposta. "
                    "Atualize ou tente novamente em instantes."
                )
                st.session_state.history.append(
                    f"<div class='message ai-message error'><strong>MedIA:</strong> {friendly}</div>"
                )
                st.error(f"Erro inesperado ao consultar o modelo: {exc}")
                return

            enriched_response = st.session_state.validator.attach_disclaimer(response, context_meta)
            confidence_result = st.session_state.confidence_calibrator.score(enriched_response, context_meta)
            st.session_state.confidence_history.append(confidence_result)
            final_response = (
                f"{enriched_response}\n\n"
                f"_Confianca estimada na resposta: {confidence_result['label']} "
                f"({confidence_result['score']})._"
            )

            education_hits = st.session_state.education_manager.recommend_from_text(
                text=f"{user_input}\n{final_response}"
            )
            if education_hits:
                merged = st.session_state.education_recommendations + education_hits
                seen: set = set()
                deduped = []
                for item in merged:
                    key = (item["title"], item["url"])
                    if key not in seen:
                        deduped.append(item)
                        seen.add(key)
                st.session_state.education_recommendations = deduped

            try:
                flagged = st.session_state.active_learning_tracker.should_flag(
                    user_input,
                    final_response,
                    metadata=model_error_context,
                )
            except TypeError:
                flagged = st.session_state.active_learning_tracker.should_flag(
                    user_input,
                    final_response,
                )
            if flagged:
                st.session_state.active_learning_queue.append(flagged)

            st.session_state.history.append(
                f"<div class='message ai-message'><strong>MedIA:</strong> {final_response}</div>"
            )

            if st.session_state.triage_mode:
                update_question_progress(final_response)
            st.session_state.printable_summary = build_symptom_report()

            if st.session_state.audio_toggle:
                audio_bytes = generate_tts_audio(final_response)
                if audio_bytes:
                    st.session_state.audio_responses.append(audio_bytes)
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("gTTS nao disponivel ou falha ao gerar audio.")

            st.rerun()
    with patient_tab:
        render_patient_dashboard()

    with insights_tab:
        render_wearable_insights()
        render_explainability_panel()
        render_advanced_insights()

    st.markdown(
        """
        <style>
            #chat-history {
                height: calc(100vh - 260px);
                overflow-y: auto;
                padding-bottom: 160px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()










