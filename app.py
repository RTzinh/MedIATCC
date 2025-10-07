import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from groq import BadRequestError
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_groq import ChatGroq

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


def ensure_session_defaults() -> None:
    defaults = {
        "memory": ConversationBufferWindowMemory(
            k=60, memory_key="chat_history", return_messages=True
        ),
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

    def _extract_content(self, content: bytes, ext: str) -> (str, List[str]):
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

    def _call_service(self, content: bytes, mime: Optional[str]) -> (Dict[str, Any], List[str]):
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
    )

    def should_flag(self, user_text: str, bot_text: str) -> Optional[Dict[str, str]]:
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


def predict_with_fallback(conversation: LLMChain, text: str) -> str:
    limits = [7000, 5000, 3000]
    windows = [60, 40, 20]
    memory = st.session_state.memory
    original_messages = list(getattr(memory.chat_memory, "messages", []))
    last_error: Optional[BadRequestError] = None

    for limit, window in zip(limits, windows):
        trimmed_text = truncate_text(text, limit=limit)
        if original_messages:
            subset = original_messages[-window:] if window else original_messages
            memory.chat_memory.messages = list(subset)
        try:
            response = conversation.predict(human_input=trimmed_text)
            if original_messages:
                if window and len(original_messages) > window:
                    memory.chat_memory.messages = list(original_messages[-window:])
                else:
                    memory.chat_memory.messages = original_messages
            return response
        except BadRequestError as exc:
            last_error = exc
            continue

    if original_messages:
        memory.chat_memory.messages = original_messages
    if last_error is not None:
        raise last_error
    raise BadRequestError("Context length exceeded (fallback attempts exhausted).")


def build_context_sections() -> (str, Dict[str, Any]):
    exam_context = st.session_state.exam_pipeline.render_for_prompt(st.session_state.exam_findings)
    imaging_context = st.session_state.radiography_service.render_for_prompt(
        st.session_state.imaging_findings
    )
    wearable_payload = st.session_state.wearable_payload or {}
    wearable_context = ""
    if wearable_payload:
        wearable_context = "Dados de wearables: " + json.dumps(wearable_payload)[:800]
    medication_alerts = st.session_state.medication_alerts
    pieces = []
    if exam_context:
        pieces.append("Resumo de exames estruturados:\n" + exam_context)
    if imaging_context:
        pieces.append("Resumo de radiografias:\n" + imaging_context)
    if wearable_context:
        pieces.append(wearable_context)
    if medication_alerts:
        pieces.append("Alertas farmacologicos ativos: " + "; ".join(sorted(set(medication_alerts))))
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


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Fluxos auxiliares")

        exam_files = st.file_uploader(
            "Envie exames estruturados (PDF, CSV, HL7, imagem)",
            type=["pdf", "csv", "hl7", "json", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if exam_files:
            for exam in exam_files:
                result = st.session_state.exam_pipeline.process(exam)
                if result and result["id"] not in st.session_state.exam_ids:
                    st.session_state.exam_findings.append(result)
                    st.session_state.exam_ids.add(result["id"])

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
                result = st.session_state.radiography_service.analyze(img)
                if result and result["id"] not in st.session_state.imaging_ids:
                    st.session_state.imaging_findings.append(result)
                    st.session_state.imaging_ids.add(result["id"])
                    explain_note = st.session_state.explainability_engine.generate(result)
                    st.session_state.explainability_notes.append(explain_note)

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
        if st.session_state.education_recommendations:
            for rec in st.session_state.education_recommendations[:5]:
                st.markdown(
                    f"- **{rec['title']}** ({rec['type']}) — [Acessar]({rec['url']})"
                )
        else:
            st.caption("Recomende exames ou descreva sintomas para obter materiais.")
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
        st.subheader("Monitoramento epidemiologico")
        if st.session_state.epidemiology_snapshot:
            st.json(st.session_state.epidemiology_snapshot)
        else:
            st.caption("Ainda sem dados agregados suficientes.")

        st.markdown("---")
        if st.session_state.active_learning_queue:
            with st.expander("Fila de aprendizado ativo", expanded=False):
                for idx, item in enumerate(st.session_state.active_learning_queue[-5:], 1):
                    st.markdown(f"**Caso {idx}**")
                    st.caption(item["reason"])
                    st.text(f"Pergunta: {item['user']}")
                    st.text(f"Resposta: {item['bot']}")

        if st.session_state.confidence_history:
            last_conf = st.session_state.confidence_history[-1]
            st.metric(
                "Confianca da ultima resposta",
                f"{last_conf['label']} ({last_conf['score']})",
            )

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
        page_icon="aaaaaaaaaaaaaaa.png",
        layout="centered",
    )

    ensure_session_defaults()
    render_sidebar()

    groq_api_key = st.secrets["GROQ_API_KEY"]
    default_model = "llama-3.3-70b-versatile"
    model_aliases = {
        "llama3-8b-8192": "llama-3.1-8b-instant",
        "llama3-70b-8192": "llama-3.3-70b-versatile",
    }
    model = os.environ.get("GROQ_MODEL_NAME", "")
    if not model and "GROQ_MODEL_NAME" in st.secrets:
        model = st.secrets["GROQ_MODEL_NAME"]
    original_model = model or default_model
    resolved_model = model_aliases.get(original_model, original_model)
    if resolved_model != original_model:
        st.warning(
            f"Modelo '{original_model}' foi substituido automaticamente por '{resolved_model}'. "
            "Atualize GROQ_MODEL_NAME para evitar esta mensagem."
        )
    model = resolved_model
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    st.title("MedIA")
    st.write(
        "Sou um sistema de apoio medico com analise de exames e radiografias. "
        "Sempre consulte um profissional de saude para confirmacao."
    )

    system_prompt = (
        "Voce e um especialista em triagem medica digital. "
        "Siga a rotina de 10 perguntas, numerando uma a uma, e somente entregue diagnostico provavel apos obter todas. "
        "Use os conteudos de exames estruturados, radiografias e dados de wearables fornecidos no contexto para personalizar a conversa. "
        "Sempre inclua lista de diagnosticos diferenciais, exames complementares recomendados e orientacoes de cuidado. "
        "Se o usuario pedir calculo de IMC, execute e aguarde nova solicitacao para retomar o protocolo de perguntas. "
        "Forneca alertas de medicacao usando o contexto de interacoes. "
        "Se perguntas sobre sinais e sintomas especificos surgirem, responda diretamente sem bloquear o fluxo. "
        "Quando detectar sinais criticos, priorize orientacao emergencial. "
        "Inclua recomendacoes sobre buscar especialista humano ao interpretar exames e imagens."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=st.session_state.memory,
    )

    render_history()

    user_input = st.chat_input("Digite seus sintomas", key="user_input")
    if not user_input and st.session_state.pending_voice_input:
        user_input = st.session_state.pending_voice_input
        st.session_state.pending_voice_input = ""

    if user_input:
        st.session_state.history.append(
            f"<div class='message user-message'><strong>Voce:</strong> {user_input}</div>"
        )

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

        context_payload, context_meta = build_context_sections()
        composed_input = user_input
        if context_payload:
            composed_input = f"{context_payload}\n\nEntrada do paciente: {user_input}"

        try:
            response = predict_with_fallback(conversation, composed_input)
        except BadRequestError as exc:
            error_text = str(exc)
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
            else:
                friendly = (
                    "MedIA: nao foi possivel gerar uma resposta agora porque o pedido excedeu os limites "
                    "do modelo. Remova alguns anexos ou reduza o texto e tente novamente."
                )
                st.error("Falha ao acionar o modelo Groq (BadRequest). Ajuste o contexto e tente de novo.")
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
            f"{enriched_response}\n\nConfianca estimada: {confidence_result['label']} "
            f"({confidence_result['score']})."
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

        flagged = st.session_state.active_learning_tracker.should_flag(user_input, final_response)
        if flagged:
            st.session_state.active_learning_queue.append(flagged)

        st.session_state.history.append(
            f"<div class='message ai-message'><strong>MedIA:</strong> {final_response}</div>"
        )

        if st.session_state.audio_toggle:
            audio_bytes = generate_tts_audio(final_response)
            if audio_bytes:
                st.session_state.audio_responses.append(audio_bytes)
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning("gTTS nao disponivel ou falha ao gerar audio.")

        st.rerun()

    st.markdown(
        """
        <style>
            #chat-history {
                height: calc(100vh - 150px);
                overflow-y: auto;
                padding-bottom: 60px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()





