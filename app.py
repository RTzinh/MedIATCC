import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
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


def ensure_session_defaults() -> None:
    defaults = {
        "memory": ConversationBufferWindowMemory(
            k=50_000, memory_key="chat_history", return_messages=True
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
    context_payload = "\n\n".join(pieces)
    context_meta = {
        "critical_flags": st.session_state.critical_events,
        "medication_alerts": medication_alerts,
        "exam_findings": st.session_state.exam_findings,
        "imaging_findings": st.session_state.imaging_findings,
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

        if st.session_state.imaging_findings:
            with st.expander("Achados de radiografia", expanded=False):
                for item in st.session_state.imaging_findings:
                    st.markdown(f"**{item['name']}**")
                    st.json(item["payload"])
                    if item["notes"]:
                        st.caption("; ".join(item["notes"]))

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
    model = "llama3-8b-8192"
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

        context_payload, context_meta = build_context_sections()
        composed_input = user_input
        if context_payload:
            composed_input = f"{context_payload}\n\nEntrada do paciente: {user_input}"

        response = conversation.predict(human_input=composed_input)
        final_response = st.session_state.validator.attach_disclaimer(response, context_meta)
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
