# MedIA

MedIA e um assistente clinico multimodal construido como Trabalho de Conclusao de Curso por **Ryan Tereciani** e **Reuel Amador Mantovani**.  
O projeto integra analise de exames, conversacao com IA e um fluxo dedicado ao **Hackathon Saude Inteligente – Triagem Medica com IA**, direcionado a equipes de enfermagem que precisam gerar relatorios rapidos, auditaveis e acionaveis.

---

## Visao Geral

- Triagem conversacional com Groq (modelos Llama 3.x) e persistencia de contexto medico.
- Upload e interpretacao de exames laboratoriais e DICOM via `med_modules/`.
- Painel de voz com Gemini AI Studio para conversas hands-free.
- Aba “Perfil & triagem” consolidada: captura dados do paciente + sinais vitais e ja dispara o motor do Hackathon.
- Aba “Relatorio” que produz automaticamente o texto final (com download em `.txt`).
- Suporte opcional ao Supabase para registrar pacientes, triagens e relatorios auditaveis.

---

## Tecnologias

- Python 3.12
- Streamlit >= 1.25
- Groq SDK (`groq`) e Google Generative AI (`google-generativeai` / `google-genai`)
- PyPDF2, Pillow, pytesseract, pydicom, pyedflib, gTTS
- Supabase Python client (opcional) para persistencia

---

## Como Executar

```bash
git clone https://github.com/SEU-USUARIO/MedIATCC.git
cd MedIATCC
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

set GROQ_API_KEY=seu_token
set GEMINI_API_KEY=seu_token    # opcional (uma chave de teste ja existe no codigo)
set SUPABASE_URL=https://cwvapgovcsqspaukible.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

streamlit run app.py
```

Abra `http://localhost:8501`. As abas “Triagem”, “Perfil & triagem” e “Relatorio” compartilham o mesmo fluxo da nuvem (Streamlit Cloud).

---

## Fluxo do Hackathon

### Dados coletados

- Pressao arterial, frequencia cardiaca, temperatura, saturacao.
- Idade, sexo, tipo sanguineo, contato do paciente.
- Comorbidades, alergias, medicacoes continuas, sintomas relatados.
- Observacoes livres, uploads de exames e radiografias.

### Processo em camadas

| Etapa | Descricao |
| --- | --- |
| **Coleta** (`render_patient_panel` + `hackathon.NursingTriageInput`) | Formulario estruturado otimizado para triagens rapidas de enfermagem. |
| **Processamento** (`generate_triage_report`) | Motor de regras auditavel: gera pontuacao de risco, alertas e encaminhamentos. |
| **Relatorio** (`render_report_viewer` + `build_final_report_text`) | Sintese textual e JSON com justificativa, alertas e downloads. |

### Scripts e exemplos

- `examples/hackathon_triage_input.json`
- `examples/hackathon_triage_output.json`
- `scripts/run_triage_example.py`

```bash
python scripts/run_triage_example.py \
  --input examples/hackathon_triage_input.json \
  --output examples/hackathon_triage_output.json
```

### Uso como modulo

```python
from hackathon import NursingTriageInput, generate_triage_report

payload = NursingTriageInput(
    systolic=188,
    diastolic=118,
    heart_rate=126,
    spo2=91,
    symptoms=["dor no peito", "falta de ar"],
)
report = generate_triage_report(payload)
print(report.summary_markdown())
```

---

## Integração com Supabase

- Configure `SUPABASE_URL` e `SUPABASE_SERVICE_ROLE_KEY` (em `st.secrets` ou variaveis de ambiente).
- Ao salvar uma triagem, o app upserta o paciente (`pacientes`), registra a triagem (`triagens`) e salva o resultado (`relatorios_triagem`).
- Script SQL utilizado para criar as tabelas (fornecido pelo cliente) permanece como referencia oficial.

---

## Estrutura

```
app.py                   # Interface Streamlit
hackathon.py             # Motor de triagem e classes de dados
examples/                # JSON de entrada/saida do Hackathon
scripts/run_triage_example.py
med_modules/             # Analises extras (labs, DICOM, etc.)
requirements.txt
README.md
```

---

## Roadmap sugerido

1. Persistir historico completo de triagens e anexos no Supabase (views, dashboards).
2. Receber sinais vitais em tempo real (wearables) via WebSocket.
3. Publicar um back-end FastAPI opcional para outras interfaces (mobile, React, etc.).

---

## Autores

- Ryan Tereciani  
- Reuel Amador Mantovani  
- Paula Morgatto  
- Raul Pavan  
- Felipe Bento e Souza

> Projeto academico. Nao substitui avaliacao medica presencial.
