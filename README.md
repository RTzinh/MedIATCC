# MedIA

MedIA is a multimodal clinical assistant built as a capstone project (Trabalho de Conclusão de Curso) by **Ryan Tereciani** and **Reuel Amador Mantovani**.
The project combines exam analysis, AI conversation and a dedicated workflow for the **Smart Health Hackathon – AI Medical Triage**, aimed at nursing teams that need fast, auditable and actionable reports.

---

## Overview

- Conversational triage powered by Groq (Llama 3.x models) with persistent clinical context.
- Upload and interpretation of laboratory exams and DICOM studies via `med_modules/`.
- Voice panel powered by Gemini AI Studio for hands-free conversations.
- Consolidated "Profile & triage" tab: captures patient data + vital signs and immediately runs the Hackathon engine.
- "Report" tab that automatically produces the final text (with `.txt` download).
- Optional Supabase support to register patients, triages and auditable reports.

---

## Tech Stack

- Python 3.12
- Streamlit >= 1.25
- Groq SDK (`groq`) and Google Generative AI (`google-generativeai` / `google-genai`)
- PyPDF2, Pillow, pytesseract, pydicom, pyedflib, gTTS
- Supabase Python client (optional) for persistence

---

## How to Run

```bash
git clone https://github.com/YOUR-USERNAME/MedIATCC.git
cd MedIATCC
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

set GROQ_API_KEY=your_token
set GEMINI_API_KEY=your_token    # optional (a test key already ships in the code)
set SUPABASE_URL=https://cwvapgovcsqspaukible.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

streamlit run app.py
```

Open `http://localhost:8501`. The "Triage", "Profile & triage" and "Report" tabs share the same cloud flow (Streamlit Cloud).

---

## Hackathon Workflow

### Collected data

- Blood pressure, heart rate, temperature, oxygen saturation.
- Age, sex, blood type, patient contact.
- Comorbidities, allergies, continuous medications, reported symptoms.
- Free-text notes, exam uploads and radiographs.

### Layered process

| Stage | Description |
| --- | --- |
| **Collection** (`render_patient_panel` + `hackathon.NursingTriageInput`) | Structured form optimized for fast nursing triage. |
| **Processing** (`generate_triage_report`) | Auditable rule engine: produces a risk score, alerts and referrals. |
| **Report** (`render_report_viewer` + `build_final_report_text`) | Textual and JSON synthesis with rationale, alerts and downloads. |

### Scripts and examples

- `examples/hackathon_triage_input.json`
- `examples/hackathon_triage_output.json`
- `scripts/run_triage_example.py`

```bash
python scripts/run_triage_example.py \
  --input examples/hackathon_triage_input.json \
  --output examples/hackathon_triage_output.json
```

### Use as a module

```python
from hackathon import NursingTriageInput, generate_triage_report

payload = NursingTriageInput(
    systolic=188,
    diastolic=118,
    heart_rate=126,
    spo2=91,
    symptoms=["chest pain", "shortness of breath"],
)
report = generate_triage_report(payload)
print(report.summary_markdown())
```

---

## Supabase Integration

- Configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (in `st.secrets` or environment variables).
- When saving a triage, the app upserts the patient (`pacientes`), records the triage (`triagens`) and stores the result (`relatorios_triagem`).
- The SQL script used to create the tables (provided by the client) remains the official reference.

> Note: the Supabase table and column names above are kept in Portuguese because they map to the existing database schema.

---

## Structure

```
app.py                   # Streamlit interface
hackathon.py             # Triage engine and data classes
examples/                # Hackathon input/output JSON
scripts/run_triage_example.py
med_modules/             # Extra analyses (labs, DICOM, etc.)
requirements.txt
README.md
```

---

## Suggested Roadmap

1. Persist the full history of triages and attachments in Supabase (views, dashboards).
2. Receive real-time vital signs (wearables) via WebSocket.
3. Publish an optional FastAPI back-end for other interfaces (mobile, React, etc.).

---

## Authors

- Ryan Tereciani
- Reuel Amador Mantovani
- Paula Morgatto
- Raul Pavan
- Felipe Bento e Souza

> Academic project. It does not replace an in-person medical evaluation.
