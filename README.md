# MedIA

MedIA e um assistente clinico multimodal construido como Trabalho de Conclusao de Curso por **Ryan Tereciani** e **Reuel Amador Mantovani**.  
O projeto integra analise de exames, conversacao com IA e um novo fluxo dedicado ao **Hackathon Saude Inteligente  Triagem Medica com IA**, direcionado a equipes de enfermagem que precisam gerar relatorios rapidos, auditaveis e acionaveis.

---

## Visao Geral

- Triagem conversacional com modelos Groq (Llama 3.x) e persistencia de contexto clinico.
- Upload e interpretacao de exames laboratoriais e DICOM via modulos em `med_modules/`.
- Painel de voz com Gemini AI Studio para conversas hands-free.
- Nova aba **Hackathon** com pipeline em camadas (coleta  processamento  relatorio  interface) e exportacao JSON.

---

## Tecnologias

- Python 3.12
- Streamlit 1.25+
- Groq SDK (`groq`) para o chat principal
- Google Generative AI (`google-generativeai`) para voz e triagens rapidas
- PyPDF2, Pillow, pytesseract, pydicom, pyedflib, gTTS e utilitarios clinicos personalizados

---

## Configuracao

```bash
git clone https://github.com/SEU-USUARIO/MedIATCC.git
cd MedIATCC
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

set GROQ_API_KEY=seu_token
set GEMINI_API_KEY=seu_token    # opcional: o app ja inclui uma chave padrao para testes

streamlit run app.py
```

Abra `http://localhost:8501` e escolha o fluxo desejado (chat, voz ou aba Hackathon).

---

## Hackathon Saude Inteligente  Triagem Medica com IA

### Problema

Unidades de saude com alta demanda precisam que enfermeiros registrem rapidamente sinais vitais e sintomas para priorizar quem corre mais risco. O desafio pede:

- Coleta de pressao arterial, frequencia cardiaca, temperatura, saturacao, idade/sexo.
- Registro de doencas cronicas, alergias, tipo sanguineo, medicamentos continuos.
- Classificacao automatica dos sintomas relatados e sugestao de encaminhamentos.
- Relatorios claros, interpretaveis e exportaveis.

### Camadas implementadas

| Etapa | Descricao |
| --- | --- |
| **Coleta** (`hackathon.NursingTriageInput`) | Normaliza dados vindos da enfermagem, incluindo listas livres de sintomas/alergias. |
| **Processamento** (`hackathon.generate_triage_report`) | Motor de regras auditavel; atribui pontuacao de risco, dispara alertas por metrica e sugere exames/acoes. |
| **Relatorio** (`hackathon.TriagePipelineOutput`) | Estrutura prioridade, justificativa, alertas e encaminhamentos com exportacao JSON. |
| **Interface** (`render_hackathon_triage_tab`) | Formulario otimizado para ambientes simples, botoes de exemplo e download imediato. |

### Como usar na interface

1. Entre na aba **Hackathon** no Streamlit.
2. Preencha os campos (ou clique em **Carregar exemplo oficial do Hackathon** para usar o dataset padrao).
3. Clique em **Gerar relatorio automatizado** para ver:
   - Resumo em linguagem natural.
   - Barra de risco com pontuacao (baixo  critico).
   - JSON pronto para prontuario eletronico ou API.
   - Cada camada (coleta/processamento/relatorio) em expanders separados.

### Scripts e exemplos

- `examples/hackathon_triage_input.json`: dados coletados pela enfermagem.
- `examples/hackathon_triage_output.json`: relatorio gerado pelo motor.
- `scripts/run_triage_example.py`: script CLI para testar o pipeline sem UI.

```bash
python scripts/run_triage_example.py \
  --input examples/hackathon_triage_input.json \
  --output examples/hackathon_triage_output.json
```

Resultado: resumo textual no terminal e JSON salvo em `examples/hackathon_triage_output.json`.

### Uso como modulo/API

```python
from hackathon import NursingTriageInput, generate_triage_report

payload = NursingTriageInput(
    systolic=150,
    diastolic=95,
    heart_rate=110,
    spo2=93,
    symptoms=["falta de ar", "dor no peito"],
)
report = generate_triage_report(payload)
print(report.summary_markdown())
print(report.to_dict())        # pronto para expor via REST/FastAPI
```

A trilha de auditoria (`audit_trail`) garante interpretabilidade e atende ao criterio de modelo auditavel citado no desafio.

---

## Estrutura do Repositorio

```
MedIATCC/
 app.py                       # Streamlit app (chat, voz, Hackathon)
 hackathon.py                 # Logica em camadas da triagem
 med_modules/                 # Interpretadores externos (labs, DICOM, etc.)
 examples/
    hackathon_triage_input.json
    hackathon_triage_output.json
 scripts/
    run_triage_example.py
 requirements.txt
 README.md
```

---

## Roadmap sugerido

1. Publicar endpoints REST usando FastAPI que reutilizem `hackathon.py`.
2. Receber sinais vitais em tempo real (wearables) na aba Hackathon via WebSocket.
3. Versionar regras clinicas para permitir ajustes por comites hospitalares.

---

## Autores

- Ryan Tereciani  
- Reuel Amador Mantovani

> Projeto academico. Nao substitui avaliacao medica presencial.
