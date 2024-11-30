# MedIATCC

**MedIATCC** é um projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) por **Ryan Tereciani** e **Reuel Amador Mantovani**.  
Trata-se de um sistema de inteligência artificial voltado para auxiliar usuários na análise de sintomas médicos e oferecer direcionamentos iniciais para cuidados e exames.

---

## 🩺 **Sobre o Projeto**

O **MedIATCC** utiliza um modelo de inteligência artificial baseado na integração com a API Groq e a biblioteca LangChain para oferecer um sistema interativo e inteligente que:

- Faz perguntas relevantes para entender os sintomas do usuário.
- Oferece sugestões de possíveis diagnósticos preliminares.
- Recomenda exames médicos e medicamentos básicos de venda livre.
- Responde perguntas específicas sobre sintomas de doenças.
- Direciona emergências, como ferimentos graves, para serviços de emergência (190).

O objetivo principal é fornecer um assistente confiável que auxilie os usuários no entendimento inicial de suas condições de saúde.

---

## 🚀 **Tecnologias Utilizadas**

O projeto foi desenvolvido utilizando as seguintes tecnologias e bibliotecas:

- **Python 3.12**
- **Streamlit**: Para criar a interface interativa do usuário.
- **LangChain**: Para gerenciar as interações de linguagem natural.
- **LangChain-Groq**: Para integração com o modelo Groq.
- **Groq API**: Modelo Llama3 usado como base de IA.

---

## 📂 **Estrutura do Projeto**

MedIATCC/
├── .streamlit/
│   └── secrets.toml        # Arquivo de configuração da API
├── app.py                  # Código principal da aplicação
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação do projeto




---

## 🛠️ **Como Configurar e Executar**

### Pré-requisitos

- **Python 3.12** instalado.
- Pip atualizado:
  ```
  pip install --upgrade pip
Chave de acesso à Groq API.
Passos para Executar
Clone este repositório:

```

git clone https://github.com/SEU-USUARIO/MedIATCC.git
cd MedIATCC
```
-**Instale as dependências**:

```
pip install -r requirements.txt
```
Configure a chave da API no arquivo .streamlit/secrets.toml:

```
mkdir -p .streamlit
echo "GROQ_API_KEY = 'SUA_CHAVE_AQUI'" > .streamlit/secrets.toml
```
Execute o aplicativo:

```
streamlit run app.py
Acesse a aplicação no navegador pelo link fornecido (geralmente http://localhost:8501).
```

##💻** Autores**
Este projeto foi desenvolvido por:

Ryan Tereciani
Reuel Amador Mantovani

##📜 **Licença**
Este projeto é apenas para fins acadêmicos e não deve ser utilizado como substituto de consultas médicas.

