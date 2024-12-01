import os
import streamlit as st

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq


def main():
    """
    Ponto de entrada principal do aplicativo.
    """

    # Inicializar chave da API e modelo
    groq_api_key = st.secrets["GROQ_API_KEY"]
    model = 'llama3-8b-8192'

    # Inicializar cliente Groq
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Configurar interface do Streamlit
    st.title("MedIA")
    st.write(
        """
        Seja bem-vindo ao MedIA! \n\n
        Sou um sistema de inteligência artificial treinado para auxiliar na análise de sintomas e direcionar você para o caminho certo. 
        Baseado em suas respostas, tentarei traçar um panorama do que pode estar acontecendo.
        """
    )

    # Inicializar prompt
    system_prompt = (
        "Você é um especialista em diagnósticos médicos. Baseado nos sintomas apresentados pelo usuário, "
        "personalize um possível diagnóstico. Sugira ao paciente que ele responda todas as perguntas sem exceção. "
        "Não dê a resposta enquanto ele não responder todas as perguntas. Após ele responder as 10 perguntas, "
        "você pode dar o diagnóstico e recomendar possíveis exames que um médico pediria. "
        "Se ele perguntar os sintomas de alguma doença, dê a ele a resposta imediata nesse caso, não faça perguntas. "
        "Receite alguns remédios básicos que não precisam ser orientados por um profissional e também recomende alguns exames. "
        "Coloque todas as doenças relacionadas possíveis. Faça sempre 10 perguntas muito úteis, nem menos nem mais que isso. "
        "Faça 1 pergunta de cada vez. Quando estiver acabando as perguntas, avise o paciente. Só não faça pergunta se ele fizer uma pergunta sobre os sintomas de alguma doença, nesse caso, dê a ele uma resposta imediata."
        "Se o usuario dizer que levou tiro ou golpe de faca oriente-o a ligar ao 190 e pedir ajuda imediata"
    )

    conversational_memory_length = 50000
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

    if 'history' not in st.session_state:
        st.session_state.history = []

    st.markdown(
        """
        <style>
            .chat-input-container {
                position: fixed;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 60%;
                z-index: 1000;
                background-color: #f0f2f6;
                padding: 10px;
                border-top: 1px solid #ddd;
            }
            #chat-history {
                height: calc(100vh - 150px);
                overflow-y: auto;
                padding-bottom: 60px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Respostas do MedIA")
    if st.session_state.history:
        st.markdown(
            f"""
            <div id="chat-history" style="display: flex; flex-direction: column;">
                {"<hr>".join(st.session_state.history)}
            </div>
            """, 
            unsafe_allow_html=True
        )

    user_input = st.chat_input(
        "Digite seus sintomas",
        key='user_input'
    )

    if user_input:
        st.session_state.history.append(f"<div class='message user-message'><strong>Você:</strong> {user_input}</div>")

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

        response = conversation.predict(human_input=user_input)

        st.session_state.history.append(f"<div class='message ai-message'><strong>MedIA:</strong> {response}</div>")
        st.rerun()

    st.markdown(
        """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const chatHistory = document.getElementById('chat-history');
                const observer = new MutationObserver(() => {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                });
                if (chatHistory) {
                    observer.observe(chatHistory, { childList: true });
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            });
        </script>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
