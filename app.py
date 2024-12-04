import os  # Biblioteca para manipulação do sistema operacional (não usada neste código).
import streamlit as st  # Biblioteca para criar interfaces de aplicativos web.

st.set_page_config(
    page_title="MedIA",  # Nome exibido na aba do navegador.
    page_icon="aaaaaaaaaaaaaaa.png",    # Ícone exibido ao lado do título da página.
    layout="centered",       # Layout da página (pode ser 'centered' ou 'wide').
)

# Importações do LangChain, usadas para criar cadeias de processamento de linguagem natural.
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,  # Modelo de prompt para interações baseadas em mensagens de chat.
    HumanMessagePromptTemplate,  # Modelo para representar mensagens de entrada do usuário.
    MessagesPlaceholder,  # Placeholder para mensagens do histórico.
)
from langchain.schema import SystemMessage  # Representa mensagens de sistema em conversas.
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  # Gerencia o histórico de conversas.
from langchain_groq import ChatGroq  # Integração com o modelo LLM Groq.



# Função principal do aplicativo.
def main():
    """
    Ponto de entrada principal do aplicativo.

    Configura a interface, inicializa o modelo LLM Groq, gerencia o histórico de conversas
    e processa a interação do usuário com o chatbot MedIA.
    """

    # Obtém a chave de API do modelo Groq a partir dos segredos configurados no Streamlit.
    groq_api_key = st.secrets["GROQ_API_KEY"]
    # Define o modelo a ser usado.
    model = 'llama3-8b-8192'

    # Inicializa o cliente Groq com a chave de API e o modelo especificado.
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Configura o título e a descrição do aplicativo.
    st.title("MedIA")
    st.write(
        """
        Seja bem-vindo ao MedIA! \n\n
        Sou um sistema de inteligência artificial treinado para auxiliar na análise de sintomas 
        e direcionar você para o caminho certo. Baseado em suas respostas, tentarei traçar um panorama 
        do que pode estar acontecendo.
        """
    )

    # Mensagem de sistema usada para configurar o comportamento do modelo.
    system_prompt = (
        "Você é um especialista em diagnósticos médicos. Baseado nos sintomas apresentados pelo usuário, "
        "personalize um possível diagnóstico. Sugira ao paciente que ele responda todas as perguntas sem exceção. "
        "Não dê a resposta enquanto ele não responder todas as perguntas. Após ele responder as 10 perguntas, "
        "você pode dar o diagnóstico e recomendar possíveis exames que um médico pediria. "
        "Se ele perguntar os sintomas de alguma doença, dê a resposta imediatamente nesse caso, sem fazer perguntas. "
        "Receite alguns remédios básicos que não precisam ser orientados por um profissional e recomende alguns exames. "
        "Coloque todas as doenças relacionadas possíveis. Faça sempre 10 perguntas muito úteis, nem menos nem mais que isso. "
        "Faça 1 pergunta de cada vez. Quando estiver acabando as perguntas, avise o paciente e numere-as. "
        "Se o usuário disser algo que seja de atendimento imediato como (infarto, tiro, golpe de faca e etc), nesse caso não faça perguntas e oriente-o a ligar para a emergencia seja 190 (policia) o 192 (Samu, Ambulância) e pedir ajuda imediata."
    )

    # Comprimento máximo do histórico de conversas.
    conversational_memory_length = 50000

    # Inicializa a memória da conversa no estado da sessão, se ainda não estiver configurada.
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, memory_key="chat_history", return_messages=True
        )

    # Inicializa o histórico de mensagens do usuário e do chatbot, se não existir.
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Adiciona estilos customizados para a interface do chat.
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

    # Exibe o histórico da conversa na interface.
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

    # Campo de entrada para o usuário digitar os sintomas.
    user_input = st.chat_input(
        "Digite seus sintomas",
        key='user_input'
    )

    if user_input:  # Se o usuário forneceu uma entrada:
        # Adiciona a entrada do usuário ao histórico.
        st.session_state.history.append(f"<div class='message user-message'><strong>Você:</strong> {user_input}</div>")

        # Cria o prompt do chatbot usando os templates de mensagens.
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # Mensagem de sistema inicial.
                MessagesPlaceholder(variable_name="chat_history"),  # Histórico de conversas.
                HumanMessagePromptTemplate.from_template("{human_input}"),  # Entrada do usuário.
            ]
        )

        # Configura a cadeia de interação com o modelo LLM.
        conversation = LLMChain(
            llm=groq_chat,  # Cliente Groq para interações.
            prompt=prompt,  # Prompt configurado.
            verbose=False,  # Sem logs detalhados.
            memory=st.session_state.memory,  # Memória da conversa.
        )

        # Obtém a resposta do modelo para a entrada do usuário.
        response = conversation.predict(human_input=user_input)

        # Adiciona a resposta do modelo ao histórico.
        st.session_state.history.append(f"<div class='message ai-message'><strong>MedIA:</strong> {response}</div>")
        # Atualiza a interface para refletir o novo estado.
        st.rerun()

    # Script para rolar automaticamente o histórico de mensagens ao final.
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

# Ponto de entrada do aplicativo.
if __name__ == "__main__":
    main()
