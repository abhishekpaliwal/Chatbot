import os
import streamlit as st
from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_chain():
    prompt = ChatPromptTemplate([
        ("system", "You are an AI chatbot. Answer the user based on the query."),
        ("user", "Question:{question}")
    ])
    llm = Ollama(model="gemma3:12b", base_url=base_url)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

chain = get_chain()
##streamlit framework
st.title("ChatBot with Ollama")

# Create a container for the chat history
chat_container = st.container()

# Create a container at the bottom for input
input_container = st.empty()

# Inject custom CSS to fix input box at bottom
st.markdown("""
    <style>
    .bottom-input {
        position: fixed;
        bottom: 20px;
        width: 100%;
        left: 0;
        padding: 10px;
        background-color: #f9f9f9;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)


# Input box styled at bottom
with st.container():
    st.markdown('<div class="bottom-input">', unsafe_allow_html=True)
    input_text = st.text_input("💬 Type your message here...", key="user_input")
    st.markdown('</div>', unsafe_allow_html=True)

if input_text:
    with chat_container:
        with st.spinner("Thinking..."):
            response = chain.stream({"question": input_text})
            st.write_stream(response)
