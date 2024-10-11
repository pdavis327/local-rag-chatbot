# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: hflc
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate


import os
from util import embedding
from util import query

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

# env params
chroma_collection_name = os.getenv('CHROMA_COLLECTION_NAME')
embedding_model = embedding.init_embedding_model()
chroma_persist_path = os.getenv('CHROMA_PERSIST_PATH')
llm = query.init_llm()

# Load data from vector db
db = Chroma(
    collection_name=chroma_collection_name,
    embedding_function=embedding_model,
    persist_directory = 'db'
)

# Setting the title of the Streamlit application
st.title('Ollama Langchain RAG using chroma')

msgs = StreamlitChatMessageHistory(key="special_app_key")
history = StreamlitChatMessageHistory(key="chat_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI chatbot having a conversation with a human.
        Answer the question prioritizing the following context:{context}
        Given the context information with no prior knowledge, 
        answer the question. If you can't answer the question
        with the context information, don't try. 
        In a new line, providing citations to each context
        document used, and include all metadata, source tile, title, date, and year.
        Provide each citation on a new line
        Original question: """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
    )

chain = query.query_rag_streamlit(db, llm, prompt)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response)
