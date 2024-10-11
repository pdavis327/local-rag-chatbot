import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from util import query

# RAG setup

# user params
chroma_persist_path = 'db'
chroma_collection_name = 'disaster_response_collection_new'
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama2")

# Load data from vector db
db = Chroma(
    collection_name=chroma_collection_name,
    embedding_function=embedding_model,
    persist_directory = 'db'
)

# Setting the title of the Streamlit application
st.title('Ollama Langchain RAG using chroma')

# Creating a sidebar input widget for the OpenAI API key, input type is password for security
# openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

msgs = StreamlitChatMessageHistory(key="special_app_key")
history = StreamlitChatMessageHistory(key="chat_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", """Answer the question based ONLY on the following context:
            {context}
            - -
            Given the context information with no prior knowledge, 
            answer the question. If you can't answer the question
            with the context information, don't try. 
            In a new line, providing citations to each context
            document used, and include all metadata, source tile, title, date, and year.
            Provide each citation on a new line
            Original question: {question}"""),
    ]
)


chain_with_history = RunnableWithMessageHistory(
    query.query_rag(db, llm, prompt),
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
    st.chat_message("ai").write(response.content)

# # Creating a form in the Streamlit app for user input
# with st.form('my_form'):
#     # Adding a text area for user input with a default prompt
#     text = st.text_area('Enter text:', '')
#     # Adding a submit button for the form
#     submitted = st.form_submit_button('Submit')

#     # If the form is submitted and the API key is valid, generate a response
#     if submitted:
#         st.info(query.query_rag(db, text, llm, query.template))