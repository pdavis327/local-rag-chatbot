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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from langchain_chroma import Chroma
import chromadb
# %reload_ext autoreload
# %autoreload 2
from util import chroma
from util import embedding
from util import query


# %%
# user params
chroma_persist_path = 'db'
chroma_collection_name = 'disaster_response_collection_new'
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama2")

# %%
# Load data from vector db
db = Chroma(
    collection_name=chroma_collection_name,
    embedding_function=embedding_model,
    persist_directory = 'db'
)

# %%
query_str = 'What roles do schools play in emergencies?'

# %%
query.query_rag(db, query_str, llm, query.template)
