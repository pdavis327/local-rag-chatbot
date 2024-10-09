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
from getpass import getpass
from datasets import load_dataset
import chromadb
import faiss
from sentence_transformers import SentenceTransformer
import time
import json
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from chromadb.utils import embedding_functions

# %reload_ext autoreload
# %autoreload 2
from util import chroma


# %%
# user params
pdf_docs = 'assets/library'
chroma_collection_name= 'disaster_response_collection'
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
chroma.load_chunk_embed_pdf(pdf_docs, chroma_collection_name, embedding_model, chunk_size=512, chunk_overlap=30)


# %%
documents = []
pdf_folder_path = "assets/library"
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)


# %%
chunked_docs = text_splitter.split_documents(documents)

# %%
client = chromadb.Client()

# %%

try:
    collection = client.get_collection(name='disaster_reponse_test')
except:
    print('collection doesnt exist, creating collection')
    collection = client.create_collection('disaster_reponse_test')

# %%
default_ef = embedding_functions.DefaultEmbeddingFunction()

# %%
vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
    )


# %%
vectordb.persist()

# %%

# %%

# %%

# %%
vectordb = Chroma.from_documents(
    documents=chunked_docs,
    embedding=hf,
)
# vectordb.persist()

# %%
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


# %%


# %%
def query_database(query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

# %%
# The init_cache() function  initializes the semantic cache.

# It employs the FlatLS index, which might not be the fastest but is ideal for small datasets.
#  Depending on the characteristics of the data intended 
# for the cache and the expected dataset size, another index such as HNSW or IVF could be utilized.

# %%
cache = embedding.semantic_cache("4cache.json")

# %%



