import argparse
import warnings
warnings.filterwarnings('ignore') 
from util.query import query_rag, init_llm, prompt_template
import os
from langchain_community.vectorstores import Chroma
from util import embedding
# import langchain
# langchain.debug = True


parser = argparse.ArgumentParser(
                prog='Simple langchian RAG',
                description='This RAG LLM uses Ollama, Langchain, and Chromadb')

parser.add_argument("queryString", type=str, help='Input string to query RAG grounding docs and LLM')
args = parser.parse_args()

collection_name = os.getenv('CHROMA_COLLECTION_NAME')
embedding_model = embedding.init_embedding_model()
persist_directory = os.getenv('CHROMA_PERSIST_PATH')
llm = init_llm()

# Load data from vector db
db = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory = persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

print(f'\n{args.queryString}\n')

query_rag(db, args.queryString, llm, prompt_template)