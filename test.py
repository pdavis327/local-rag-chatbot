from langchain_community.vectorstores import Chroma
import os 
from dotenv import load_dotenv
load_dotenv()
import argparse
import warnings
warnings.filterwarnings('ignore') 

# %reload_ext autoreload
# %autoreload 2
from util import chroma
from util import embedding
from util import query

parser = argparse.ArgumentParser(
                    prog='Simple langchian RAG',
                    description='This RAG LLM uses Ollama, Langchain, and Chromadb')
parser.add_argument("queryString", type=str, help='Input string to query RAG grounding docs and LLM')
args = parser.parse_args()

collection_name = os.getenv('CHROMA_COLLECTION_NAME')
embedding_model = embedding.init_embedding_model()
persist_directory = os.getenv('CHROMA_PERSIST_PATH')
llm = query.init_llm()

# Load data from vector db
db = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory = persist_directory
)

print(f'\n{args.queryString}\n')

query.query_rag(db, args.queryString, llm, query.template)
