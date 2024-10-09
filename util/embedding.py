# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def create_embedding(model_name = "sentence-transformers/all-mpnet-base-v2",
model_kwargs = {'device': 'cpu'},
encode_kwargs = {'normalize_embeddings': False}):
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# load pdfs from directory as langchain documents
def load_pdf(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

# recursively split and chunk langchain documents
def rec_split_chunk(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs
