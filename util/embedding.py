# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


# def create_embedding(model_name = "sentence-transformers/all-mpnet-base-v2",
# model_kwargs = {'device': 'cpu'},
# encode_kwargs = {'normalize_embeddings': False}):
#     hf = HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs
#     )

# load pdfs from directory as langchain documents
def load_pdf_docs(pdf_directory_path):
  """
  Load PDF documents from the specified directory
  Returns:
  Langchain Document objects.
  """
  # Initialize PDF loader with specified directory
  loader = PyPDFDirectoryLoader(pdf_directory_path) 
  # Load PDF files as langchain docs
  loaded_docs = loader.load()
  return loaded_docs

# recursively split and chunk langchain documents
def rec_split_chunk(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function=len, add_start_index=True)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs
