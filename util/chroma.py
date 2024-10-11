import chromadb
from langchain.vectorstores.chroma import Chroma
import os
from util import embedding

def chroma_unique_id(data):
    data = data.to_pandas()
    data['unique_id'] = data.index
    print('indexing complete')

def query_database(collection, query_text, n_results=5):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

def upload_to_collection(collection_name, chunked_documents, embedding_model, persist_path = None):

    if persist_path:
        vectordb = Chroma.from_documents(
            collection_name = collection_name,
            documents=chunked_documents,
            embedding=embedding_model,
            persist_directory = persist_path
            )
    
    else:
        vectordb = Chroma.from_documents(
            collection_name = collection_name,
            documents=chunked_documents,
            embedding=embedding_model
            )
    return vectordb

def get_vector_db(collection_name, embedding_model, persist_path):
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_path,
        embedding_function=embedding_model
    )


if __name__ == "__main__":
    # create chroma db of pdf documents
    # user params
    pdf_docs = ''
    chroma_persist_path = 'db'
    chroma_collection_name = 'disaster_response_collection_new'
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load pdf as langchain docs
    docs = embedding.load_pdf_docs(pdf_docs)

    # split and chunk the documents to prep for embedding
    chunked = embedding.rec_split_chunk(docs, chunk_size = 512, chunk_overlap = 30)

    # create persistent vector db of embeddings in chroma
    db = upload_to_collection(collection_name=chroma_collection_name, 
    chunked_documents= chunked, 
    embedding_model= embedding_model,
    persist_path = "db")

