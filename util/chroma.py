import chromadb
from langchain.vectorstores.chroma import Chroma
import os

def chroma_unique_id(data):
    data = data.to_pandas()
    data['unique_id'] = data.index
    print('indexing complete')

def query_database(collection, query_text, n_results=5):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

def upload_to_collection(collection_name, chunked_documents, embedding_model, persist_path = None):
    # client = chromadb.Client()
    # try:
    #     collection = client.get_collection(name=collection_name)
    #     print(f'collection already exists: {collection_name}')
    # except:
    #     print(f'collection doesnt exist, creating collection: {collection_name}')
    #     collection = client.create_collection(collection_name)
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

    return db

