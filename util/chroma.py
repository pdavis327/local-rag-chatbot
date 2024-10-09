import chromadb
from langchain.vectorstores import Chroma
import os

def chroma_unique_id(data):
    data = data.to_pandas()
    data['unique_id'] = data.index
    print('indexing complete')


def query_database(collection, query_text, n_results=5):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

def upload_to_collection(collection_name, chunked_documents, embedding_model):
    client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
    except:
        print(f'collection doesnt exist, creating collection: {collection_name}')
        collection = client.create_collection(collection_name)

    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_model


    )
    return vectordb
