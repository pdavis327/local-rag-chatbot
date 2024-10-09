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
#     display_name: semantic_cache
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


# %%
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")


# %%
data = data.to_pandas()
data["id"] = data.index
data.head(10)

# %%
MAX_ROWS = 15000
DOCUMENT = "Answer"
TOPIC = "qtype"
subset_data = data.head(MAX_ROWS)

# %%
chroma_client = chromadb.PersistentClient(path="assets/chromadb")

# %%
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)

# %%
collection.add(
    documents=subset_data[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_data[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_ROWS)],
)


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
def init_cache():
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")

    # Initialize Sentence Transformer model
    encoder = SentenceTransformer("all-mpnet-base-v2")

    return index, encoder


# %%
# In the retrieve_cache function, the .json file is retrieved from disk 
# in case there is a need to reuse the cache across sessions.

def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}

    return cache


# %%
# The store_cache function saves the file containing the cache data to disk.

def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)


# %%
cache = semantic_cache("4cache.json")

# %%
