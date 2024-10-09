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
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)
import chromadb
# %reload_ext autoreload
# %autoreload 2
from util import chroma
from util import embedding


# %%
# user params
pdf_docs = 'assets/library'
chroma_collection_name = 'disaster_response_collection_new'
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
# Load PDFs from directory as type langchain documents
docs = embedding.load_pdf_docs(pdf_docs)

# %%
# split and chunk the documents to prep for embedding
chunked = embedding.rec_split_chunk(docs, chunk_size = 512, chunk_overlap = 30)


# %%
# create persistent vector db of embeddings in chroma
test = chroma.upload_to_collection(collection_name=chroma_collection_name, 
chunked_documents= chunked, 
embedding_model= embedding_model,
persist_path = "db")

# %%
client = chromadb.PersistentClient(path='db') 

# %%
collection = client.get_collection(name=chroma_collection_name)#, embedding_function=embedding_model)

# %%
collection.count()

# %%
results = collection.query(
    query_texts=["What roles do schools play in disaster repsonse?"],
    n_results=3
)

results

# %%
query_str = "Explain how the schools operate in disaster situations"

# %%
# provide additional context
new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query while providing citations to the documents used.\n"
    "Query: {query_str}\n"
    "Answer: "
)


# %%
def query_rag(query_str, embedding_model):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_str (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  # Prepare the database
  db = Chroma(embedding_model=embedding_model)
  
  # Retrieving the context from the DB using similarity search
  results = db.similarity_search_with_relevance_scores(query_str, k=3)

  # Check if there are any matching results or if the relevance score is too low
  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
  # Create prompt template using context and query text
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_str)
  
  # Initialize OpenAI chat model
  model = ChatOpenAI()

  # Generate response text based on the prompt
  response_text = model.predict(prompt)
 
   # Get sources of the matching documents
  sources = [doc.metadata.get("source", None) for doc, _score in results]
 
  # Format and return response including generated text and sources
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text


# %%
# Let's call our function we have defined
formatted_response, response_text = query_rag(query_str)
# and finally, inspect our final response!
print(response_text)


# %%

# %%

# %%

# %%

# %%

# %%

# %%
def query_database(query_str, n_results=10):
    results = collection.query(query_strs=query_str, n_results=n_results)
    return results

# %%
# The init_cache() function  initializes the semantic cache.

# It employs the FlatLS index, which might not be the fastest but is ideal for small datasets.
#  Depending on the characteristics of the data intended 
# for the cache and the expected dataset size, another index such as HNSW or IVF could be utilized.

# %%
cache = embedding.semantic_cache("4cache.json")

# %%



