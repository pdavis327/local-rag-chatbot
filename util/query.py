from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from operator import itemgetter
import os 
from dotenv import load_dotenv
load_dotenv()


template = """
  Answer the question based ONLY on the following context:
  {context}
  - -
  Given the context information with no prior knowledge, 
  answer the question. If you can't answer the question
  with the context information, don't try. 
  In a new line, providing citations to each context
  document used, and include all metadata, source tile, title, date, and year.
  Provide each citation on a new line
  Original question: {question}
  """

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def init_prompt(promp_template):
  prompt = ChatPromptTemplate.from_template(promp_template)
  return prompt

def init_llm():
  llm = OllamaLLM(model= os.getenv('LLM'))
  return llm

def query_rag(Chroma_collection, query_text, llm_model, promp_template):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma db.
  Args:
    - query_text (str): The text to query the RAG system with.
    -prompt_template (str): Query prompt template inclding context and question
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
  
  db = Chroma_collection
  prompt = ChatPromptTemplate.from_template(promp_template)

  retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 5})

  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
  )

  for chunk in rag_chain.stream(query_text):
      print(chunk, end="", flush=True)
  
  return


def query_rag_streamlit(Chroma_collection, llm_model, promp_template): 
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma db.
  Args:
    - query_text (str): The text to query the RAG system with.
    -prompt_template (str): Query prompt template inclding context and question
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  db = Chroma_collection

  retriever = db.as_retriever(search_type="similarity_score_threshold", 
                              search_kwargs={"score_threshold": 0.5, "k": 5})

  context = itemgetter("question") | retriever | format_docs
  first_step = RunnablePassthrough.assign(context=context)
  chain = first_step | promp_template | llm_model

  return chain
