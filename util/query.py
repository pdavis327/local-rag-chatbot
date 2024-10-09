import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

llama = Ollama(model="llama2",
request_timeout=40.0)

def create_prompt():
    '''
    Create prompt template
    '''
    modified_query = PromptTemplate(
        input_variables=["question"],
        template="""Given the context information with no prior knowledge, 
        answer the question while providing citations to the context documents used.
        If you can't answer the question, don't try to. Provide each citation on a new line
        Original question: {question}""",
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return modified_query, prompt

# Main function to handle the query process
def query(input):
    if input:
        # Initialize the language model with the specified model name
        llm = ChatOllama(model=LLM_MODEL)
        # Get the vector database instance
        db = chroma.get_vector_db(collection_name, embedding_model, persist_path)
        # Get the prompt templates
        modified_query, prompt = get_prompt()

        # Set up the retriever to generate multiple queries using the language model and the query prompt
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(), 
            llm,
            prompt=modified_query
        )

        # Define the processing chain to retrieve context, generate the answer, and parse the output
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(input)

        return response

    return None