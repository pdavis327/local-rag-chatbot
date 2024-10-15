# Simple langchain rag app using Ollama and Chroma

This is as simple open source rag app. It uses Chroma as the vector db, ollama for the llm. Within the repo there is a chroma vector database with some example documents already embedded and loaded. 

## Description

I built this to become more familiar with LLMs and RAG pipelines. I followed a number of tutorials I found online, and relied heavily on the langchain, chroma, and Ollama documentation. 

## Getting Started


### Dependencies

[Ollama](https://ollama.com/) is required to run this code. 

Required packages are in requirements.txt

To create a new conda environment:
```
conda create --name my_project_env
conda activate my_project_env
pip install -r requirements.txt
```

Pull the desired model, and run ollama
```
ollama pull llama2
ollama serve
```

### Installing

1. Clone the repo and navigate to the directory
2. Rename .env.example to .env
3. Assing the variable names. For instance:

```
CHROMA_PERSIST_PATH = 'db'
CHROMA_COLLECTION_NAME = 'disaster_response_collection_new'
LLM = 'llama2'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
```

### Executing program

test.py can be run from the cmd line for testing. It doesn't retain any history to the chat. In order to run it, just supply a query. 

```
python test.py 'What role do schools play in natural disaster response?'
```

To run the app
```
streamlit run app.py
```

to run in docker:

```
docker run -p 8501:8501 streamlit

You can now view your Streamlit app in your browser.

URL: http://0.0.0.0:8501
```