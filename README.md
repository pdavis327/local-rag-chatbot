# Simple local RAG chatbot app using Ollama, langchain, and chroma

This is as simple open source local rag app. It uses Chroma as the vector db, and ollama for the llm. Within the repo there is a chroma vector database with some example documents already embedded and loaded. It also includes the ability to embed and store new documents in a new chroma database collection


## Getting Started

### Dependencies

Requirements differ depending on whether you plan to run the code in Docker or not. 

1. To create a new conda environment:
```
conda create --name my_project_env
conda activate my_project_env
pip install -r requirements.txt
```

2. Pull the desired model, and run ollama
```
ollama pull llama2
ollama serve
```

If you plan to run the code in Docker, pulling and serving the model will be taken care of in the build process. 

### Installing

1. Clone the repo and navigate to the directory
2. Rename .env.example to .env
3. Specify the environmeental paramaters. 


## Executing program

### Creating a chroma db and embedding documents

`util/chroma.py` takes one argument `directory` which is the filepath to the documents you wish to embed and store. 

You can run the code using the following: 

```
python chroma.py rag_exploration/assets/library
```

The results will be stored using your .env variables in a new chroma db `CHROMA_COLLECTION_NAME` in `CHROMA_PERSIST_PATH`

### Running the chatbot

`test.py` can be run from the cmd line for testing. It doesn't retain any history to the chat. In order to run it, just supply a query. 

```
python test.py 'what role do schools play in disaster response?'
```

To run the app
```
streamlit run app.py
```

To run the app in docker:
```
docker-compose up
```