import faiss
from sentence_transformers import SentenceTransformer
import time
import json

def init_cache():
    '''
    Initializes the semantic cache.
    It employs the FlatLS index, which might not be the fastest 
    but is ideal for small datasets.
    Depending on the characteristics of the data intended 
    for the cache and the expected dataset size, another index 
    such as HNSW or IVF could be utilized.
    
    Args:
    Returns:
    '''
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")

    # Initialize Sentence Transformer model
    encoder = SentenceTransformer("all-mpnet-base-v2")

    return index, encoder

def retrieve_cache(json_file):
    '''
    In the retrieve_cache function, the .json file is retrieved from disk 
    in case there is a need to reuse the cache across sessions.
    '''
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}

    return cache


def store_cache(json_file, cache):
    '''
    The store_cache function saves the file containing the cache data to disk.
    '''
    with open(json_file, "w") as file:
        json.dump(cache, file)

class semantic_cache:
    def __init__(self, json_file="cache_file.json", thresold=0.35, max_response=100, eviction_policy=None):
        """Initializes the semantic cache.

        Args:
        json_file (str): The name of the JSON file where the cache is stored.
        thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
        max_response (int): The maximum number of responses the cache can store.
        eviction_policy (str): The policy for evicting items from the cache.
                                This can be any policy, but 'FIFO' (First In First Out) has been implemented for now.
                                If None, no eviction policy will be applied.
        """

        # Initialize Faiss index with Euclidean distance
        self.index, self.encoder = init_cache()

        # Set Euclidean distance threshold
        # a distance of 0 means identicals sentences
        # We only return from cache sentences under this thresold
        self.euclidean_threshold = thresold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):
        """Evicts an item from the cache based on the eviction policy."""
        if self.eviction_policy and len(self.cache["questions"]) > self.max_size:
            for _ in range((len(self.cache["questions"]) - self.max_response)):
                if self.eviction_policy == "FIFO":
                    self.cache["questions"].pop(0)
                    self.cache["embeddings"].pop(0)
                    self.cache["answers"].pop(0)
                    self.cache["response_text"].pop(0)
    
    # looks in the cache for the closest question to the one just made by the use
    def ask(self, question: str) -> str:
        # Method to retrieve an answer from the cache or generate a new one
        start_time = time.time()
        try:
            # First we obtain the embeddings corresponding to the user question
            embedding = self.encoder.encode([question])

            # Search for the nearest neighbor in the index
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])

                    print("Answer recovered from Cache. ")
                    print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                    print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")
                    print(f"response_text: " + self.cache["response_text"][row_id])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache["response_text"][row_id]

            # Handle the case when there are not enough results
            # or Euclidean distance is not met, asking to chromaDB.
            answer = query_database([question], 1)
            response_text = answer["documents"][0][0]

            self.cache["questions"].append(question)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(answer)
            self.cache["response_text"].append(response_text)

            print("Answer recovered from ChromaDB. ")
            print(f"response_text: {response_text}")

            self.index.add(embedding)

            self.evict()

            store_cache(self.json_file, self.cache)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")