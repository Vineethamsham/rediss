 ## REdis
 
 project/
│
├── redis_indexer.py
├── chunks/
│   └── azure_blob_chunks.json
├── .env
├── utils/
│   ├── embeddings.py
│   └── redis_helpers.py


OPENAI_API_KEY=your_openai_key
REDIS_URL=redis://:<your_primary_key>@devdiaedswu2-redis01.westus2.redis.azure.net:6380/0
REDIS_INDEX_NAME=tfb_chunks_index


import openai
import os
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-ada-002"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embed_text_batch(texts):
    response = openai.Embedding.create(input=texts, model=EMBED_MODEL)
    return [item["embedding"] for item in response["data"]]



import os
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import json
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
INDEX_NAME = os.getenv("REDIS_INDEX_NAME")

r = redis.from_url(REDIS_URL, decode_responses=False)

def create_index(vector_dim=1536):
    try:
        r.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists.")
    except:
        print(f"Creating index: {INDEX_NAME}")
        schema = (
            TextField("text"),
            VectorField("embedding",
                        "FLAT", {
                            "TYPE": "FLOAT32",
                            "DIM": vector_dim,
                            "DISTANCE_METRIC": "COSINE",
                            "INITIAL_CAP": 10000,
                            "BLOCK_SIZE": 1000
                        })
        )
        definition = IndexDefinition(prefix=[f"{INDEX_NAME}:"], index_type=IndexType.HASH)
        r.ft(INDEX_NAME).create_index(schema, definition=definition)

def index_documents(texts, embeddings):
    pipe = r.pipeline()
    for i, (text, vector) in enumerate(zip(texts, embeddings)):
        key = f"{INDEX_NAME}:{i}"
        vector_bytes = bytearray(float(x).hex() for x in vector)
        pipe.hset(key, mapping={
            "text": text,
            "embedding": bytes(vector)
        })
    pipe.execute()
    print(f"✅ Indexed {len(texts)} documents into Redis.")



import os
import json
from dotenv import load_dotenv
from utils.embeddings import embed_text_batch
from utils.redis_helpers import create_index, index_documents

load_dotenv()

CHUNKS_FILE = "chunks/azure_blob_chunks.json"
BATCH_SIZE = 10  # You can change based on performance

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data if item.get("text")]

def embed_chunks(chunks):
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} of {len(chunks) // BATCH_SIZE + 1}")
        result = embed_text_batch(batch)
        embeddings.extend(result)
    return embeddings

if __name__ == "__main__":
    create_index()
    chunks = load_chunks()
    embedded = embed_chunks(chunks)
    index_documents(chunks, embedded)
