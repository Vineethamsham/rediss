import os
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
INDEX_NAME = os.getenv("REDIS_INDEX_NAME")
VECTOR_DIM = int(os.getenv("REDIS_VECTOR_DIM", 1536))
DISTANCE_METRIC = os.getenv("REDIS_VECTOR_DISTANCE", "COSINE")

r = redis.from_url(REDIS_URL, decode_responses=False)

def create_index():
    try:
        r.ft(INDEX_NAME).info()
        print(f"Index '{INDEX_NAME}' already exists.")
    except:
        print(f"Creating index '{INDEX_NAME}'...")
        schema = (
            TextField("text"),
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": DISTANCE_METRIC.upper(),
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
        # Convert list[float] to float32 byte string
        import numpy as np
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        pipe.hset(key, mapping={
            "text": text,
            "embedding": vector_bytes
        })
    pipe.execute()
    print(f"✅ Indexed {len(texts)} documents into Redis.")
