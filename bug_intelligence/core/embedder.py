import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import login


token=os.getenv("HUGGINGFACE_TOKEN")

if token:
    login(token)

# Load model (works without token too)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    embeddings = model.encode(text,normalize_embeddings=True)
    return embeddings