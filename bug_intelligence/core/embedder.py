from sentence_transformers import SentenceTransformer

model=SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    embeddings=model.encode(text)
    return embeddings