import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# login (optional)
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token)

model = None  # ❗ don't load immediately


def get_model():
    global model
    if model is None:
        print("Loading ML model...")  # for debugging
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


# def embed_text(text):
#     model_instance = get_model()
#     embeddings = model_instance.encode(text, normalize_embeddings=True)
#     return embeddings
# in embedder.py
def embed_text(text: str | list):
    model_instance = get_model()
    return model_instance.encode(text, normalize_embeddings=True)