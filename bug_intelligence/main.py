from core.embedder import embed_text

if __name__=="__main__":
    text=input("Enter bug: ")

    embeddings=embed_text(text)

    print("embeddings shape",len(embeddings))
