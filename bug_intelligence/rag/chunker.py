##this will take a text document and splits it into overlapping chunks,
#Why overlapping? so that a revelant sentence near a chunk boundary
#doesn't get cut off and missed during retrieval
from pathlib import Path
import os

CHUNK_SIZE=200 #chars per chunk
CHUNK_OVERLAP=50 # how many chars bleed into the nect chunk

def chunk_text(text: str) -> list[str]:
    """split a string into overlapping chunks"""
    chunks=[]
    start=0

    while start <len(text):
        end=start+CHUNK_SIZE
        chunk=text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        start+=CHUNK_SIZE-CHUNK_OVERLAP #overlapping

    return chunks


def load_and_chunk_docs(docs_dir:str) -> list[dict]:
    """
    Walk through all .txt files in docs_dir (recursively).
    Return chunks with language and filename metadata
    """

    docs_path=Path(docs_dir)
    all_chunks=[]

    #recursively find all .txt files

    for filepath in docs_path.rglob("*.txt"): #recursive walking → .rglob()
        language=filepath.parent.name
        filename=filepath.name

        text=filepath.read_text(encoding="utf-8")
        chunks=chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "text":chunk,
                "language":language,
                "source":filename
            })
    
    return all_chunks


if __name__ == "__main__":
    sample = "This is a test. " * 50
    chunks = chunk_text(sample)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
    print(f"Second chunk: {chunks[1]}")