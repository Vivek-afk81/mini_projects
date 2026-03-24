# main.py

from core.embedder import embed_text
from core.similarity import find_similar
from core.analyzer import detect_pattern
from core.classifier import classify_bug

bug_database = [
    {
        "text": "cannot concatenate str and int",
        "embedding": embed_text("cannot concatenate str and int"),
        "category": "type_error"
    },
    {
        "text": "unsupported operand type int and str",
        "embedding": embed_text("unsupported operand type int and str"),
        "category": "type_error"
    },
    {
        "text": "coroutine was never awaited",
        "embedding": embed_text("coroutine was never awaited"),
        "category": "async_misuse"
    }
]

if __name__ == "__main__":
    text = input("Enter bug: ")

    # classification
    classification = classify_bug(text)
    category = classification["category"]
    method   = classification["method"]

    # embeddings
    embedding = embed_text(text)

    # similarity
    similar = find_similar(embedding, bug_database, category)

    # pattern detection
    pattern = detect_pattern(similar, category)

    print(f"\nCategory:  {category}")
    print(f"Detected by: {method}")   # tells you which layer caught it

    print("\nSimilar Bugs:")
    for item in similar:
        print(f"  {item['text']} → {item['similarity']:.2f}")

    print("\nPattern Analysis:")
    print(pattern["message"])
