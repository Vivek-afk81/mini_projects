from core.embedder import embed_text
from core.similarity import find_similar
from core.analyzer import detect_pattern
from core.classifier import classify_bug
from core.root_concept import find_root_concept
from core.challenge_generator import generate_challenge

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
    text       = input("Enter bug: ")
    language   = input("Language (python/c/cpp): ").strip() or "python"
    user_level = input("Level (beginner/intermediate): ").strip() or "beginner"

    # stage 1
    classification = classify_bug(text)
    category       = classification["category"]
    method         = classification["method"]
    confidence     = classification["confidence"]

    # stage 2
    concept = find_root_concept(text, category, language)

    # stage 3
    challenge = generate_challenge(
        error_text   = text,
        category     = category,
        root_concept = concept["name"],
        language     = language,
        user_level   = user_level
    )

    # similarity + pattern
    embedding = embed_text(text)
    similar   = find_similar(embedding, bug_database, category)
    pattern   = detect_pattern(similar, category)

    # output
    print(f"\n--- STAGE 1: CLASSIFICATION ---")
    print(f"Category:    {category}")
    print(f"Detected by: {method} (confidence: {confidence})")

    print(f"\n--- STAGE 2: ROOT CONCEPT ---")
    print(f"Concept:     {concept['name']}")
    print(f"Explanation: {concept['explanation']}")
    print(f"Found by:    {concept['layer_used']}")

    print(f"\n--- STAGE 3: TOMORROW'S CHALLENGE ---")
    print(f"Instruction: {challenge['instruction']}")
    print(f"Code:\n{challenge['code_snippet']}")
    print(f"Hint:        {challenge['hint']}")
    print(f"Concept:     {challenge['correct_concept']}")

    print(f"\n--- PATTERN ANALYSIS ---")
    print(pattern["message"])
