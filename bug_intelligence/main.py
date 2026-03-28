from analyze import analyze

if __name__ == "__main__":
    print("FIXORA — AI Layer Test")

    text       = input("\nPaste your error:\n> ").strip()
    language   = input("Language (python/c/cpp): ").strip() or "python"
    user_level = input("Level (beginner/intermediate): ").strip() or "beginner"

    print("\nAnalyzing...")
    result = analyze(text, language, user_level)

    if not result["success"]:
        print(f"Error: {result['error']}")
    else:
        c = result["classification"]
        r = result["root_concept"]
        ch = result["challenge"]


        print("STAGE 1 — CLASSIFICATION")
        print(f"{'=' * 50}")
        print(f"Category:    {c['category']}")
        print(f"Detected by: {c['method']}")
        print(f"Confidence:  {c['confidence']}")

        print(f"\n{'-' * 50}")
        print("STAGE 2 — ROOT CONCEPT")
        print(f"{'-' * 50}")
        print(f"Concept:     {r['name']}")
        print(f"Explanation: {r['explanation']}")
        print(f"Found by:    {r['layer_used']}")

        print(f"\n{'-' * 50}")
        print("STAGE 3 — TOMORROW'S CHALLENGE")
        print(f"{'-' * 50}")
        print(f"Task:    {ch['instruction']}")
        print(f"\nCode:\n{ch['code_snippet']}")
        print(f"\nHint:    {ch['hint']}")
        print(f"Concept: {ch['correct_concept']}")