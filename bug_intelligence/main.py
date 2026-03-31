#for testing purpose
from analyze import analyze
from core.evaluator import evaluate_answer

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

        print("\n" + "=" * 50)
        print("EVALUATOR TEST")
        print("=" * 50)

        challenge_instruction = "What's wrong with this code and why does it crash?"
        correct_concept = "input() always returns a string, never a number. You must convert it using int() before comparing."

        test_answers = [
            {
                "label": "Good answer",
                "answer": "input() returns a string so comparing it with 18 using > fails because you cant compare str and int directly you need to wrap it with int()"
            },
            {
                "label": "Vague answer",
                "answer": "its a type error"
            },
            {
                "label": "Wrong answer",
                "answer": "the print statement is missing"
            },
            {
                "label": "Empty answer",
                "answer": ""
            }
        ]

        for test in test_answers:
            print(f"\n--- {test['label']} ---")
            print(f"Answer: {test['answer']}")
            
            result = evaluate_answer(
                user_answer           = test["answer"],
                correct_concept       = correct_concept,
                challenge_instruction = challenge_instruction
            )
            
            print(f"Understood:  {result['understood']}")
            print(f"Confidence:  {result['confidence']}")
            print(f"Feedback:    {result['feedback']}")
            print(f"Missed:      {result['missed']}")

                # ==============================



