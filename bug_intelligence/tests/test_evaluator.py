import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evaluator import evaluate_answer

# ── shared challenge context ──────────────────────────────────────────────────

CHALLENGE_INSTRUCTION = "What is wrong with this code and why does it crash?"

CODE = """
age = input("Enter age: ")
if age > 18:
    print("adult")
"""

CORRECT_CONCEPT    = "input() always returns a string, never a number"
CONCEPT_EXPLANATION = "Python never converts types automatically. input() always returns a string, not a number. You have to convert yourself using int(), str(), or float()."

# ── test cases ────────────────────────────────────────────────────────────────

TESTS = [
    {
        "name": "Clear correct answer",
        "answer": "age is a string because input() returns text, so comparing it with > 18 doesn't work",
        "expected_understood": True
    },
    {
        "name": "Informal phrasing but correct understanding",
        "answer": "age is text not a number so the comparison breaks",
        "expected_understood": True
    },
    {
        "name": "Different wording, same concept",
        "answer": "you need to convert age to int using int(age) before the if statement",
        "expected_understood": True
    },
    {
        "name": "Partially correct — one valid fix among multiple",
        "answer": "the if condition is wrong because age hasn't been converted",
        "expected_understood": True
    },
    {
        "name": "Completely wrong — no understanding",
        "answer": "the print statement is indented incorrectly",
        "expected_understood": False
    },
    {
        "name": "Empty answer",
        "answer": "",
        "expected_understood": False
    },
    {
        "name": "Too short",
        "answer": "type error",
        "expected_understood": False
    },
]

# ── runner ────────────────────────────────────────────────────────────────────

def run_tests():
    passed = 0
    failed = 0

    print("\n" + "="*60)
    print("EVALUATOR TEST SUITE")
    print("="*60)

    for i, test in enumerate(TESTS, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Answer: \"{test['answer']}\"")

        result = evaluate_answer(
            user_answer           = test["answer"],
            correct_concept       = CORRECT_CONCEPT,
            challenge_instruction = CHALLENGE_INSTRUCTION,
            concept_explanation   = CONCEPT_EXPLANATION
        )

        understood = result.get("understood")
        expected   = test["expected_understood"]
        ok         = understood == expected

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"Result:   understood={understood}, confidence={result.get('confidence')}")
        print(f"Feedback: {result.get('feedback')}")
        if result.get("missed"):
            print(f"Missed:   {result.get('missed')}")
        print(f"Status:   {status}  (expected understood={expected})")

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_tests()