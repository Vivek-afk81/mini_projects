import os
import json
import re
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

# SAFE JSON PARSER
def _safe_parse(text: str) -> dict:
    """Extract JSON even if model adds extra text around it."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response")

    raw = match.group()
    raw = re.sub(
        r'\\(?!["\\/bfnrtu])',
        r'\\\\',
        raw
    )
    return json.loads(raw)

# MAIN FUNCTION

def evaluate_answer(
    user_answer: str,
    correct_concept: str,
    challenge_instruction: str
) -> dict:
    """
    Evaluate whether the user's answer shows genuine understanding.

    Input:
        user_answer           → what the user typed
        correct_concept       → what they needed to demonstrate
        challenge_instruction → the original challenge question

    Output:
        understood  → true / false / None (None = couldn't evaluate)
        confidence  → 0.0 to 1.0
        feedback    → warm encouraging sentence shown to user
        missed      → what they got wrong (null if understood is true)
    """

    # guard — empty answer
    if not user_answer or not user_answer.strip():
        return {
            "understood": False,
            "confidence": 0.0,
            "feedback":   "Please write your answer first — even a rough explanation helps!",
            "missed":     "No answer was provided"
        }

    # guard — too short to evaluate meaningfully
    if len(user_answer.strip().split()) < 4:
        return {
            "understood": False,
            "confidence": 0.0,
            "feedback":   "Can you explain a bit more? Try to describe what's wrong and why.",
            "missed":     "Answer too brief to evaluate understanding"
        }

    prompt = f"""A student was given this coding challenge:
"{challenge_instruction}"

The correct concept they needed to demonstrate:
"{correct_concept}"

Their answer:
"{user_answer}"

Evaluate whether their answer shows genuine understanding.
Be warm and encouraging — not harsh.

Return ONLY this JSON with no extra text:
{{
    "understood": true or false,
    "confidence": 0.0 to 1.0,
    "feedback": "one sentence of warm encouraging feedback",
    "missed": "if understood is false — what specifically they missed, else null"
}}"""

    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a warm coding mentor evaluating student understanding. Always respond with valid JSON only. No extra text before or after."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.2
        )
        return _safe_parse(response.choices[0].message.content)

    except Exception as e:
        print(f"DEBUG — Evaluator failed: {e}")
        return {
            "understood": None,
            "confidence": 0.0,
            "feedback":   "We couldn't evaluate your answer right now. Keep going!",
            "missed":     None
        }