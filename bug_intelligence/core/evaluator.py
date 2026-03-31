import os
import json
import re
from huggingface_hub import InferenceClient


client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",  
    token=os.getenv("HUGGINGFACE_TOKEN")
)

def _safe_parse(text: str) -> dict:
    """Extract JSON even if model adds extra text around it."""
    
    # find the JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response")
    
    raw = match.group()
    
    # fix invalid escape sequences before parsing
    # replaces any \x that isn't a valid JSON escape
    raw = re.sub(
        r'\\(?!["\\/bfnrtu])',  # match \ not followed by valid escape chars
        r'\\\\',                # replace with double backslash
        raw
    )
    
    return json.loads(raw)

def evaluate_answer(
    user_answer: str,
    correct_concept: str,
    challenge_instruction: str
) -> dict:

    prompt = f"""A student was given this coding challenge:
"{challenge_instruction}"

The correct concept they needed to demonstrate:
"{correct_concept}"

Their answer:
"{user_answer}"

Evaluate whether their answer shows genuine understanding.
Be warm and encouraging — not harsh.

Return ONLY this JSON:
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
                    "content": "You are a warm coding mentor evaluating student understanding. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.2  # low — consistent evaluation
        )
        return _safe_parse(response.choices[0].message.content)

    except Exception:
        return {
            "understood": None,
            "confidence": 0.0,
            "feedback":   "We couldn't evaluate your answer right now. Keep going!",
            "missed":     None
        }