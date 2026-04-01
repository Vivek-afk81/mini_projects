# core/challenge_generator.py

import os
import json
import re
from huggingface_hub import InferenceClient

from utils.text import normalize
from utils.fallback_challenges import FALLBACK_CHALLENGES

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

# ==============================
# CONFIG
# ==============================
LEVEL_DESCRIPTIONS = {
    "beginner":     "has less than 6 months of experience",
    "intermediate": "has 6 months to 2 years of experience"
}

# ==============================
# SAFE JSON PARSER
# ==============================
def _safe_parse(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response")
    raw = match.group()
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    return json.loads(raw)

# ==============================
# PROMPT BUILDER
# ==============================
def _build_prompt(
    error_text: str,
    category: str,
    root_concept: str,
    language: str,
    user_level: str
) -> str:
    level_desc = LEVEL_DESCRIPTIONS.get(user_level, "is a beginner")

    return f"""You are a warm coding mentor inside an app called Fixora.

A {language} developer who {level_desc} hit this error yesterday:
"{error_text}"

Root concept they are missing: {root_concept}

Your job:
1. Write a short buggy {language} code snippet (max 6 lines)
2. The bug in YOUR code must directly demonstrate: {root_concept}
3. Use a completely different scenario from their original error
4. Write the correct_concept to describe the EXACT bug in YOUR code snippet
5. The hint should point toward the bug without giving it away

Example of a good instruction:
"Hey! Something looks off in this snippet — can you spot what the compiler would complain about and why?"

Return ONLY this JSON with no extra text:
{{
    "instruction": "a specific warm question about YOUR code snippet",
    "code_snippet": "your buggy {language} code here — different scenario, max 6 lines",
    "hint": "gentle nudge toward the bug in your code without giving the answer",
    "correct_concept": "one sentence describing the exact bug in your code snippet specifically"
}}"""

# ==============================
# FALLBACK HELPER
# ==============================
def _get_fallback(category: str) -> dict:
    return FALLBACK_CHALLENGES.get(
        category,
        FALLBACK_CHALLENGES["logic_error"]
    )

# ==============================
# MAIN FUNCTION
# ==============================
def generate_challenge(
    error_text: str,
    category: str,
    root_concept: str,
    language: str = "python",
    user_level: str = "beginner"
) -> dict:
    """
    Generate a personalized daily challenge from the user's error.
    Uses HuggingFace LLM. Falls back to hardcoded challenge if LLM fails.
    """

    prompt = _build_prompt(
        error_text   = error_text,
        category     = category,
        root_concept = root_concept,
        language     = language,
        user_level   = user_level
    )

    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a coding mentor. Always respond with valid JSON only. No extra text before or after."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
            temperature=0.7
        )

        result = _safe_parse(response.choices[0].message.content)

        # validate all required fields exist
        required = ["instruction", "code_snippet", "hint", "correct_concept"]
        if not all(field in result for field in required):
            raise ValueError("Missing fields in LLM response")

        # guard — instruction still a placeholder
        if "warm question" in result["instruction"].lower():
            raise ValueError("LLM returned placeholder instruction")

        return result

    except Exception as e:
        print(f"DEBUG — Challenge generation failed: {e}")
        return _get_fallback(category)
