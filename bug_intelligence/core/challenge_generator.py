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


# CONFIG

LEVEL_DESCRIPTIONS = {
    "beginner":     "has less than 6 months of experience",
    "intermediate": "has 6 months to 2 years of experience"
}


# SAFE JSON PARSER

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


# PROMPT BUILDER

def _build_prompt(
    error_text: str,
    category: str,
    root_concept: str,
    language: str,
    user_level: str
) -> str:
    level_desc = LEVEL_DESCRIPTIONS.get(user_level, "is a beginner")

    return f"""You are a warm, encouraging coding mentor inside an app called Fixora.

A {language} developer who {level_desc} hit this error yesterday:
"{error_text}"

The root concept they are missing is: {root_concept}

Generate a daily challenge that follows these rules:
1. Use a DIFFERENT scenario than their original error
   (they must think, not just remember their fix)
2. Test whether they now understand: {root_concept}
3. Write warmly — like a mentor, not an examiner
4. Give a hint that points toward the concept without giving the answer
5. Keep the code snippet short — maximum 6 lines
6. Write the code snippet in {language}

Return ONLY this JSON with no extra text:
{{
    "instruction": "warm question asking them to spot the problem",
    "code_snippet": "short buggy code in a different scenario",
    "hint": "gentle nudge toward the right concept",
    "correct_concept": "one sentence — what they should understand after this"
}}"""


# HARD FALLBACK CHALLENGES

# if LLM fails completely — these fire instead
# one per category, better than crashing
def _get_fallback(category: str) -> dict:
    """Return safe fallback challenge if LLM fails."""
    return FALLBACK_CHALLENGES.get(
        category,
        FALLBACK_CHALLENGES["logic_error"]
    )


# MAIN FUNCTION

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
            temperature=0.7   # slightly higher than root_concept
                              # more creative challenges
        )

        result = _safe_parse(response.choices[0].message.content)

        # validate all required fields exist
        required = ["instruction", "code_snippet", "hint", "correct_concept"]
        if not all(field in result for field in required):
            raise ValueError("Missing fields in LLM response")

        return result
    
    except Exception as e:
        print(f"DEBUG — LLM failed: {e}")
        return _get_fallback(category)

    # except Exception:
    #     # LLM failed — return hardcoded fallback for this category
    #     return FALLBACK_CHALLENGES.get(
    #         category,
    #         FALLBACK_CHALLENGES["logic_error"]  # default if category unknown
    #     )

