import os
import json
import re
import numpy as np
from huggingface_hub import InferenceClient
from core.embedder import embed_text
from core.similarity import cosine_similarity
from utils.text import normalize

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

SEMANTIC_OVERRIDE_THRESHOLD = 0.60
SEMANTIC_OVERRIDE_AFTER_LLM = 0.50

def _safe_parse(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in response")
    raw = match.group()
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    return json.loads(raw)

def _semantic_similarity(user_answer: str, concept_explanation: str) -> float:
    a = embed_text(normalize(user_answer))
    b = embed_text(normalize(concept_explanation))
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(cosine_similarity(a, b))

def evaluate_answer(user_answer, correct_concept, challenge_instruction, concept_explanation="") -> dict:
    # guard — empty
    if not user_answer or not user_answer.strip():
        return {
            "understood": False,
            "confidence": 0.0,
            "feedback":   "Please write your answer first — even a rough explanation helps!",
            "missed":     "No answer was provided"
        }

    # guard — too short
    if len(user_answer.strip().split()) < 4:
        return {
            "understood": False,
            "confidence": 0.0,
            "feedback":   "Can you explain a bit more? Try to describe what's wrong and why.",
            "missed":     "Answer too brief to evaluate understanding"
        }

    # semantic check — fires before LLM to catch phrasing mismatches
    if concept_explanation:
        sim = _semantic_similarity(user_answer, concept_explanation)
        if sim >= SEMANTIC_OVERRIDE_THRESHOLD:
            return {
                "understood": True,
                "confidence": round(sim, 4),
                "feedback":   "You got it — your explanation shows you understand what's going on here.",
                "missed":     None
            }

    # LLM evaluation
    prompt = f"""You are a warm coding mentor evaluating a student's understanding.

CHALLENGE THEY WERE GIVEN:
"{challenge_instruction}"

UNDERLYING CONCEPT BEING TESTED:
"{concept_explanation or correct_concept}"

ONE POSSIBLE CORRECT ANSWER:
"{correct_concept}"

STUDENT'S ANSWER:
"{user_answer}"

EVALUATION RULES — follow these exactly:
- Mark understood=true if the student demonstrates they grasp the CONCEPT, even partially
- Accept any valid fix — not just the one in "correct answer"
- Accept informal language — "age is text not a number" is the same as "input() returns a string"
- If the code has multiple bugs and they found one real one, mark understood=true
- Only mark understood=false if they show no understanding of the core concept at all
- Never penalise for incomplete sentences or rough phrasing

Return ONLY this JSON with no extra text:
{{
    "understood": true or false,
    "confidence": 0.0 to 1.0,
    "feedback": "one warm encouraging sentence — acknowledge what they got right even if partial",
    "missed": "only if understood=false — what concept gap remains, else null"
}}"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a warm coding mentor. Evaluate understanding generously. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2
        )
        result = _safe_parse(response.choices[0].message.content)

        # semantic override — if LLM says False but answer is close enough
        if not result.get("understood") and concept_explanation:
            sim = _semantic_similarity(user_answer, concept_explanation)
            if sim >= SEMANTIC_OVERRIDE_AFTER_LLM:
                result["understood"] = True
                result["confidence"] = round(sim, 4)
                result["feedback"]   = "You're on the right track — your thinking shows you understand the core issue."
                result["missed"]     = None

        return result

    except Exception as e:
        print(f"DEBUG — Evaluator failed: {e}")
        return {
            "understood": None,
            "confidence": 0.0,
            "feedback":   "We couldn't evaluate your answer right now. Keep going!",
            "missed":     None
        }