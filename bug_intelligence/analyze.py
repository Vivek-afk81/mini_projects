# analyze.py

from core.classifier import classify_bug
from core.root_concept import find_root_concept
from core.challenge_generator import generate_challenge

# MAIN ENTRY POINT

def analyze(
    error_text: str,
    language:   str = "python",
    user_level: str = "beginner"
) -> dict:
    """
    Full pipeline. Three stages. One response.
    This is the only function backend ever calls.
    
    Input:
        error_text  → the raw error the user pasted
        language    → python / c / cpp
        user_level  → beginner / intermediate

    Output:
        category      → what type of error
        method        → how it was classified
        confidence    → how sure the classifier was
        root_concept  → what concept they're missing
        challenge     → tomorrow's personalized challenge
    """

    # guard — empty input
    if not error_text or not error_text.strip():
        return {
            "success": False,
            "error":   "No error text provided"
        }

    # ── stage 1: classify 
    classification = classify_bug(error_text)
    category       = classification["category"]
    method         = classification["method"]
    confidence     = classification["confidence"]

    # ── stage 2: root concept
    concept = find_root_concept(
        error_text = error_text,
        category   = category,
        language   = language
    )

    # ── stage 3: challenge
    challenge = generate_challenge(
        error_text   = error_text,
        category     = category,
        root_concept = concept["name"],
        language     = language,
        user_level   = user_level
    )

    # ── pack and return ──
    return {
        "success": True,
        "classification": {
            "category":   category,
            "method":     method,
            "confidence": confidence
        },
        "root_concept": {
            "name":        concept["name"],
            "explanation": concept["explanation"],
            "layer_used":  concept["layer_used"]
        },
        "challenge": {
            "instruction":     challenge["instruction"],
            "code_snippet":    challenge["code_snippet"],
            "hint":            challenge["hint"],
            "correct_concept": challenge["correct_concept"]
        }
    }