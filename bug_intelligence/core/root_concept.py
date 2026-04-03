from typing import Optional
import numpy as np
import json
import re
import os

from huggingface_hub import InferenceClient

from core.embedder import embed_text
from core.similarity import cosine_similarity
from data.concept_map import CONCEPT_MAP
from utils.text import normalize

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HUGGINGFACE_TOKEN")
)


# CONFIG
CONCEPT_SIMILARITY_THRESHOLD = 0.4

# ==============================
# PRECOMPUTE CONCEPT EMBEDDINGS
# ==============================
CONCEPT_EMBEDDINGS = {}

for language, categories in CONCEPT_MAP.items():
    CONCEPT_EMBEDDINGS[language] = {}
    for category, concepts in categories.items():
        embedded = []
        for concept in concepts:
            emb = embed_text(normalize(concept["explanation"]))
            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm
            embedded.append((concept, emb))
        CONCEPT_EMBEDDINGS[language][category] = embedded

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
# LAYER 1: RULE BASED LOOKUP
# ==============================
def _rule_based_concept(
    category: str,
    error_text: str,
    language: str
) -> Optional[dict]:

    lang_map  = CONCEPT_MAP.get(language, CONCEPT_MAP["python"])
    concepts  = lang_map.get(category)

    if not concepts:
        return None

    if len(concepts) == 1:
        return concepts[0]

    error_normalized = normalize(error_text)
    for concept in concepts:
        if any(kw in error_normalized for kw in concept["keywords"]):
            return concept

    return concepts[0]

# ==============================
# LAYER 2: EMBEDDING MATCH
# ==============================
def _embedding_concept(
    category: str,
    error_text: str,
    language: str
) -> Optional[dict]:

    lang_embeddings   = CONCEPT_EMBEDDINGS.get(language, CONCEPT_EMBEDDINGS["python"])
    concept_embeddings = lang_embeddings.get(category)

    if not concept_embeddings:
        return None

    error_embedding = embed_text(normalize(error_text))
    norm = np.linalg.norm(error_embedding)
    if norm != 0:
        error_embedding = error_embedding / norm

    best_concept = None
    best_score   = -1

    for concept, emb in concept_embeddings:
        score = cosine_similarity(error_embedding, emb)
        if score > best_score:
            best_score   = score
            best_concept = concept

    if best_score < CONCEPT_SIMILARITY_THRESHOLD:
        return None

    return best_concept

# ==============================
# LAYER 3: LLM FALLBACK
# ==============================
def _llm_concept(
    category: str,
    error_text: str,
    language: str
) -> dict:

    prompt = f"""A {language} developer hit this error:
"{error_text}"

It is classified as: {category}

Identify the root concept they are missing.
The name must be 2-4 words like "C++ Syntax Rules" or "Pointer Safety".
The explanation must be 2-3 sentences a junior developer understands.
Do not return a category label as the name.

Return ONLY this JSON with no extra text:
{{
    "name": "2-4 word concept name",
    "explanation": "clear explanation for a junior developer"
}}"""

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
            max_tokens=200,
            temperature=0.3
        )
        return _safe_parse(response.choices[0].message.content)

    except Exception as e:
        print(f"DEBUG — LLM concept failed: {e}")
        return {
            "name":        "Programming Fundamentals",
            "explanation": "There is a fundamental concept missing here. Review the basics of this area carefully."
        }

# ==============================
# MAIN PIPELINE
# ==============================
def find_root_concept(
    error_text: str,
    category: str,
    language: str = "python"
) -> dict:

    # layer 1 — rules
    concept = _rule_based_concept(category, error_text, language)
    if concept:
        return {
            "name":        concept["name"],
            "explanation": concept["explanation"],
            "layer_used":  "rules"
        }

    # layer 2 — embeddings
    concept = _embedding_concept(category, error_text, language)
    if concept:
        return {
            "name":        concept["name"],
            "explanation": concept["explanation"],
            "layer_used":  "embeddings"
        }

    # layer 3 — LLM fallback
    concept = _llm_concept(category, error_text, language)
    return {
        "name":        concept["name"],
        "explanation": concept["explanation"],
        "layer_used":  "llm"
    }