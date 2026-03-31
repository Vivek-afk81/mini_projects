import os
from typing import Optional
import numpy as np
import re
import json

from huggingface_hub import InferenceClient

from core.embedder import embed_text
from core.similarity import cosine_similarity
from data.concept_map import CONCEPT_MAP
from utils.text import normalize

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",  
    token=os.getenv("HUGGINGFACE_TOKEN")
)

"""Configiration"""
CONCEPT_SIMILARITY_THRESHOLD = 0.4

#precomputing concept threshold
CONCEPT_EMBEDDINGS={}

# in root_concept.py — replace your precomputation block

for language, categories in CONCEPT_MAP.items():
    CONCEPT_EMBEDDINGS[language] = {}

    for category, concepts in categories.items():
        # collect all explanations at once
        texts = [normalize(c["explanation"]) for c in concepts]
        
        # one model call instead of N calls
        embeddings = embed_text(texts)  # returns array of shape (N, dim)
        
        embedded = []
        for concept, emb in zip(concepts, embeddings):
            norm = np.linalg.norm(emb)
            if norm != 0:
                emb = emb / norm
            embedded.append((concept, emb))
        
        CONCEPT_EMBEDDINGS[language][category] = embedded


"""LAYER 1: RULE BASED LOOKUP"""
def _rule_based_concept(category: str, error_text: str, language: str) -> Optional[dict]:
    lang_map = CONCEPT_MAP.get(language, CONCEPT_MAP["python"])
    concepts = lang_map.get(category)

    if not concepts:
        return None

    if len(concepts) == 1:
        return concepts[0]

    error_normalized = normalize(error_text)
    for concept in concepts:
        if any(kw in error_normalized for kw in concept["keywords"]):
            return concept

    return None  # ← no keyword matched, let embeddings try


"""LAYER 2 : EMBEDDING MATCH"""
    
def _embedding_concept(
    category: str,
    error_text: str,
    language: str
) -> Optional[dict]:

    lang_embeddings = CONCEPT_EMBEDDINGS.get(
        language,
        CONCEPT_EMBEDDINGS["python"]
    )
    concept_embeddings = lang_embeddings.get(category)

    if not concept_embeddings:
        return None

    error_embedding = embed_text(normalize(error_text))
    norm = np.linalg.norm(error_embedding)
    if norm != 0:
        error_embedding = error_embedding / norm

    best_concept = None
    best_score = -1

    for concept, emb in concept_embeddings:
        score = cosine_similarity(error_embedding, emb)
        if score > best_score:
            best_score = score
            best_concept = concept

    # threshold check — don't return if not confident
    if best_score < CONCEPT_SIMILARITY_THRESHOLD:
        return None

    return best_concept


"""LAYER 3 LLM FALLBACK"""

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

def _llm_concept(
    category: str,
    error_text: str,
    language: str
) -> dict:

    prompt = f"""You are a coding mentor. Respond with valid JSON only.

A {language} developer hit this error:
"{error_text}"

It is classified as: {category}

Return ONLY this JSON with no extra text:
{{
    "name": "short concept name",
    "explanation": "clear explanation for a junior developer"
}}"""

    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a coding mentor. Always respond with valid JSON only."
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

    except Exception:
        return {
            "name": "Unknown Concept",
            "explanation": "This error points to a gap in your understanding. Review the fundamentals of this area carefully."
        }
    

def find_root_concept(
    error_text: str,
    category: str,
    language: str = "python"
) -> dict:

    # layer 1 — rules
    concept = _rule_based_concept(category, error_text, language)
    if concept:
        return {
            "name":       concept["name"],
            "explanation": concept["explanation"],
            "layer_used": "rules"
        }

    # layer 2 — embeddings
    concept = _embedding_concept(category, error_text, language)
    if concept:
        return {
            "name":       concept["name"],
            "explanation": concept["explanation"],
            "layer_used": "embeddings"
        }

    # layer 3 — LLM fallback
    concept = _llm_concept(category, error_text, language)
    return {
        "name":       concept["name"],
        "explanation": concept["explanation"],
        "layer_used": "llm"
    }