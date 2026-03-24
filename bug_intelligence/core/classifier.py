from typing import Optional, Tuple, Dict
import numpy as np
from core.embedder import embed_text
from core.similarity import cosine_similarity
from data.reference_errors import REFERENCE_ERRORS
from utils.text import normalize

"""config"""
SIMILARITY_THRESHOLD = 0.5

# precomputing the reference embeddings
#runs once at stratup,never again
REFERENCE_EMBEDDINGS: Dict[str, list] = {}

for category, examples in REFERENCE_ERRORS.items():
    embeddings = [embed_text(example) for example in examples]
    normalized = [
        emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb
        for emb in embeddings
    ]
    REFERENCE_EMBEDDINGS[category] = normalized

"""LAYER 1 Keyword rules"""
KEYWORD_MAP = {
    "async_misuse":    ["await", "coroutine", "asyncio", "event loop"],
    "type_error":      ["typeerror", "unsupported operand"],
    "off_by_one":      ["indexerror", "index out of range", "list index"],
    "null_deref":      ["nonetype"],
    "scope_confusion": ["nameerror", "not defined"],
    "syntax_error":    ["syntaxerror", "invalid syntax"],
}

def _keyword_classify(text: str) -> Optional[str]:
    text = normalize(text)

    # special rule — must run before general loop
    if "int" in text and "str" in text:
        return "type_error"

    for category, keywords in KEYWORD_MAP.items():
        if any(keyword in text for keyword in keywords):
            return category

    return None

"""LAYER 2: EMBEDDING FALLBACK"""
def _embedding_classify(text: str) -> Tuple[str, float]:
    text = normalize(text)  # same normalization as reference errors
    unknown_embedding = embed_text(text)

    # normalize the vector
    norm = np.linalg.norm(unknown_embedding)
    if norm != 0:
        unknown_embedding = unknown_embedding / norm

    best_category = None
    best_score = -1

    for category, embeddings in REFERENCE_EMBEDDINGS.items():
        for ref_embedding in embeddings:
            score = cosine_similarity(unknown_embedding, ref_embedding)
            if score > best_score:
                best_score = score
                best_category = category

    # confidence too low — nothing matched well enough
    if best_score < SIMILARITY_THRESHOLD:
        return "logic_error", best_score  # logic_error as final fallback

    return best_category, best_score

"""MAIN PIPELINE"""
def classify_bug(text: str) -> dict:
    # guard — empty input
    if not text or not text.strip():
        return {
            "category":   "invalid_input",
            "method":     "none",
            "confidence": 0.0
        }

    # layer 1 — keywords always win when they match
    keyword_result = _keyword_classify(text)
    if keyword_result:
        return {
            "category":   keyword_result,
            "method":     "keyword",
            "confidence": 1.0
        }

    # layer 2 — only runs if keywords missed
    embedding_result, score = _embedding_classify(text)
    return {
        "category":   embedding_result,
        "method":     "embedding",
        "confidence": round(float(score), 4)
    }