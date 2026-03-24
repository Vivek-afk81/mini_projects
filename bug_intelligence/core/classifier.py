from typing import Optional

from core.embedder import embed_text
from core.similarity import cosine_similarity
from data.reference_errors import REFERENCE_ERRORS

import numpy as np

# precompute reference embeddings once at startup 
# instead of recomputing every time a bug is submitted

REFERENCE_EMBEDDINGS={}

for category,examples in REFERENCE_ERRORS.items():
    embeddings=[embed_text(example) for example in examples]

    REFERENCE_EMBEDDINGS[category]=np.mean(embeddings,axis=0)


"""LAYER 1 : defining the keyword rules"""

def _keyword_classify(text: str) -> Optional[str]:
    """
    Classify error type based on keyword matching.
    Returns a category string or None if no match is found.
    """
    normalized_text = text.lower()

    keyword_map = {
        "async_misuse": ["await", "coroutine", "asyncio", "event loop"],
        "type_error": ["typeerror", "unsupported operand", "int", "str"],
        "off_by_one": ["indexerror", "index out of range", "list index"],
        "null_deref": ["nonetype", "none"],
        "scope_confusion": ["nameerror", "not defined"],
        "syntax_error": ["syntaxerror", "invalid syntax"],
    }

    for category, keywords in keyword_map.items():
        if any(keyword in normalized_text for keyword in keywords):
            return category

    return None  # No match → fallback to embeddings

"""LAYER 2 : embedding fallback"""

def _embedding_classify(text:str) ->str:
    unknown_embedding=embed_text(text)

    best_category="logic_error"
    best_score=-1

    for caegory, ref_embedding in REFERENCE_EMBEDDINGS.items():
        score=cosine_similarity(unknown_embedding,ref_embedding)

        if score>best_score:
            best_score=score

            best_category=category

    return best_category

"""the main function that calls everything"""

def classify_bug(text:str) -> dict:
    result=_keyword_classify(text)
    method ="keyword"


    if result is None:
        result=_embedding_classify(text)
        method="embeddings"
    
    return {
        "category":result,
        "method":method
    }
