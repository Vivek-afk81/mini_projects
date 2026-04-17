## ML integration layer
import sys
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, "bug_intelligence"))


from core.classifier import classify_bug
from core.root_concept import find_root_concept
from core.embedder import embed_text
from core.analyzer import detect_pattern
from bugs.models import BugLog
from core.similarity import CATEGORY_THRESHOLDS
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b)


def find_similar_from_db(new_embedding, category):
    new_embedding = np.array(new_embedding)
    threshold = CATEGORY_THRESHOLDS.get(category, 0.65)
    similar = []

    past_bugs = BugLog.objects.filter(category=category)

    for bug in past_bugs:
        if not bug.embedding:
            continue
        stored_emb = np.array(bug.embedding)
        sim = cosine_similarity(new_embedding, stored_emb)

        if sim > threshold:
            similar.append({
                "text": bug.error_message,
                "similarity": round(sim, 2)
            })

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return similar


def process_bug(user,text, language="python"):
    
    # 1. classify
    classification = classify_bug(text)
    category = classification["category"]

    # 2. concept
    concept = find_root_concept(text, category, language)

    # 3. embedding
    embedding = embed_text(text).tolist()

    # 4. similarity
    similar = find_similar_from_db(embedding, category)

    # 5. pattern
    pattern = detect_pattern(similar, category)

    # 6. save
    bug = BugLog.objects.create(
        user=user,
        error_message=text,
        language=language,
        category=category,
        confidence=classification["confidence"],
        method=classification["method"],
        concept_name=concept["name"],
        explanation=concept["explanation"],
        embedding=embedding,
        pattern_detected=pattern["pattern"],
        pattern_message=pattern["message"]
    )

    # 7. response
    return {
        "id": bug.id,
        "category": category,
        "confidence": classification["confidence"],
        "method": classification["method"],
        "concept": concept,
        "similar_bugs": similar,
        "pattern": pattern
    }