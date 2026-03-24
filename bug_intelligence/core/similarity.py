import numpy as np

def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)

CATEGORY_THRESHOLDS = {
    "type_error": 0.7,
    "async_misuse": 0.65,
    "syntax_error": 0.6,
    "logic_error": 0.75
}


def find_similar(new_embeddings,database,category):

    threshold=CATEGORY_THRESHOLDS.get(category, 0.65)
    results=[]

    for bug in database:
        if bug["category"] !=category:
            continue

        sim=cosine_similarity(new_embeddings,bug["embedding"])

        if sim>threshold:
            results.append({
                "text":bug["text"],
                "similarity":round(sim,2)

            })
            

    #sorting by similarity
    results.sort(key=lambda x: x["similarity"],reverse=True)

    return results
