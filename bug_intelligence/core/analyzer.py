def detect_pattern(similar_bugs,category=None):
    count=len(similar_bugs)

    if count>=3:
        return {
            "pattern":True,
            "message":f"You are repeatedly making {category} mistakes ({count} times)"
        }
    
    return {
        "pattern":False,
        "message":"No strong pattern yet"
    }
