def detect_pattern(similar_bugs,category):
    count=len(similar_bugs)

    if count>=2:
        return {
            "pattern":True,
            "message": f" Pattern detected: You frequently make '{category}' mistakes ({count} similar bugs)"
        }
    
    return {
        "pattern":False,
        "message":"No strong pattern yet"
    }
