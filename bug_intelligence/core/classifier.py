def classify_bug(text):
    text = text.lower()

    if any(word in text for word in ["await", "coroutine", "asyncio", "event loop"]):
        return "async_misuse"

    if "typeerror" in text or "unsupported operand" in text or ("int" in text and "str" in text):
        return "type_error"

    if "indexerror" in text or "index out of range" in text or "list index" in text:
        return "off_by_one"

    if "nonetype" in text or "none type" in text:
        return "null_deref"

    if "nameerror" in text or "not defined" in text:
        return "scope_confusion"

    if "syntaxerror" in text or "invalid syntax" in text:
        return "syntax_error"

    return "logic_error"