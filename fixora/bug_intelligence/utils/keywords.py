# ── Special rules — checked before the main keyword loop ─────────
# Each entry: (condition_fn, category)
# Order matters — first match wins
SPECIAL_RULES = [
    # "int" + "str" anywhere = type confusion
    (lambda text: "int" in text and "str" in text, "type_error"),

    # C/C++ null pointer patterns
    (lambda text: "null" in text and "pointer" in text, "null_deref"),
    (lambda text: "nullptr" in text, "null_deref"),

    # local variable before assignment — very specific, must run early
    (lambda text: "local variable" in text and "assignment" in text, "scope_confusion"),
]


# ── Main keyword map — language aware ────────────────────────────
KEYWORD_MAP = {

    "async_misuse": [
        "await",
        "coroutine",
        "asyncio",
        "event loop",
        "runtimewarning coroutine",      # Python specific warning
        "was never awaited",
    ],

    "type_error": [
        "typeerror",
        "module object is not callable",
        "unsupported operand",
        "unsupported operand type",
        "must be str not int",
        "must be int not str",
        "cant multiply sequence",
        "str object does not support item assignment",
        "cannot concatenate",
        "can only concatenate",          # "can only concatenate str"
        "invalid literal",               # int("abc") → invalid literal for int()
        # C / C++
        "operand types",
        "no operator",                   # "no operator+ for string and int"
        "implicit conversion",
    ],

    "attribute_error":[
        "str object has no attribute reverse" 
    ],

    "off_by_one": [
        "indexerror",
        "index out of range",
        "list index out of range",
        "string index out of range",
        "tuple index out of range",
        # C / C++
        "array index",
        "out of bounds",
        "buffer overflow",
    ],

    "null_deref": [
        "nonetype",
        "nonetype object has no attribute",
        "none has no attribute",
        "object is none",
        "attributeerror",                # often caused by calling method on None
        # C / C++
        "null pointer",
        "segmentation fault",
        "segfault",
        "access violation",              # Windows equivalent of segfault
        "dereference",
        # Java
        "nullpointerexception",
    ],

    "scope_confusion": [
        "nameerror",
        "not defined",
        "is not defined",
        "undefined",
        "local variable referenced before assignment",
        "cannot access",                 # JS: "cannot access before initialization"
        "referenced before assignment",
        # C / C++
        "undeclared identifier",
        "was not declared",
    ],

    "syntax_error": [
        "syntaxerror",
        "invalid syntax",
        "indentationerror",
        "indentation error",
        "unexpected token",
        "unexpected indent",
        "expected expression",
        "missing colon",
        "unterminated string",
        # C / C++
        "expected semicolon",
        "expected",                      # broad but catches most C syntax errors
        "missing semicolon",
    ],

    "memory_error": [
        # C / C++ only — Python doesn't have these
        "malloc",
        "free",
        "memory leak",
        "heap corruption",
        "double free",
        "use after free",
        "buffer overflow",
        "stack overflow",
        "segmentation fault",            # can also mean null_deref — special rule handles ambiguity
    ],

    "logic_error": [
        # logic_error is hard to keyword-match reliably
        # these are weak signals — embedding layer handles most logic errors
        "wrong output",
        "unexpected result",
        "incorrect result",
        "expected but got",
        "assertion failed",
        "assertionerror",
    ],
}