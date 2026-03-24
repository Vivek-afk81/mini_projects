# data/reference_errors.py

REFERENCE_ERRORS = {
    "type_error": [
        "cannot concatenate str and int",
        "unsupported operand type int and str",
        "must be str not int",
        "can't multiply sequence by non-int"
    ],
    "off_by_one": [
        "list index out of range",
        "index out of range",
        "string index out of range"
    ],
    "null_deref": [
        "NoneType object has no attribute",
        "object is None",
        "none has no attribute"
    ],
    "scope_confusion": [
        "name is not defined",
        "NameError variable not defined",
        "local variable referenced before assignment"
    ],
    "syntax_error": [
        "invalid syntax SyntaxError",
        "unexpected token",
        "missing colon"
    ],
    "async_misuse": [
        "coroutine was never awaited",
        "asyncio event loop",
        "RuntimeWarning coroutine"
    ],
    "logic_error": [
        "wrong output unexpected result",
        "incorrect behavior",
        "unexpected value returned"
    ]
}