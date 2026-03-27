from utils.text import normalize

REFERENCE_ERRORS = {
    "type_error": [
        normalize("cannot concatenate str and int"),
        normalize("unsupported operand type int and str"),
        normalize("must be str not int"),
        normalize("cant multiply sequence by non int")
    ],
    "off_by_one": [
        normalize("list index out of range"),
        normalize("index out of range"),
        normalize("string index out of range")
    ],
    "null_deref": [
        normalize("NoneType object has no attribute"),
        normalize("object is None"),
        normalize("none has no attribute")
    ],
    "scope_confusion": [
        normalize("name is not defined"),
        normalize("NameError variable not defined"),
        normalize("local variable referenced before assignment")
    ],
    "syntax_error": [
        normalize("invalid syntax SyntaxError"),
        normalize("unexpected token"),
        normalize("missing colon")
    ],
    "async_misuse": [
        normalize("coroutine was never awaited"),
        normalize("asyncio event loop"),
        normalize("RuntimeWarning coroutine")
    ],
    "logic_error": [
        normalize("wrong output unexpected result"),
        normalize("incorrect behavior"),
        normalize("unexpected value returned")
    ],

    "memory_error": [
    normalize("malloc returned null memory allocation failed"),
    normalize("double free or corruption"),
    normalize("use after free invalid memory access"),
    normalize("heap buffer overflow"),
    normalize("segmentation fault core dumped"),
]
}