# data/concept_map.py

CONCEPT_MAP = {
    "python": {
        "type_error": [
            {
                "name": "Python Type System",
                "explanation": "Python never converts types automatically. input() always returns a string, not a number. You have to convert yourself using int(), str(), or float().",
                "keywords": ["input", "concatenate", "str", "int", "float", "operand"]
            },
            {
                "name": "String Formatting",
                "explanation": "When building strings with variables, use f-strings: f'Hello {name}' instead of 'Hello' + name. The + operator cannot mix strings and other types.",
                "keywords": ["concatenate", "combine", "format", "print"]
            }
        ],
        "off_by_one": [
            {
                "name": "Zero Based Indexing",
                "explanation": "Lists in Python start at index 0 not 1. A list with 5 items has indices 0,1,2,3,4. Index 5 does not exist. Always subtract 1 when accessing the last item.",
                "keywords": ["index", "range", "list", "array", "last"]
            }
        ],
        "null_deref": [
            {
                "name": "None Checking",
                "explanation": "A variable is None when a function returns nothing or an operation fails silently. Always check if x is not None before using x.",
                "keywords": ["none", "null", "attribute", "object", "return"]
            }
        ],
        "scope_confusion": [
            {
                "name": "Variable Scope",
                "explanation": "Variables defined inside a function or loop do not exist outside it. Define variables before the block where you need them or return them explicitly.",
                "keywords": ["defined", "scope", "global", "local", "function"]
            }
        ],
        "syntax_error": [
            {
                "name": "Python Syntax Rules",
                "explanation": "Python is strict about colons, indentation, and brackets. Every if, for, while, and def needs a colon. Every open bracket needs closing.",
                "keywords": ["syntax", "indent", "colon", "bracket", "def", "if", "for", "eof", "parenthesis"]
            }
        ],
        "async_misuse": [
            {
                "name": "Async Await Pattern",
                "explanation": "async functions must be called with await. You can only use await inside an async function. Without await you get a coroutine object not a result.",
                "keywords": ["await", "async", "coroutine", "event loop"]
            }
        ],
        "logic_error": [
            {
                "name": "Program Logic",
                "explanation": "The code runs without crashing but produces wrong results. This usually means a wrong condition, wrong variable, or wrong order of operations. Add print statements to trace what is actually happening.",
                "keywords": ["wrong", "incorrect", "unexpected", "output", "result"]
            }
        ]
    },
    "c": {
        "type_error": [
            {
                "name": "C Type System",
                "explanation": "C is strictly typed. You cannot implicitly mix pointers and integers or signed and unsigned types. Explicit casting is required and must be done carefully.",
                "keywords": ["operand", "conversion", "cast", "pointer", "int"]
            }
        ],
        "null_deref": [
            {
                "name": "Pointer Safety",
                "explanation": "Dereferencing a NULL pointer crashes the program immediately. Always check if pointer != NULL before using it. A pointer is just a memory address — NULL means address zero.",
                "keywords": ["null", "pointer", "segfault", "segmentation", "dereference"]
            }
        ],
        "memory_error": [
            {
                "name": "Manual Memory Management",
                "explanation": "In C you own the memory. Every malloc needs a matching free. Every allocation needs bounds checking. Forgetting free causes memory leaks. Freeing twice corrupts the allocator.",
                "keywords": ["malloc", "free", "leak", "heap", "allocation"]
            }
        ],
        "syntax_error": [
            {
                "name": "C Syntax Rules",
                "explanation": "C requires semicolons at the end of every statement. Every opening brace needs a closing brace. Variables must be declared before use.",
                "keywords": ["semicolon", "brace", "declared", "syntax", "expected", "before", "return"]
            }
        ]
    },
    "cpp": {
        "type_error": [
            {
                "name": "C++ Type System and Operator Overloading",
                "explanation": "C++ operators are defined per type. std::string and int have no built-in + operator. Use std::to_string() to convert numbers to strings before combining.",
                "keywords": ["operator", "string", "int", "convert", "overload"]
            }
        ],
        "null_deref": [
            {
                "name": "Pointer and Reference Safety",
                "explanation": "Dereferencing a nullptr crashes immediately. Prefer references over raw pointers when possible. When using pointers always check if ptr != nullptr before use.",
                "keywords": ["nullptr", "null", "pointer", "segfault", "dereference"]
            }
        ],
        "memory_error": [
            {
                "name": "RAII and Smart Pointers",
                "explanation": "Raw pointers leak memory if you forget delete. Prefer unique_ptr and shared_ptr — they free memory automatically when they go out of scope. This is the modern C++ approach.",
                "keywords": ["new", "delete", "leak", "unique_ptr", "shared_ptr", "malloc"]
            }
        ],
        "scope_confusion": [
            {
                "name": "Variable Scope and Lifetime",
                "explanation": "Variables in C++ live only within their enclosing block. A variable declared inside an if or loop does not exist outside it. References to out-of-scope variables cause undefined behavior.",
                "keywords": ["declared", "scope", "lifetime", "block", "undefined"]
            }
        ],
        "syntax_error": [
            {
                "name": "C++ Syntax Rules",
                "explanation": "C++ requires semicolons at the end of every statement. Every opening bracket, brace, and parenthesis needs a matching closing one. For loops need exactly two semicolons inside the header.",
                "keywords": ["semicolon", "brace", "bracket", "expected", "before", "return", "syntax", "parenthesis", "for"]
            }
        ],
        "off_by_one": [
            {
                "name": "Array Bounds and Loop Ranges",
                "explanation": "Arrays in C++ start at index 0. Accessing index N on an array of size N is out of bounds. Loop conditions using < N are correct — using <= N goes one step too far.",
                "keywords": ["index", "array", "bounds", "range", "overflow"]
            }
        ]
    }
}