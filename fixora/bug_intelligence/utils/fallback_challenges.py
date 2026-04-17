FALLBACK_CHALLENGES = {
    "type_error": {
        "instruction": "Something's wrong here — what is it and why does it crash?",
        "code_snippet": "price = 9.99\nprint('Total cost: ' + price)",
        "hint": "Think about what types are on each side of the + operator",
        "correct_concept": "You cannot add a string and a float directly. Use str(price) or an f-string."
    },
    "off_by_one": {
        "instruction": "This code crashes on the last step — can you see why?",
        "code_snippet": "names = ['Ali', 'Sara', 'John']\nfor i in range(4):\n    print(names[i])",
        "hint": "How many items are in the list? What indices exist?",
        "correct_concept": "A list with 3 items has indices 0, 1, 2. Index 3 does not exist."
    },
    "null_deref": {
        "instruction": "What could go wrong when this runs?",
        "code_snippet": "def get_user():\n    return None\n\nuser = get_user()\nprint(user.name)",
        "hint": "What does get_user() actually return?",
        "correct_concept": "Always check if a value is None before trying to use it."
    },
    "scope_confusion": {
        "instruction": "Why does this crash even though result looks defined?",
        "code_snippet": "def calculate():\n    result = 42\n\ncalculate()\nprint(result)",
        "hint": "Where is result defined? Where are you trying to use it?",
        "correct_concept": "Variables defined inside a function don't exist outside it."
    },
    "syntax_error": {
        "instruction": "Python refuses to run this — what's missing?",
        "code_snippet": "def greet(name)\n    print('Hello ' + name)\n\ngreet('Sara')",
        "hint": "Look carefully at the end of the first line",
        "correct_concept": "Every function definition needs a colon at the end."
    },
    "async_misuse": {
        "instruction": "This doesn't return what you'd expect — why?",
        "code_snippet": "import asyncio\n\nasync def fetch_data():\n    return 'data'\n\nresult = fetch_data()\nprint(result)",
        "hint": "What does calling an async function without await actually give you?",
        "correct_concept": "Async functions must be called with await. Without it you get a coroutine object, not the result."
    },
    "logic_error": {
        "instruction": "This runs without crashing but gives the wrong answer — why?",
        "code_snippet": "def is_adult(age):\n    if age > 18:\n        return True\n    return False\n\nprint(is_adult(18))",
        "hint": "Test the boundary case — what about exactly 18?",
        "correct_concept": "The condition should be >= 18, not > 18. Off-by-one in conditions is a common logic error."
    },
    "memory_error": {
    "instruction": "This C code compiles but causes a crash at runtime — what's wrong?",
    "code_snippet": "int *ptr = malloc(sizeof(int));\n*ptr = 42;\nfree(ptr);\nfree(ptr);",
    "hint": "How many times is free() being called on the same pointer?",
    "correct_concept": "Calling free() twice on the same pointer corrupts the allocator. Free each allocation exactly once."
},
}