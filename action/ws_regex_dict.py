REGEX_DICT = {
    "remove": [
        "strip\(",
        "pop\(",
        "sub\(",
        "delete",
        "del ",
        "trim",
        "remove",
        "split\(",
        "^(?!.*\s+else\s+.*).*\[.*\s+if\s+.*\]", # filter list comprehension
    ],

}