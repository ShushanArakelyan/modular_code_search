REGEX_DICT = {
    "remove": [
        "pop",
        "sub",
        "split",
        "clean",
        "delete",
        "del ",
        "trim",
        "strip",
        "remove",
        "wipe",
        "^(?!.*\s+else\s+.*).*\[.*\s+if\s+.*\]", # filter list comprehension
        "^(?!.*\s+else\s+.*).*\{.*\s+if\s+.*\}", # filter dict comprehension
        "return.*\[.*\:\s+]&", # filter indexes
        "return.*\[\s+\:.*]&", # filter indexes
    ],

}