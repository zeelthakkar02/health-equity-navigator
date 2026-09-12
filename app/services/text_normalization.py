"""Shared lightweight text normalisation.

Used by the offline hashing embeddings and by need detection, so a query and a
resource description are reduced the same way in both places. Deliberately crude
and dependency-free — this is not linguistics, it is just enough normalisation
for plurals and common verb forms to meet.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and would otherwise dominate the
# short queries community members actually type.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can do does for from had has have he her him his
    i if in into is it its me my need needs of on or our she that the their them then
    there these they this to us was we were what when where which who will with you your
    """.split()
)

# Stripping a suffix must leave a real word behind. Without this floor 'caring'
# becomes 'car', which would make a caregiving query look like a request for a
# ride — exactly the kind of accidental match this module exists to avoid.
_MIN_STEM_LENGTH = 4


def stem(token: str) -> str:
    """Crude suffix stripper so 'rides' and 'ride' share a feature.

    Rules are applied in order and at most one fires, plurals first — stripping
    'es' from 'rides' before the simpler 's' rule would yield 'rid', which would
    never meet the 'ride' in a resource description.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 7 and token.endswith("ation"):
        return token[:-5]
    if len(token) > 5 and token.endswith("ing") and len(token) - 3 >= _MIN_STEM_LENGTH:
        return token[:-3]
    if len(token) > 4 and token.endswith("ed") and len(token) - 2 >= _MIN_STEM_LENGTH:
        return token[:-2]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords, and stem."""
    return [
        stem(token)
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def normalize(text: str) -> str:
    """Space-padded stem sequence, so phrases can be matched by substring."""
    tokens = tokenize(text)
    return f" {' '.join(tokens)} " if tokens else " "
