"""Topic names: format clean-up in code (D25). Meaning (synonyms) is left to the registry prompt."""

import re


def normalise_topic(raw: str) -> str:
    text = re.sub(r"\\[a-z]+", " ", raw.lower())  # LaTeX junk like \text{_}
    words = re.findall(r"[a-z0-9]+", text)
    half = len(words) // 2
    if words[:half] == words[half:]:  # the LLM sometimes writes the whole name twice
        words = words[:half]
    if words[0:1] == ["topic"] and len(words) > 1:
        words = words[1:]
    return "_".join(words)
