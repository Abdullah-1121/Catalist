"""Citations are made by code, never trusted from the LLM (D2). Scoring: >= 6 words, copied
exactly from one turn; case and spacing ignored."""

import re
from difflib import SequenceMatcher

MIN_WORDS = 6


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _bare(word: str) -> str:
    return re.sub(r"\W", "", word.lower())  # for scoring only: "screen," == "screen"


def is_exact(quote: str, turn_text: str) -> bool:
    return len(quote.split()) >= MIN_WORDS and _norm(quote) in _norm(turn_text)


def best_quote(claim: str, turn_text: str, max_words: int = 20) -> str | None:
    """The claim itself if it is exact, else the run of real words in the turn closest to it."""
    if is_exact(claim, turn_text):
        return claim
    words = turn_text.split()
    if len(words) < MIN_WORDS:
        return None
    size = min(max(len(claim.split()), MIN_WORDS), max_words, len(words))
    target = [_bare(w) for w in claim.split()]

    def score(start: int) -> float:
        span = [_bare(w) for w in words[start : start + size]]
        return SequenceMatcher(None, span, target).ratio()

    start = max(range(len(words) - size + 1), key=score)
    return " ".join(words[start : start + size])
