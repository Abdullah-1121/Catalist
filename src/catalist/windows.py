"""Split a meeting's turns into windows small enough for one LLM call. Plain code, no AI."""

from catalist.parser import Turn


def label(turn: Turn) -> str:
    return f"[#{turn.index} {turn.speaker}] {turn.text}"


def tokens(turn: Turn) -> int:
    # Measured on our data: 3.13 chars/token (Gemma). /3 overestimates a little = safe side.
    return len(label(turn)) // 3 + 1


def windows(turns: list[Turn], max_tokens: int = 10_000) -> list[list[Turn]]:
    """Cut between turns, never inside one. A turn over budget gets a window of its own."""
    result, current, size = [], [], 0
    for turn in turns:
        if current and size + tokens(turn) > max_tokens:
            result.append(current)
            current, size = [], 0
        current.append(turn)
        size += tokens(turn)
    if current:
        result.append(current)
    return result
