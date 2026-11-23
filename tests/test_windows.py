from catalist.parser import Turn
from catalist.windows import tokens, windows


def make(sizes):
    return [
        Turn(f"M#{i}", "M", i, "00:00:00", "A", "x" * n) for i, n in enumerate(sizes)
    ]


def test_short_meeting_is_one_window():
    turns = make([30, 30, 30])
    assert windows(turns, max_tokens=10_000) == [turns]


def test_long_meeting_is_cut_between_turns_within_budget():
    turns = make([90] * 10)  # ~32 tokens each, ~320 total
    result = windows(turns, max_tokens=100)
    assert len(result) > 1
    assert [t for w in result for t in w] == turns  # every turn once, in order
    assert all(sum(tokens(t) for t in w) <= 100 for w in result)


def test_giant_turn_gets_its_own_window():
    turns = make([30, 1000, 30])  # middle turn alone is over budget
    assert windows(turns, max_tokens=100) == [[turns[0]], [turns[1]], [turns[2]]]
