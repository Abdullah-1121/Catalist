import pytest

from catalist.cards import Card
from catalist.extract import check_card, live_cards, registry_text
from catalist.store import CardRow

BASE = {"topic": "colour", "summary": "s", "who": ["A"], "evidence": [5]}
KNOWN = {"M-1": "decision", "M-2": "action", "M-3": "proposal"}


@pytest.mark.parametrize(
    "fields, problem",
    [
        ({"type": "proposal"}, None),
        ({"type": "proposal", "evidence": [5, 99]}, "evidence"),
        ({"type": "decision", "supersedes": "M-1"}, None),
        ({"type": "decision", "supersedes": "M-9"}, "supersedes"),  # unknown id
        ({"type": "decision", "supersedes": "M-3"}, "supersedes"),  # not a decision
        ({"type": "proposal", "answers": "M-2"}, None),
        ({"type": "proposal", "answers": "M-1"}, "answers"),  # not an action
    ],
)
def test_check_card(fields, problem):
    error = check_card(Card(**BASE | fields), {4, 5, 6}, KNOWN)
    assert (error is None) if problem is None else (problem in error)


def row(card_id, type, summary, supersedes=None, answers=None, topic="colour"):
    return CardRow(
        card_id=card_id,
        meeting_id="M",
        team="T",
        order=1,
        type=type,
        topic=topic,
        summary=summary,
        who=["A"],
        reason="",
        value=None,
        unit="",
        positions="",
        supersedes=supersedes,
        answers=answers,
        evidence=["M#5"],
    )


def test_superseded_decision_and_answered_action_are_not_live():
    cards = [
        row("M-1", "decision", "yellow"),
        row("M-2", "decision", "red", supersedes="M-1"),
        row("M-3", "action", "find battery price", topic="battery"),
        row("M-4", "number", "2 euros", answers="M-3", topic="battery"),
    ]
    assert [c.card_id for c in live_cards(cards)] == ["M-2", "M-4"]
    text = registry_text(cards)
    assert "red" in text and "yellow" not in text
