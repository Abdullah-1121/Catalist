import pytest
from pydantic import ValidationError

from catalist.cards import Card

GOOD = {
    "type": "number",
    "topic": "price",
    "summary": "B says the remote should cost 12.50 euros.",
    "who": ["B"],
    "value": 12.5,
    "unit": "EUR",
    "evidence": [7, 9],
}


def test_good_card_is_accepted():
    card = Card(**GOOD)
    assert card.value == 12.5
    assert card.supersedes is None


@pytest.mark.parametrize(
    "broken",
    [
        {"type": "idea"},  # rule 1: unknown type
        {"value": None},  # rule 2: number without value
        {"evidence": []},  # rule 3: no evidence
        {"type": "disagreement", "positions": ""},  # rule 6: no positions
        {"type": "disagreement", "positions": "B: too cheap"},  # needs 2+ people
        {"type": "proposal", "supersedes": "c1"},  # only a decision can replace one
        {"who": ["Speaker B"]},  # who = speaker letters only
    ],
)
def test_broken_card_is_rejected(broken):
    with pytest.raises(ValidationError):
        Card(**(GOOD | broken))
