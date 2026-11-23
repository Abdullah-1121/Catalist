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


def test_number_value_is_read_from_summary_when_missing():
    card = Card(
        **GOOD | {"value": None, "summary": "The budget is 12.50 EUR per unit."}
    )
    assert card.value == 12.5


def test_topic_is_normalised():
    assert Card(**GOOD | {"topic": "topic_Casing Colour"}).topic == "casing_colour"


@pytest.mark.parametrize(
    "broken",
    [
        {"type": "idea"},  # rule 1: unknown type
        {"value": None, "summary": "no figure here"},  # rule 2: number without value
        {"evidence": []},  # rule 3: no evidence
        {"type": "disagreement", "positions": ""},  # rule 6: no positions
        {"type": "disagreement", "positions": "B: too cheap"},  # needs 2+ people
        {"type": "disagreement", "positions": "x", "who": ["B", "B"]},  # same person
        {"type": "proposal", "supersedes": "c1"},  # only a decision can replace one
        {"who": ["Speaker B"]},  # who = speaker letters only
        {"topic": "การ์ด"},  # junk topic: nothing left after clean-up
    ],
)
def test_broken_card_is_rejected(broken):
    with pytest.raises(ValidationError):
        Card(**(GOOD | broken))
