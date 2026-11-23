import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from catalist.topics import normalise_topic


class Card(BaseModel):
    """One fact from a meeting. Fill only the fields that fit the type; leave the rest empty."""

    type: Literal["proposal", "decision", "action", "number", "disagreement"]
    topic: str = Field(
        description="Short topic name; reuse a registry topic if it fits"
    )
    summary: str = Field(description="One sentence. For an action: the task")
    who: list[Annotated[str, Field(pattern="^[A-Z]$")]] = Field(
        description="Speaker letters: proposer, decider, owner or sayer"
    )
    reason: str = Field("", description="Why (proposal, decision)")
    value: float | None = Field(None, description="Number cards: the figure")
    unit: str = Field("", description="Number cards: unit, e.g. EUR")
    positions: str = Field(
        "", description="Disagreement cards: who thinks what, and why"
    )
    supersedes: str | None = Field(
        None, description="card_id of the earlier decision this replaces"
    )
    answers: str | None = Field(
        None, description="card_id of the earlier action this reports back on"
    )
    evidence: list[int] = Field(
        min_length=1, description="Window turn numbers that support this card"
    )

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str) -> str:
        topic = normalise_topic(value)
        if not topic:
            raise ValueError(f"topic {value!r} is empty after clean-up")
        return topic

    @model_validator(mode="after")
    def check_type_rules(self) -> Self:
        if self.type == "number" and self.value is None:
            # Gemma often writes the figure only in the summary ("budget is 12.50 EUR")
            found = re.search(r"\d+(?:[.,]\d+)?", self.summary)
            if not found:
                raise ValueError("number card needs a value")
            self.value = float(found.group().replace(",", "."))
        if self.type == "disagreement" and not self.positions:
            raise ValueError("disagreement card needs positions")
        if self.type == "disagreement" and len(set(self.who)) < 2:
            raise ValueError("disagreement needs at least 2 speakers")
        if self.supersedes and self.type != "decision":
            raise ValueError("only a decision can supersede")
        return self
