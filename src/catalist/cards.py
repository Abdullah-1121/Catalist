from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def check_type_rules(self) -> Self:
        if self.type == "number" and self.value is None:
            raise ValueError("number card needs a value")
        if self.type == "disagreement" and not self.positions:
            raise ValueError("disagreement card needs positions")
        if self.supersedes and self.type != "decision":
            raise ValueError("only a decision can supersede")
        return self
