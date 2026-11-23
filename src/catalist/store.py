"""Store parsed meetings in Postgres. One transaction per meeting: all or nothing."""

from dataclasses import asdict

from sqlalchemy import Engine, ForeignKey, delete
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from catalist.parser import Meeting


class Base(DeclarativeBase):
    pass


class MeetingRow(Base):
    __tablename__ = "meetings"
    meeting_id: Mapped[str] = mapped_column(primary_key=True)
    team: Mapped[str]
    order: Mapped[int]
    content_hash: Mapped[
        str | None
    ]  # sha256 of the file; unchanged -> skip extraction (D26)


class TurnRow(Base):
    __tablename__ = "turns"
    turn_id: Mapped[str] = mapped_column(primary_key=True)
    # indexed: Postgres does not index foreign keys itself, and we delete/read by meeting
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.meeting_id"), index=True
    )
    index: Mapped[int]
    start: Mapped[str]
    speaker: Mapped[str]
    text: Mapped[str]  # not Optional -> NOT NULL


class LLMCacheRow(Base):
    __tablename__ = "llm_cache"
    # sha256 of (model settings + prompt): always 64 chars. The prompt itself (a whole
    # meeting) is too big for a Postgres index entry (max 8191 bytes).
    key: Mapped[str] = mapped_column(primary_key=True)
    response: Mapped[str]


class CardRow(Base):
    """An extracted fact (append-only, D20). card_id = "TS3012b-7"."""

    __tablename__ = "cards"
    card_id: Mapped[str] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(index=True)
    team: Mapped[str] = mapped_column(index=True)
    order: Mapped[int]  # meeting order a=1..d=4, for timelines
    type: Mapped[str]
    topic: Mapped[str]
    summary: Mapped[str]
    who: Mapped[list[str]] = mapped_column(JSONB)
    reason: Mapped[str]
    value: Mapped[float | None]
    unit: Mapped[str]
    positions: Mapped[str]
    supersedes: Mapped[str | None]
    answers: Mapped[str | None]
    evidence: Mapped[list[str]] = mapped_column(JSONB)  # turn_ids


class RejectedCardRow(Base):
    """Cards the checks refused (D23): kept with the reason, never silently dropped."""

    __tablename__ = "rejected_cards"
    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[str] = mapped_column(index=True)
    window: Mapped[int]
    reason: Mapped[str]
    raw: Mapped[dict] = mapped_column(JSONB)


def store_meeting(engine: Engine, meeting: Meeting) -> None:
    """Replace the meeting and all its turns. Safe to re-run (idempotent)."""
    with (
        Session(engine) as session,
        session.begin(),
    ):  # commit at end, rollback on error
        session.execute(delete(TurnRow).where(TurnRow.meeting_id == meeting.meeting_id))
        session.merge(
            MeetingRow(
                meeting_id=meeting.meeting_id, team=meeting.team, order=meeting.order
            )
        )
        session.add_all(TurnRow(**asdict(turn)) for turn in meeting.turns)
