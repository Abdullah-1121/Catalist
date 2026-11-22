"""Store parsed meetings in Postgres. One transaction per meeting: all or nothing."""

from dataclasses import asdict

from sqlalchemy import Engine, ForeignKey, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from catalist.parser import Meeting


class Base(DeclarativeBase):
    pass


class MeetingRow(Base):
    __tablename__ = "meetings"
    meeting_id: Mapped[str] = mapped_column(primary_key=True)
    team: Mapped[str]
    order: Mapped[int]


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
